import csv
import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from emotion_classifier.config import LLMConfig, ZeroShotConfig
from emotion_classifier.data import LABELS, DatasetSplits
from emotion_classifier.device import pick_device
from emotion_classifier.metrics import compute_metrics
from emotion_classifier.results import RunResult, make_run_id, save_predictions, save_result
from emotion_classifier.timing import Timer

def labels_to_ids( names: list[ str ] ) -> list[ int ]:
    """Label names (pipeline output) -> ids (index in data.LABELS)."""
    ids = []
    for name in names:
        if name not in LABELS:
            raise ValueError( f"Unknown label { repr( name ) }; expected one of { LABELS }" )
        ids.append( LABELS.index( name ) )
    return ids

def top_label_ids( outputs: dict | list[ dict ] ) -> list[ int ]:
    """Highest-scoring label id per pipeline result (picked by score, not by list position)."""
    if isinstance( outputs, dict ):
        outputs = [ outputs ]
    top = [ max( zip( output[ "scores" ], output[ "labels" ] ) )[ 1 ] for output in outputs ]
    return labels_to_ids( top )

def build_pipeline( cfg: ZeroShotConfig, device: str ) -> Callable[ ..., Any ]:
    """HF zero-shot NLI pipeline (imported lazily: transformers is slow to import)."""
    from transformers import pipeline

    return pipeline( "zero-shot-classification", model = cfg.model_name, device = device )

def predict_zeroshot(
    texts: list[ str ],
    cfg: ZeroShotConfig,
    classifier: Callable[ ..., Any ] | None = None,
    device: str = "cpu",
) -> tuple[ list[ int ], float ]:
    """Single-label zero-shot predictions, and inference seconds per 1,000 examples.
    Labels are the six emotion names and the hypothesis template is fixed by the config."""
    if not texts:
        return [], 0.0

    classifier = classifier or build_pipeline( cfg, device )
    with Timer() as timer:
        outputs = classifier(
            list( texts ),
            candidate_labels = LABELS,
            hypothesis_template = cfg.hypothesis_template,
            batch_size = cfg.batch_size,
            multi_label = False,
        )
    return top_label_ids( outputs ), timer.seconds / len( texts ) * 1000

def run_zeroshot_experiment(
    cfg: ZeroShotConfig,
    splits: DatasetSplits,
    config_name: str,
    results_dir: Path,
    timestamp: str,
    git_sha: str | None = None,
    classifier: Callable[ ..., Any ] | None = None,
    device: str | None = None,
) -> list[ RunResult ]:
    """Evaluate on the test split only (TRD §4.5: zero-shot has no tuning), saving a RunResult
    + predictions CSV. `config_name` is the config file stem and becomes RunResult.model."""
    random.seed( cfg.seed )
    np.random.seed( cfg.seed )

    device = device or pick_device()
    classifier = classifier or build_pipeline( cfg, device )
    predictions, infer_per_1k = predict_zeroshot( splits.test_texts, cfg, classifier = classifier )

    model = getattr( classifier, "model", None )
    n_params = sum( p.numel() for p in model.parameters() ) if model is not None else None

    run_id = make_run_id( cfg.experiment, config_name, "test", cfg.seed, timestamp )
    preds_path = save_predictions( run_id, splits.test_texts, splits.test_labels, predictions, results_dir )
    result = RunResult(
        run_id = run_id,
        experiment = cfg.experiment,
        model = config_name,
        config = json.loads( json.dumps( asdict( cfg ) ) ),
        split = "test",
        metrics = compute_metrics( splits.test_labels, predictions ),
        train_seconds = None,
        infer_seconds_per_1k = infer_per_1k,
        n_params = n_params,
        device = device,
        seed = cfg.seed,
        git_sha = git_sha,
        predictions_path = str( preds_path.relative_to( results_dir ) ),
    )
    save_result( result, results_dir )
    return [ result ]

# ---- E5: zero-shot instruct LLM ------------------------------------------------------------------

INVALID = -1  # prediction stored when the model's answer is not exactly one label

def build_prompt( text: str, labels: list[ str ] ) -> str:
    """The user message for one tweet. Fixed for every example; a copy with a placeholder tweet is saved in the RunResult config."""
    return (
        "Classify the emotion expressed in the tweet below. "
        f"Choose exactly one of these labels: { ', '.join( labels ) }. "
        "Answer with only the label, in lowercase, and nothing else.\n\n"
        f"Tweet: { text }\n"
        "Label:"
    )

def parse_label( raw: str, labels: list[ str ] ) -> int | None:
    """Label id if the output contains exactly one distinct label as a whole word (any case, punctuation ignored).
    Synonyms and word forms ('happiness', 'surprised'), several labels, or no label give None."""
    text = raw.lower()
    found = [ index for index, label in enumerate( labels ) if re.search( rf"(?<![a-z]){ re.escape( label.lower() ) }(?![a-z])", text ) ]
    return found[ 0 ] if len( found ) == 1 else None

class HFGenerator:
    """Greedy chat generation with a local Hugging Face causal LM. Called with a batch of prompts, returns the new text per prompt."""

    def __init__( self, cfg: LLMConfig, device: str ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained( cfg.model_name, padding_side = "left" )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype = torch.float32 if device == "cpu" else torch.bfloat16
        ).to( device ).eval()
        self.device, self.max_new_tokens = device, cfg.max_new_tokens

    def __call__( self, prompts: list[ str ] ) -> list[ str ]:
        import torch

        templated = bool( getattr( self.tokenizer, "chat_template", None ) )
        texts = [
            self.tokenizer.apply_chat_template( [ { "role": "user", "content": prompt } ], tokenize = False, add_generation_prompt = True )
            if templated else prompt
            for prompt in prompts
        ]
        inputs = self.tokenizer( texts, return_tensors = "pt", padding = True, add_special_tokens = not templated ).to( self.device )
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens = self.max_new_tokens, do_sample = False,  # greedy = temperature 0
                temperature = None, top_p = None, top_k = None, pad_token_id = self.tokenizer.pad_token_id,
            )
        return self.tokenizer.batch_decode( generated[ :, inputs[ "input_ids" ].shape[ 1 ]: ], skip_special_tokens = True )

def build_generator( cfg: LLMConfig, device: str ) -> Callable[ [ list[ str ] ], list[ str ] ]:
    return HFGenerator( cfg, device )

def predict_llm(
    texts: list[ str ],
    cfg: LLMConfig,
    generator: Callable[ [ list[ str ] ], list[ str ] ] | None = None,
    device: str = "cpu",
) -> tuple[ list[ str ], float ]:
    """Raw model output per text (input order) and inference seconds per 1,000 examples. Prompts go to the generator in batches."""
    if not texts:
        return [], 0.0

    generator = generator or build_generator( cfg, device )
    prompts = [ build_prompt( text, LABELS ) for text in texts ]
    outputs: list[ str ] = []
    with Timer() as timer:
        for start in range( 0, len( prompts ), cfg.batch_size ):
            batch = prompts[ start:start + cfg.batch_size ]
            batch_outputs = generator( batch )
            if len( batch_outputs ) != len( batch ):
                raise ValueError( f"generator returned { len( batch_outputs ) } outputs for { len( batch ) } prompts" )
            outputs.extend( batch_outputs )
    return outputs, timer.seconds / len( texts ) * 1000

def save_raw_outputs( run_id: str, texts: list[ str ], raws: list[ str ], results_dir: Path ) -> Path:
    """results/preds/<run_id>.raw.csv with columns text,raw, so the parser can be audited. Never overwrites."""
    preds_dir = Path( results_dir ) / "preds"
    preds_dir.mkdir( parents = True, exist_ok = True )
    path = preds_dir / f"{ run_id }.raw.csv"
    with path.open( "x", encoding = "utf-8", newline = "" ) as handle:
        writer = csv.writer( handle, lineterminator = "\n" )
        writer.writerow( [ "text", "raw" ] )
        writer.writerows( zip( texts, raws ) )
    return path

def run_llm_experiment(
    cfg: LLMConfig,
    splits: DatasetSplits,
    config_name: str,
    results_dir: Path,
    timestamp: str,
    git_sha: str | None = None,
    generator: Callable[ [ list[ str ] ], list[ str ] ] | None = None,
    device: str | None = None,
) -> list[ RunResult ]:
    """Evaluate on test only (no tuning). An output that is not exactly one label counts as wrong (prediction -1)
    and is reported as `invalid_rate` and `n_invalid` in the metrics."""
    random.seed( cfg.seed )
    np.random.seed( cfg.seed )

    device = device or pick_device()
    generator = generator or build_generator( cfg, device )
    raws, infer_per_1k = predict_llm( splits.test_texts, cfg, generator = generator )

    parsed = [ parse_label( raw, LABELS ) for raw in raws ]
    predictions = [ INVALID if label is None else label for label in parsed ]
    n_invalid = predictions.count( INVALID )
    metrics = compute_metrics( splits.test_labels, predictions )
    metrics[ "n_invalid" ] = n_invalid
    metrics[ "invalid_rate" ] = n_invalid / len( predictions ) if predictions else 0.0

    model = getattr( generator, "model", None )
    n_params = sum( p.numel() for p in model.parameters() ) if model is not None else None

    config = json.loads( json.dumps( asdict( cfg ) ) )
    config[ "prompt_example" ] = build_prompt( "<tweet>", LABELS )
    config[ "decoding" ] = "greedy (temperature 0)"

    run_id = make_run_id( cfg.experiment, config_name, "test", cfg.seed, timestamp )
    preds_path = save_predictions( run_id, splits.test_texts, splits.test_labels, predictions, results_dir )
    save_raw_outputs( run_id, splits.test_texts, raws, results_dir )
    result = RunResult(
        run_id = run_id,
        experiment = cfg.experiment,
        model = config_name,
        config = config,
        split = "test",
        metrics = metrics,
        train_seconds = None,
        infer_seconds_per_1k = infer_per_1k,
        n_params = n_params,
        device = device,
        seed = cfg.seed,
        git_sha = git_sha,
        predictions_path = str( preds_path.relative_to( results_dir ) ),
    )
    save_result( result, results_dir )
    return [ result ]
