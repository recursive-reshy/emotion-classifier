import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from emotion_classifier.config import ZeroShotConfig
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
