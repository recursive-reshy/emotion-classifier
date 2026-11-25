import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from emotion_classifier.config import TransformerConfig
from emotion_classifier.data import LABELS, DatasetSplits
from emotion_classifier.device import pick_device
from emotion_classifier.metrics import compute_metrics
from emotion_classifier.results import RunResult, make_run_id, save_predictions, save_result
from emotion_classifier.timing import Timer

@dataclass( frozen = True )
class FineTuneOutput:
    model: object
    tokenizer: object
    val_pred: list[ int ]
    test_pred: list[ int ]
    train_seconds: float
    val_infer_seconds_per_1k: float
    test_infer_seconds_per_1k: float
    n_params: int
    device: str

class _EncodedDataset( torch.utils.data.Dataset ):
    """Tokenised examples, unpadded; DataCollatorWithPadding pads each batch."""

    def __init__( self, encoded, labels: list[ int ] | None = None ):
        self.encoded = encoded
        self.labels = labels

    def __len__( self ) -> int:
        return len( self.encoded[ "input_ids" ] )

    def __getitem__( self, index: int ) -> dict:
        item = { key: self.encoded[ key ][ index ] for key in ( "input_ids", "attention_mask" ) }
        if self.labels is not None:
            item[ "labels" ] = self.labels[ index ]
        return item

def label_maps() -> tuple[ dict[ int, str ], dict[ str, int ] ]:
    """( id2label, label2id ) derived from data.LABELS."""
    return dict( enumerate( LABELS ) ), { name: index for index, name in enumerate( LABELS ) }

def tokenize( tokenizer, texts: list[ str ], max_length: int ):
    """Truncate to max_length; no padding here (dynamic padding at batch time)."""
    return tokenizer( list( texts ), truncation = True, max_length = max_length )

def build_model( model_name: str ):
    id2label, label2id = label_maps()
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels = len( LABELS ), id2label = id2label, label2id = label2id,
        ignore_mismatched_sizes = True,  # the 6-way head is always freshly initialised
    )

def count_params( model ) -> int:
    return sum( p.numel() for p in model.parameters() )

def predict( model, tokenizer, texts: list[ str ], max_length: int, batch_size: int, device: str ) -> tuple[ list[ int ], float ]:
    """Argmax predictions in input order, and inference seconds per 1,000 examples."""
    dataset = _EncodedDataset( tokenize( tokenizer, texts, max_length ) )
    collator = DataCollatorWithPadding( tokenizer )
    loader = torch.utils.data.DataLoader( dataset, batch_size = batch_size, shuffle = False, collate_fn = collator )

    model.to( device ).eval()
    predictions: list[ int ] = []
    with Timer() as timer, torch.inference_mode():
        for batch in loader:
            batch = { key: value.to( device ) for key, value in batch.items() }
            predictions.extend( model( **batch ).logits.argmax( dim = -1 ).cpu().tolist() )
    return predictions, timer.seconds / max( len( texts ), 1 ) * 1000

def _trainer_metrics( eval_prediction ) -> dict[ str, float ]:
    logits, labels = eval_prediction.predictions, eval_prediction.label_ids
    metrics = compute_metrics( labels.tolist(), np.argmax( logits, axis = -1 ).tolist() )
    return { "macro_f1": metrics[ "macro_f1" ], "accuracy": metrics[ "accuracy" ] }

def fine_tune( splits: DatasetSplits, cfg: TransformerConfig, seed: int, output_dir: Path ) -> FineTuneOutput:
    """Fine-tune on train; the best epoch by validation macro-F1 is restored (early stopping).
    Predicts val and test with that checkpoint. Test is never used for selection."""
    random.seed( seed )
    np.random.seed( seed )
    set_seed( seed )

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained( cfg.model_name )
    model = build_model( cfg.model_name )

    train_set = _EncodedDataset( tokenize( tokenizer, splits.train_texts, cfg.max_length ), splits.train_labels )
    val_set = _EncodedDataset( tokenize( tokenizer, splits.val_texts, cfg.max_length ), splits.val_labels )

    args = TrainingArguments(
        output_dir = str( output_dir ),
        learning_rate = cfg.lr,
        num_train_epochs = cfg.epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size = cfg.batch_size,
        weight_decay = cfg.weight_decay,
        warmup_steps = cfg.warmup_ratio,  # transformers 5.x: a float in [0, 1) is a ratio of total steps
        eval_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "macro_f1",
        greater_is_better = True,
        save_total_limit = 2,
        seed = seed,
        report_to = "none",
        logging_strategy = "epoch",
    )
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_set,
        eval_dataset = val_set,
        data_collator = DataCollatorWithPadding( tokenizer ),
        processing_class = tokenizer,
        compute_metrics = _trainer_metrics,
        callbacks = [ EarlyStoppingCallback( early_stopping_patience = cfg.early_stopping_patience ) ],
    )

    with Timer() as train_timer:
        trainer.train()

    model = trainer.model
    val_pred, val_per_1k = predict( model, tokenizer, splits.val_texts, cfg.max_length, cfg.batch_size, device )
    test_pred, test_per_1k = predict( model, tokenizer, splits.test_texts, cfg.max_length, cfg.batch_size, device )
    return FineTuneOutput(
        model = model,
        tokenizer = tokenizer,
        val_pred = val_pred,
        test_pred = test_pred,
        train_seconds = train_timer.seconds,
        val_infer_seconds_per_1k = val_per_1k,
        test_infer_seconds_per_1k = test_per_1k,
        n_params = count_params( model ),
        device = device,
    )

FineTuner = Callable[ [ DatasetSplits, TransformerConfig, int, Path ], FineTuneOutput ]

def run_transformer_experiment(
    cfg: TransformerConfig,
    splits: DatasetSplits,
    config_name: str,
    results_dir: Path,
    output_dir: Path,
    timestamp: str,
    git_sha: str | None = None,
    splits_to_save: str = "both",
    fine_tuner: FineTuner | None = None,
) -> list[ RunResult ]:
    """Fine-tune once per seed in cfg.seeds; save a RunResult + predictions CSV per seed per split.
    Checkpoints go to output_dir / config_name / seed<N>. `config_name` becomes RunResult.model."""
    if splits_to_save not in ( "val", "test", "both" ):
        raise ValueError( f"splits_to_save must be val, test or both, got { repr( splits_to_save ) }" )

    fine_tuner = fine_tuner or fine_tune
    wanted = [ "val", "test" ] if splits_to_save == "both" else [ splits_to_save ]
    config = json.loads( json.dumps( asdict( cfg ) ) )  # JSON-safe, so in-memory == saved

    results = []
    for seed in cfg.seeds:
        output = fine_tuner( splits, cfg, seed, Path( output_dir ) / config_name / f"seed{ seed }" )
        per_split = {
            "val": ( splits.val_texts, splits.val_labels, output.val_pred, output.val_infer_seconds_per_1k ),
            "test": ( splits.test_texts, splits.test_labels, output.test_pred, output.test_infer_seconds_per_1k ),
        }
        for split in wanted:
            texts, labels, predictions, infer_per_1k = per_split[ split ]
            run_id = make_run_id( cfg.experiment, config_name, split, seed, timestamp )
            preds_path = save_predictions( run_id, texts, labels, predictions, results_dir )
            result = RunResult(
                run_id = run_id,
                experiment = cfg.experiment,
                model = config_name,
                config = config,
                split = split,
                metrics = compute_metrics( labels, predictions ),
                train_seconds = output.train_seconds,
                infer_seconds_per_1k = infer_per_1k,
                n_params = output.n_params,
                device = output.device,
                seed = seed,
                git_sha = git_sha,
                predictions_path = str( preds_path.relative_to( results_dir ) ),
            )
            save_result( result, results_dir )
            results.append( result )
    return results
