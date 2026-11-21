import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from emotion_classifier.config import BASELINE_STRATEGIES, BaselineConfig, ClassicalConfig
from emotion_classifier.data import DatasetSplits
from emotion_classifier.features import build_vectorizer
from emotion_classifier.metrics import compute_metrics
from emotion_classifier.results import RunResult, make_run_id, save_predictions, save_result
from emotion_classifier.timing import Timer

# Which hyperparameter each classifier tunes (TRD §5).
TUNING_PARAMS = { "nb": "alpha", "logreg": "C", "svm": "C" }

Scorer = Callable[ [ ClassicalConfig, float, DatasetSplits ], float ]

@dataclass( frozen = True )
class FitOutput:
    model: object
    val_pred: list[ int ]
    test_pred: list[ int ]
    train_seconds: float
    val_infer_seconds_per_1k: float
    test_infer_seconds_per_1k: float

def build_baseline( cfg: BaselineConfig ) -> DummyClassifier:
    if cfg.strategy not in BASELINE_STRATEGIES:
        raise ValueError( f"Unknown baseline strategy { repr( cfg.strategy ) }; expected one of { list( BASELINE_STRATEGIES ) }" )
    return DummyClassifier( strategy = cfg.strategy, random_state = cfg.seed )

def build_model( cfg: ClassicalConfig, value: float ) -> Pipeline:
    """Vectorizer + classifier, with the tuned hyperparameter set to `value`."""
    kind = cfg.classifier.kind
    if cfg.tuning.param != TUNING_PARAMS[ kind ]:
        raise ValueError( f"Classifier '{ kind }' tunes '{ TUNING_PARAMS[ kind ] }', but tuning.param is '{ cfg.tuning.param }'" )
    if kind == "nb" and cfg.classifier.class_weight is not None:
        raise ValueError( "class_weight is not supported by MultinomialNB" )

    if kind == "nb":
        classifier = MultinomialNB( alpha = value )
    elif kind == "logreg":
        classifier = LogisticRegression( C = value, max_iter = 2000, class_weight = cfg.classifier.class_weight, random_state = cfg.seed )
    else:
        classifier = LinearSVC( C = value, class_weight = cfg.classifier.class_weight, random_state = cfg.seed )

    vectorizer = build_vectorizer(
        cfg.vectorizer.kind,
        cfg.vectorizer.ngram_range,
        cfg.vectorizer.min_df,
        cfg.vectorizer.max_features,
        cfg.vectorizer.stop_words,
    )
    return Pipeline( [ ( "vectorizer", vectorizer ), ( "classifier", classifier ) ] )

def _val_macro_f1( cfg: ClassicalConfig, value: float, splits: DatasetSplits ) -> float:
    model = build_model( cfg, value )
    model.fit( splits.train_texts, splits.train_labels )
    return compute_metrics( splits.val_labels, list( model.predict( splits.val_texts ) ) )[ "macro_f1" ]

def tune_on_validation(
    splits: DatasetSplits,
    cfg: ClassicalConfig,
    scorer: Scorer | None = None,
) -> tuple[ float, dict[ float, float ] ]:
    """Manual loop over cfg.tuning.values, scored by validation macro-F1 (never touches test).
    Ties go to the first value in the config."""
    scorer = scorer or _val_macro_f1
    val_scores = { value: scorer( cfg, value, splits ) for value in cfg.tuning.values }
    best = max( cfg.tuning.values, key = lambda value: val_scores[ value ] )
    return best, val_scores

def _predict_timed( model, texts: list[ str ] ) -> tuple[ list[ int ], float ]:
    with Timer() as timer:
        predictions = [ int( label ) for label in model.predict( texts ) ]
    return predictions, timer.seconds / max( len( texts ), 1 ) * 1000

def fit_predict( splits: DatasetSplits, model ) -> FitOutput:
    """Fit on train once, predict val and test."""
    with Timer() as train_timer:
        model.fit( splits.train_texts, splits.train_labels )
    val_pred, val_per_1k = _predict_timed( model, splits.val_texts )
    test_pred, test_per_1k = _predict_timed( model, splits.test_texts )
    return FitOutput( model, val_pred, test_pred, train_timer.seconds, val_per_1k, test_per_1k )

def run_experiment(
    cfg: ClassicalConfig | BaselineConfig,
    splits: DatasetSplits,
    config_name: str,
    results_dir: Path,
    timestamp: str,
    git_sha: str | None = None,
    splits_to_save: str = "both",
    scorer: Scorer | None = None,
) -> list[ RunResult ]:
    """Tune (classical only) on val, fit on train, save a RunResult + predictions CSV per split.
    `config_name` is the config file stem and becomes RunResult.model."""
    if splits_to_save not in ( "val", "test", "both" ):
        raise ValueError( f"splits_to_save must be val, test or both, got { repr( splits_to_save ) }" )

    random.seed( cfg.seed )
    np.random.seed( cfg.seed )

    config = json.loads( json.dumps( asdict( cfg ) ) )  # JSON-safe, so in-memory == saved
    if isinstance( cfg, ClassicalConfig ):
        value, val_scores = tune_on_validation( splits, cfg, scorer )
        model = build_model( cfg, value )
        config[ "tuned" ] = {
            "param": cfg.tuning.param,
            "value": value,
            "val_scores": { str( key ): score for key, score in val_scores.items() },
        }
    else:
        model = build_baseline( cfg )

    output = fit_predict( splits, model )

    per_split = {
        "val": ( splits.val_texts, splits.val_labels, output.val_pred, output.val_infer_seconds_per_1k ),
        "test": ( splits.test_texts, splits.test_labels, output.test_pred, output.test_infer_seconds_per_1k ),
    }
    wanted = [ "val", "test" ] if splits_to_save == "both" else [ splits_to_save ]

    results = []
    for split in wanted:
        texts, labels, predictions, infer_per_1k = per_split[ split ]
        run_id = make_run_id( cfg.experiment, config_name, split, cfg.seed, timestamp )
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
            n_params = None,
            device = "cpu",
            seed = cfg.seed,
            git_sha = git_sha,
            predictions_path = str( preds_path.relative_to( results_dir ) ),
        )
        save_result( result, results_dir )
        results.append( result )
    return results
