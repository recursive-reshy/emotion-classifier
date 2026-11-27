import csv
import random
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from emotion_classifier.classical import build_model
from emotion_classifier.config import ClassicalConfig, config_from_dict
from emotion_classifier.data import LABELS, DatasetSplits
from emotion_classifier.report import latest_runs
from emotion_classifier.results import RunResult

LINEAR_KINDS = ( "logreg", "svm" )

def _is_linear_classical( run: RunResult ) -> bool:
    return run.config.get( "family" ) == "classical" and run.config.get( "classifier", {} ).get( "kind" ) in LINEAR_KINDS

def best_linear_run( results: list[ RunResult ] ) -> RunResult:
    """The linear classical model (logreg or svm) with the best validation macro-F1: its validation run.
    Selected on validation only; test results are ignored."""
    candidates = [ run for run in latest_runs( results, "val" ) if _is_linear_classical( run ) ]
    if not candidates:
        raise ValueError( "No validation results for a linear classical model (logreg or svm)" )
    return max( sorted( candidates, key = lambda run: run.run_id ), key = lambda run: run.metrics[ "macro_f1" ] )

def refit_linear( run: RunResult, splits: DatasetSplits ) -> Pipeline:
    """Rebuild the pipeline a classical run used (with the C it tuned on validation) and fit it on the train split.
    Training is deterministic, so this is the model behind the run's predictions."""
    if not _is_linear_classical( run ):
        raise ValueError( f"{ run.model } is not a linear classical model (logreg or svm)" )
    cfg = config_from_dict( { key: value for key, value in run.config.items() if key != "tuned" }, run.run_id )
    assert isinstance( cfg, ClassicalConfig )
    return build_model( cfg, run.config[ "tuned" ][ "value" ] ).fit( splits.train_texts, splits.train_labels )

def top_features( pipeline: Pipeline, k: int = 15 ) -> dict[ str, list[ tuple[ str, float ] ] ]:
    """Per class, the k n-grams with the largest coefficients, best first: { label: [ ( ngram, weight ) ] }.
    Needs a linear classifier (`coef_`): logistic regression or linear SVM."""
    classifier = pipeline.named_steps[ "classifier" ]
    if not hasattr( classifier, "coef_" ):
        raise ValueError( f"{ type( classifier ).__name__ } has no coef_; top_features needs a linear model (logreg or svm)" )

    names = pipeline.named_steps[ "vectorizer" ].get_feature_names_out()
    features = {}
    for class_id, weights in zip( classifier.classes_, classifier.coef_ ):
        best = np.argsort( -weights, kind = "stable" )[ :k ]
        features[ LABELS[ int( class_id ) ] ] = [ ( str( names[ index ] ), float( weights[ index ] ) ) for index in best ]
    return features

def sample_errors( preds_csv: Path, true: int, pred: int, n: int = 5, seed: int = 42 ) -> list[ str ]:
    """Up to n texts whose true label is `true` but were predicted `pred` (ids), reproducible for a seed.
    Reads the text,true,pred CSV written with each run; returns the sample in file order."""
    if true == pred:
        raise ValueError( "true and pred must differ: a confusion pair is a misclassification" )

    with Path( preds_csv ).open( encoding = "utf-8", newline = "" ) as handle:
        matching = [ row[ "text" ] for row in csv.DictReader( handle ) if int( row[ "true" ] ) == true and int( row[ "pred" ] ) == pred ]

    chosen = random.Random( seed ).sample( range( len( matching ) ), min( n, len( matching ) ) )
    return [ matching[ index ] for index in sorted( chosen ) ]

def confusion_pairs( confusion: list[ list[ int ] ], k: int = 5 ) -> list[ tuple[ int, int, int ] ]:
    """The k largest off-diagonal cells as ( true_id, pred_id, count ), biggest first.
    Ties go to the lower ( true, pred ); zero cells are skipped."""
    cells = [
        ( true_id, pred_id, int( count ) )
        for true_id, row in enumerate( confusion )
        for pred_id, count in enumerate( row )
        if true_id != pred_id and count > 0
    ]
    cells.sort( key = lambda cell: ( -cell[ 2 ], cell[ 0 ], cell[ 1 ] ) )
    return cells[ :k ]
