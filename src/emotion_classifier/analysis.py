import csv
import random
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from emotion_classifier.data import LABELS

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
