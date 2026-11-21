import json

import pytest

from emotion_classifier.data import LABELS
from emotion_classifier.metrics import compute_metrics, row_normalise

ABC = [ "a", "b", "c" ]

# index: 0 1 2 3 4 5
Y_TRUE = [ 0, 0, 0, 1, 1, 2 ]
Y_PRED = [ 0, 0, 1, 1, 2, 2 ]
# class a: TP 2, FP 0, FN 1 -> P 1.0, R 2/3, F1 0.8
# class b: TP 1, FP 1, FN 1 -> P 0.5, R 0.5, F1 0.5
# class c: TP 1, FP 1, FN 0 -> P 0.5, R 1.0, F1 2/3
# accuracy = 4/6; macro F1 = (0.8 + 0.5 + 2/3) / 3; weighted F1 = (0.8*3 + 0.5*2 + 2/3*1) / 6

def test_hand_computed_headline_metrics():
    metric = compute_metrics( Y_TRUE, Y_PRED, labels = ABC )
    assert metric[ "accuracy" ] == pytest.approx( 4 / 6 )
    assert metric[ "macro_f1" ] == pytest.approx( ( 0.8 + 0.5 + 2 / 3 ) / 3 )
    assert metric[ "weighted_f1" ] == pytest.approx( ( 0.8 * 3 + 0.5 * 2 + 2 / 3 * 1 ) / 6 )

def test_hand_computed_per_class():
    per_class = compute_metrics( Y_TRUE, Y_PRED, labels = ABC )[ "per_class" ]
    assert per_class[ "a" ] == pytest.approx( { "precision": 1.0, "recall": 2 / 3, "f1": 0.8, "support": 3 } )
    assert per_class[ "b" ] == pytest.approx( { "precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2 } )
    assert per_class[ "c" ] == pytest.approx( { "precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1 } )

def test_hand_computed_confusion_rows_are_true_cols_are_predicted():
    confusion = compute_metrics( Y_TRUE, Y_PRED, labels = ABC )[ "confusion" ]
    assert confusion == [
        [ 2, 1, 0 ],
        [ 0, 1, 1 ],
        [ 0, 0, 1 ]
    ]

def test_class_never_predicted_still_reported_with_zero_scores():
    metric = compute_metrics( [ 0, 1, 2 ], [ 0, 1, 1 ], labels = ABC )
    assert metric[ "per_class" ][ "c" ] == { "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1 }

def test_defaullts_to_six_emotions_and_six_by_six_confusion():
    metric = compute_metrics( [ 0, 1 ], [ 0, 1 ] )
    assert list( metric[ "per_class" ].keys() ) == LABELS
    assert len( metric[ "confusion" ] ) == 6
    assert all( len( row ) == 6 for row in metric[ "confusion" ] )

def test_result_is_json_serialisable():
    metric = compute_metrics( Y_TRUE, Y_PRED, labels = ABC )
    json.dumps( metric )

def test_row_normalise_rows_sum_to_one_and_empty_rows_stay_zero():
    normalise = row_normalise( [ [ 2, 2 ], [ 0, 0 ] ] )
    assert normalise == [ [ 0.5, 0.5 ], [ 0.0, 0.0 ] ]