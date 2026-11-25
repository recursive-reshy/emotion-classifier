import pytest

from emotion_classifier.config import ZeroShotConfig
from emotion_classifier.data import LABELS
from emotion_classifier.zeroshot import labels_to_ids, predict_zeroshot, top_label_ids

def make_cfg( **overrides ) -> ZeroShotConfig:
    values = dict(
        experiment = "E4", family = "zeroshot", model_name = "facebook/bart-large-mnli",
        hypothesis_template = "This text expresses {}.", batch_size = 4, seed = 42,
    )
    values.update( overrides )
    return ZeroShotConfig( **values )

def output_for( top: str, text: str = "x" ) -> dict:
    """Shape of one zero-shot pipeline result: labels sorted by score, best first."""
    rest = [ label for label in LABELS if label != top ]
    return { "sequence": text, "labels": [ top, *rest ], "scores": [ 0.5, 0.2, 0.1, 0.1, 0.05, 0.05 ] }

class FakePipeline:
    """Stands in for pipeline( "zero-shot-classification" ); records how it was called."""

    def __init__( self, tops: list[ str ] ):
        self.tops = tops
        self.calls: list[ tuple ] = []

    def __call__( self, texts, **kwargs ):
        self.calls.append( ( texts, kwargs ) )
        return [ output_for( top, text ) for top, text in zip( self.tops, texts ) ]

def test_labels_to_ids_maps_each_label_name_to_its_index():
    assert labels_to_ids( LABELS ) == [ 0, 1, 2, 3, 4, 5 ]
    assert labels_to_ids( [ "fear", "joy" ] ) == [ 4, 1 ]

def test_labels_to_ids_rejects_unknown_label():
    with pytest.raises( ValueError, match = "boredom" ):
        labels_to_ids( [ "boredom" ] )

def test_top_label_ids_uses_the_highest_scoring_label():
    outputs = [ output_for( "love" ), output_for( "surprise" ), output_for( "sadness" ) ]
    assert top_label_ids( outputs ) == [ 2, 5, 0 ]

def test_top_label_ids_does_not_assume_score_order_of_the_labels_list():
    unsorted = { "sequence": "x", "labels": [ "joy", "fear", "anger" ], "scores": [ 0.1, 0.7, 0.2 ] }
    assert top_label_ids( [ unsorted ] ) == [ 4 ]

def test_top_label_ids_accepts_a_single_result_dict():
    assert top_label_ids( output_for( "anger" ) ) == [ 3 ]

def test_predict_zeroshot_passes_fixed_template_labels_and_batch_size():
    fake = FakePipeline( [ "joy", "fear" ] )
    predictions, _ = predict_zeroshot( [ "a", "b" ], make_cfg(), classifier = fake )

    assert predictions == [ 1, 4 ]
    ( texts, kwargs ), = fake.calls
    assert texts == [ "a", "b" ]
    assert kwargs[ "candidate_labels" ] == LABELS
    assert kwargs[ "hypothesis_template" ] == "This text expresses {}."
    assert kwargs[ "batch_size" ] == 4
    assert kwargs[ "multi_label" ] is False

def test_predict_zeroshot_returns_seconds_per_1k_examples():
    _, per_1k = predict_zeroshot( [ "a", "b" ], make_cfg(), classifier = FakePipeline( [ "joy", "joy" ] ) )
    assert per_1k >= 0.0

def test_predict_zeroshot_handles_empty_input():
    predictions, per_1k = predict_zeroshot( [], make_cfg(), classifier = FakePipeline( [] ) )
    assert predictions == []
    assert per_1k == 0.0
