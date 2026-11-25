import pytest

from emotion_classifier.config import ZeroShotConfig
from emotion_classifier.data import LABELS
from emotion_classifier.results import load_results
from emotion_classifier.zeroshot import labels_to_ids, predict_zeroshot, run_zeroshot_experiment, top_label_ids

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

def run_fake( tiny_splits, tmp_path, tops = None, **kwargs ):
    tops = tops if tops is not None else [ "sadness" ] * len( tiny_splits.test_labels )
    return run_zeroshot_experiment(
        make_cfg(), tiny_splits, "bart_mnli", tmp_path / "results", "20261125T130000",
        git_sha = "abc123", classifier = FakePipeline( tops ), device = "cpu", **kwargs,
    )

def test_experiment_evaluates_on_test_only( tiny_splits, tmp_path ):
    results = run_fake( tiny_splits, tmp_path )
    assert [ r.split for r in results ] == [ "test" ]
    assert [ r.split for r in load_results( tmp_path / "results" ) ] == [ "test" ]
    assert len( list( ( tmp_path / "results" / "preds" ).glob( "*.csv" ) ) ) == 1

def test_experiment_records_metrics_timing_and_provenance( tiny_splits, tmp_path ):
    tops = [ LABELS[ label ] for label in tiny_splits.test_labels ]  # a perfect zero-shot model
    ( result, ) = run_fake( tiny_splits, tmp_path, tops = tops )

    assert result.metrics[ "macro_f1" ] == 1.0
    assert result.experiment == "E4"
    assert result.model == "bart_mnli"
    assert result.run_id.startswith( "E4-bart_mnli-test-42-" )
    assert result.train_seconds is None  # nothing is trained
    assert result.infer_seconds_per_1k >= 0.0
    assert result.n_params is None  # the fake pipeline has no model to count
    assert result.device == "cpu"
    assert result.git_sha == "abc123"
    assert result.config[ "hypothesis_template" ] == "This text expresses {}."

def test_experiment_counts_params_when_the_pipeline_has_a_model( tiny_splits, tmp_path ):
    import torch

    classifier = FakePipeline( [ "joy" ] * len( tiny_splits.test_labels ) )
    classifier.model = torch.nn.Linear( 3, 2 )  # 8 parameters
    ( result, ) = run_zeroshot_experiment(
        make_cfg(), tiny_splits, "bart_mnli", tmp_path / "results", "20261125T130000", classifier = classifier, device = "cpu",
    )
    assert result.n_params == 8

def test_experiment_never_overwrites( tiny_splits, tmp_path ):
    run_fake( tiny_splits, tmp_path )
    with pytest.raises( FileExistsError ):
        run_fake( tiny_splits, tmp_path )
