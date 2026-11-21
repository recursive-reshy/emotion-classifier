import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from emotion_classifier.classical import build_model, fit_predict, run_experiment, tune_on_validation
from emotion_classifier.config import (
    BaselineConfig,
    ClassicalConfig,
    ClassifierConfig,
    TuningConfig,
    VectorizerConfig,
)
from emotion_classifier.results import load_results

def _cfg(
    clf: str = "logreg",
    values: list[ float ] | None = None,
    class_weight: str | None = None,
    param: str | None = None,
    stop_words: str | None = None,
) -> ClassicalConfig:
    return ClassicalConfig(
        experiment = "E1",
        family = "classical",
        vectorizer = VectorizerConfig( kind = "tfidf", ngram_range = ( 1, 2 ), min_df = 1, stop_words = stop_words ),
        classifier = ClassifierConfig( kind = clf, class_weight = class_weight ),
        tuning = TuningConfig( param = param or ( "alpha" if clf == "nb" else "C" ), values = values or [ 0.1, 1, 10 ] ),
        seed = 42,
    )

@pytest.mark.parametrize( "kind, cls, attr", [
    ( "logreg", LogisticRegression, "C" ),
    ( "svm", LinearSVC, "C" ),
    ( "nb", MultinomialNB, "alpha" ),
] )
def test_build_model_picks_classifier_and_applies_tuned_value( kind, cls, attr ):
    model = build_model( _cfg( kind ), 3.0 )
    assert isinstance( model, Pipeline )
    classifier = model.named_steps[ "classifier" ]
    assert isinstance( classifier, cls )
    assert getattr( classifier, attr ) == 3.0

def test_build_model_logreg_uses_max_iter_2000():
    assert build_model( _cfg( "logreg" ), 1.0 ).named_steps[ "classifier" ].max_iter == 2000

def test_build_model_passes_vectorizer_and_class_weight():
    model = build_model( _cfg( "svm", class_weight = "balanced", stop_words = "english" ), 1.0 )
    assert model.named_steps[ "classifier" ].class_weight == "balanced"
    assert model.named_steps[ "vectorizer" ].ngram_range == ( 1, 2 )
    assert "the" in model.named_steps[ "vectorizer" ].stop_words

def test_build_model_rejects_class_weight_for_nb():
    with pytest.raises( ValueError, match = "class_weight" ):
        build_model( _cfg( "nb", class_weight = "balanced" ), 1.0 )

def test_build_model_rejects_wrong_tuning_param():
    with pytest.raises( ValueError, match = "param" ):
        build_model( _cfg( "logreg", param = "alpha" ), 1.0 )

def test_tuning_returns_value_with_best_val_macro_f1( tiny_splits ):
    scores = { 0.1: 0.30, 1: 0.80, 10: 0.55 }
    best, val_scores = tune_on_validation( tiny_splits, _cfg(), scorer = lambda cfg, value, splits: scores[ value ] )
    assert best == 1
    assert val_scores == scores

def test_tuning_ties_pick_the_first_value( tiny_splits ):
    best, _ = tune_on_validation( tiny_splits, _cfg(), scorer = lambda cfg, value, splits: 0.5 )
    assert best == 0.1

def test_default_scorer_runs_on_fixture_and_stays_off_test( tiny_splits ):
    best, val_scores = tune_on_validation( tiny_splits, _cfg() )
    assert best in [ 0.1, 1, 10 ]
    assert set( val_scores ) == { 0.1, 1, 10 }
    assert all( 0.0 <= score <= 1.0 for score in val_scores.values() )

def test_fit_predict_gives_six_way_predictions( tiny_splits ):
    output = fit_predict( tiny_splits, build_model( _cfg(), 10.0 ) )
    assert len( output.val_pred ) == len( tiny_splits.val_texts )
    assert len( output.test_pred ) == len( tiny_splits.test_texts )
    assert set( output.test_pred ) <= set( range( 6 ) )
    assert output.train_seconds > 0
    assert output.val_infer_seconds_per_1k > 0
    assert output.test_infer_seconds_per_1k > 0

def test_run_experiment_saves_val_and_test_results( tiny_splits, tmp_path ):
    results = run_experiment( _cfg(), tiny_splits, "tfidf_logreg_12", tmp_path, "20261121T101500", git_sha = "abc123" )

    assert [ r.split for r in results ] == [ "val", "test" ]
    assert load_results( tmp_path ) == sorted( results, key = lambda r: r.run_id )
    for result, texts, labels in zip(
        results,
        [ tiny_splits.val_texts, tiny_splits.test_texts ],
        [ tiny_splits.val_labels, tiny_splits.test_labels ],
    ):
        assert result.run_id == f"E1-tfidf_logreg_12-{ result.split }-42-20261121T101500"
        assert result.model == "tfidf_logreg_12"
        assert result.experiment == "E1"
        assert result.device == "cpu"
        assert result.seed == 42
        assert result.git_sha == "abc123"
        assert 0.0 <= result.metrics[ "macro_f1" ] <= 1.0
        assert len( result.metrics[ "confusion" ] ) == 6
        assert result.config[ "tuned" ][ "param" ] == "C"
        assert result.config[ "tuned" ][ "value" ] in [ 0.1, 1, 10 ]
        assert set( result.config[ "tuned" ][ "val_scores" ] ) == { "0.1", "1", "10" }
        assert result.config[ "vectorizer" ][ "ngram_range" ] == [ 1, 2 ]
        assert result.train_seconds > 0
        assert result.infer_seconds_per_1k > 0

        lines = ( tmp_path / result.predictions_path ).read_text().splitlines()
        assert lines[ 0 ] == "text,true,pred"
        assert len( lines ) == len( texts ) + 1

def test_run_experiment_single_split_writes_only_that_split( tiny_splits, tmp_path ):
    results = run_experiment( _cfg(), tiny_splits, "x", tmp_path, "t", splits_to_save = "test" )
    assert [ r.split for r in results ] == [ "test" ]
    assert len( list( tmp_path.glob( "*.json" ) ) ) == 1

def test_run_experiment_refuses_to_overwrite( tiny_splits, tmp_path ):
    run_experiment( _cfg(), tiny_splits, "x", tmp_path, "t" )
    with pytest.raises( FileExistsError ):
        run_experiment( _cfg(), tiny_splits, "x", tmp_path, "t" )

def test_run_experiment_handles_baselines_without_tuning( tiny_splits, tmp_path ):
    cfg = BaselineConfig( experiment = "E0", family = "baseline", strategy = "most_frequent", seed = 42 )
    results = run_experiment( cfg, tiny_splits, "majority", tmp_path, "t" )
    assert [ r.split for r in results ] == [ "val", "test" ]
    assert "tuned" not in results[ 0 ].config
    assert results[ 0 ].config[ "strategy" ] == "most_frequent"
    assert results[ 1 ].metrics[ "accuracy" ] == pytest.approx( 1 / 6 )
