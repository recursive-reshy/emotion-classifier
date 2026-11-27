from dataclasses import replace

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from emotion_classifier.analysis import best_linear_run, confusion_pairs, refit_linear, sample_errors, top_features
from emotion_classifier.data import LABELS
from emotion_classifier.features import build_vectorizer
from emotion_classifier.results import RunResult, save_predictions

def fitted( classifier, tiny_splits, ngram_range = ( 1, 1 ) ) -> Pipeline:
    pipeline = Pipeline( [
        ( "vectorizer", build_vectorizer( "tfidf", ngram_range, 1, None, None ) ),
        ( "classifier", classifier ),
    ] )
    return pipeline.fit( tiny_splits.train_texts, tiny_splits.train_labels )

@pytest.mark.parametrize( "classifier", [ LogisticRegression( C = 10, max_iter = 2000 ), LinearSVC( C = 1 ) ] )
def test_top_features_returns_k_per_class_best_first( classifier, tiny_splits ):
    features = top_features( fitted( classifier, tiny_splits ), k = 5 )

    assert list( features ) == LABELS
    for items in features.values():
        assert len( items ) == 5
        weights = [ weight for _, weight in items ]
        assert weights == sorted( weights, reverse = True )

def test_top_features_picks_class_specific_words( tiny_splits ):
    features = top_features( fitted( LogisticRegression( C = 10, max_iter = 2000 ), tiny_splits ), k = 10 )
    names = { label: [ name for name, _ in items ] for label, items in features.items() }

    assert "happy" in names[ "joy" ]
    assert "furious" in names[ "anger" ] or "angry" in names[ "anger" ]
    assert "scared" in names[ "fear" ]
    assert "i" not in names[ "joy" ] and "feel" not in names[ "joy" ]  # shared by every class

def test_top_features_includes_bigrams_when_the_vectorizer_has_them( tiny_splits ):
    features = top_features( fitted( LogisticRegression( C = 10, max_iter = 2000 ), tiny_splits, ngram_range = ( 1, 2 ) ), k = 15 )
    assert any( " " in name for items in features.values() for name, _ in items )

def test_top_features_caps_k_at_the_vocabulary_size( tiny_splits ):
    pipeline = fitted( LogisticRegression( max_iter = 2000 ), tiny_splits )
    vocabulary = len( pipeline.named_steps[ "vectorizer" ].get_feature_names_out() )
    assert len( top_features( pipeline, k = vocabulary + 100 )[ "joy" ] ) == vocabulary

def test_top_features_rejects_models_without_coefficients( tiny_splits ):
    with pytest.raises( ValueError, match = "coef_" ):
        top_features( fitted( MultinomialNB(), tiny_splits ) )

@pytest.fixture
def preds_csv( tmp_path ):
    # ids: 0 sadness, 1 joy, 2 love
    texts = [ f"tweet {i}" for i in range( 12 ) ]
    y_true = [ 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0 ]
    y_pred = [ 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 0, 1 ]  # love->joy x7 (tweets 0-6), one love correct, joy->love x1, sadness->joy x1
    return save_predictions( "run", texts, y_true, y_pred, tmp_path )

def test_sample_errors_returns_only_the_requested_confusion_pair( preds_csv ):
    errors = sample_errors( preds_csv, true = 2, pred = 1, n = 100 )
    assert errors == [ f"tweet {i}" for i in range( 7 ) ]

def test_sample_errors_is_deterministic_for_a_seed( preds_csv ):
    first = sample_errors( preds_csv, true = 2, pred = 1, n = 3, seed = 7 )
    assert first == sample_errors( preds_csv, true = 2, pred = 1, n = 3, seed = 7 )
    assert len( first ) == 3 and len( set( first ) ) == 3

def test_sample_errors_seed_changes_the_sample( preds_csv ):
    samples = { tuple( sample_errors( preds_csv, true = 2, pred = 1, n = 3, seed = seed ) ) for seed in range( 20 ) }
    assert len( samples ) > 1

def test_sample_errors_keeps_file_order_and_texts_are_from_the_pair( preds_csv ):
    errors = sample_errors( preds_csv, true = 2, pred = 1, n = 4, seed = 1 )
    numbers = [ int( text.split()[ 1 ] ) for text in errors ]
    assert numbers == sorted( numbers )
    assert all( number < 7 for number in numbers )

def test_sample_errors_for_an_empty_pair_is_empty( preds_csv ):
    assert sample_errors( preds_csv, true = 5, pred = 4 ) == []

def test_sample_errors_rejects_equal_true_and_pred( preds_csv ):
    with pytest.raises( ValueError ):
        sample_errors( preds_csv, true = 1, pred = 1 )

def test_confusion_pairs_returns_top_off_diagonal_cells_by_count():
    confusion = [
        [ 50, 3, 0, 1, 0, 0 ],
        [ 2, 90, 9, 0, 0, 0 ],
        [ 0, 12, 20, 0, 0, 0 ],
        [ 0, 0, 0, 40, 4, 0 ],
        [ 0, 0, 0, 0, 30, 6 ],
        [ 0, 0, 0, 0, 6, 10 ],
    ]
    assert confusion_pairs( confusion, k = 3 ) == [ ( 2, 1, 12 ), ( 1, 2, 9 ), ( 4, 5, 6 ) ]

def test_confusion_pairs_ignores_the_diagonal_and_zero_cells():
    confusion = [ [ 99, 0 ], [ 1, 99 ] ]
    assert confusion_pairs( confusion, k = 5 ) == [ ( 1, 0, 1 ) ]

def test_confusion_pairs_break_ties_deterministically():
    confusion = [ [ 0, 2, 2 ], [ 2, 0, 0 ], [ 0, 0, 0 ] ]
    assert confusion_pairs( confusion, k = 3 ) == [ ( 0, 1, 2 ), ( 0, 2, 2 ), ( 1, 0, 2 ) ]

# ---- best linear model, refit for its top n-grams ---------------------------------------------------

def classical_run( model, kind, split = "val", macro_f1 = 0.8, experiment = "E1", tuned = 1.0 ) -> RunResult:
    config = {
        "experiment": experiment, "family": "classical",
        "vectorizer": { "kind": "tfidf", "ngram_range": [ 1, 1 ], "min_df": 1, "max_features": None, "stop_words": None },
        "classifier": { "kind": kind, "class_weight": None },
        "tuning": { "param": "alpha" if kind == "nb" else "C", "values": [ 0.1, 1 ] },
        "seed": 42,
        "tuned": { "param": "alpha" if kind == "nb" else "C", "value": tuned, "val_scores": {} },
    }
    return RunResult(
        run_id = f"{ experiment }-{ model }-{ split }-42-20261124T090000", experiment = experiment, model = model,
        config = config, split = split, metrics = { "macro_f1": macro_f1 },
        train_seconds = 1.0, infer_seconds_per_1k = 1.0, n_params = None,
        device = "cpu", seed = 42, git_sha = None, predictions_path = None,
    )

def transformer_run( macro_f1 ) -> RunResult:
    run = classical_run( "distilbert", "svm", macro_f1 = macro_f1 )
    return replace( run, experiment = "E3", config = { "experiment": "E3", "family": "transformer" } )

def test_best_linear_run_is_the_best_on_validation_among_linear_classical_models():
    runs = [
        classical_run( "tfidf_nb_11", "nb", macro_f1 = 0.95 ),                       # best overall, but not linear
        classical_run( "bow_svm_11", "svm", macro_f1 = 0.85 ),
        classical_run( "tfidf_logreg_11", "logreg", macro_f1 = 0.80 ),
        classical_run( "tfidf_svm_11", "svm", split = "test", macro_f1 = 0.99 ),     # test never selects
        transformer_run( macro_f1 = 0.99 ),
    ]
    assert best_linear_run( runs ).model == "bow_svm_11"

def test_best_linear_run_needs_a_linear_validation_run():
    with pytest.raises( ValueError, match = "linear" ):
        best_linear_run( [ classical_run( "tfidf_nb_11", "nb" ), classical_run( "tfidf_svm_11", "svm", split = "test" ) ] )

@pytest.mark.parametrize( "kind", [ "logreg", "svm" ] )
def test_refit_linear_rebuilds_the_tuned_pipeline_and_fits_on_train( kind, tiny_splits ):
    pipeline = refit_linear( classical_run( "m", kind, tuned = 3.0 ), tiny_splits )

    assert pipeline.named_steps[ "classifier" ].C == 3.0
    assert hasattr( pipeline.named_steps[ "classifier" ], "coef_" )

def test_refit_linear_rejects_a_model_without_coefficients( tiny_splits ):
    with pytest.raises( ValueError, match = "linear" ):
        refit_linear( classical_run( "tfidf_nb_11", "nb" ), tiny_splits )
