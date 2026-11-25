import pytest

from emotion_classifier.data import LABELS
from emotion_classifier.metrics import compute_metrics
from emotion_classifier.plots import (
    make_all, mean_per_class_f1, plot_confusion_matrices, plot_confusion_matrix, plot_f1_vs_cost,
    plot_per_class_f1, pooled_confusion, select_models,
)
from emotion_classifier.results import RunResult

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
Y_TRUE = list( range( 6 ) ) * 5  # 30 examples, 5 per class

def predictions( errors: int ) -> list[ int ]:
    """Y_TRUE with the first `errors` predictions shifted to the next class (so more errors = lower F1)."""
    return [ ( label + 1 ) % 6 if index < errors else label for index, label in enumerate( Y_TRUE ) ]

def make_result( experiment, model, split = "test", seed = 1, errors = 0, timestamp = "20261124T090000", infer = 1.0 ) -> RunResult:
    return RunResult(
        run_id = f"{ experiment }-{ model }-{ split }-{ seed }-{ timestamp }",
        experiment = experiment, model = model, config = {}, split = split,
        metrics = compute_metrics( Y_TRUE, predictions( errors ) ),
        train_seconds = None if experiment == "E4" else 10.0, infer_seconds_per_1k = infer, n_params = None,
        device = "cpu", seed = seed, git_sha = None, predictions_path = None,
    )

@pytest.fixture
def results() -> list[ RunResult ]:
    both = lambda experiment, model, val_errors, test_errors, **kwargs: [
        make_result( experiment, model, "val", errors = val_errors, **kwargs ),
        make_result( experiment, model, "test", errors = test_errors, **kwargs ),
    ]
    return [
        *both( "E0", "majority", 25, 25 ),
        *both( "E1", "tfidf_svm_12", 4, 12, infer = 0.01 ),   # best on val, worse on test
        *both( "E1", "bow_nb_11", 8, 6, infer = 0.01 ),       # worse on val, best on test
        *[ r for seed in ( 1, 2, 3 ) for r in both( "E3", "distilbert", 2, 3, seed = seed, infer = 1.2 ) ],
        *[ r for seed in ( 1, 2, 3 ) for r in both( "E3", "roberta_base", 3, 2, seed = seed, infer = 2.2 ) ],
        make_result( "E4", "bart_mnli", "test", errors = 15, infer = 39.0 ),
    ]

def test_select_models_picks_the_best_on_validation_not_test( results ):
    assert select_models( results )[ "Classical" ] == "tfidf_svm_12"
    assert select_models( results )[ "Transformer" ] == "distilbert"

def test_select_models_takes_the_only_zero_shot_model_which_has_no_val_results( results ):
    assert select_models( results )[ "Zero-shot" ] == "bart_mnli"

def test_select_models_ignores_baselines_and_missing_families( results ):
    selected = select_models( [ r for r in results if r.experiment in ( "E0", "E1" ) ] )
    assert list( selected ) == [ "Classical" ]

def test_select_models_refuses_to_guess_between_several_unvalidated_models( results ):
    extra = make_result( "E4", "other_template", "test", errors = 5 )
    with pytest.raises( ValueError, match = "Zero-shot" ):
        select_models( [ *results, extra ] )

def test_pooled_confusion_sums_the_test_matrices_over_seeds( results ):
    pooled = pooled_confusion( results, "distilbert" )
    single = next( r for r in results if r.model == "distilbert" and r.split == "test" ).metrics[ "confusion" ]
    assert pooled == [ [ 3 * cell for cell in row ] for row in single ]
    assert sum( map( sum, pooled ) ) == 90

def test_pooled_confusion_counts_a_repeated_seed_once( results ):
    rerun = make_result( "E3", "distilbert", "test", seed = 1, errors = 30, timestamp = "20261125T090000" )
    pooled = pooled_confusion( [ *results, rerun ], "distilbert" )
    assert sum( map( sum, pooled ) ) == 90  # still 3 seeds
    assert pooled[ 0 ][ 1 ] > 0 and pooled != pooled_confusion( results, "distilbert" )  # the latest run replaced seed 1

def test_mean_per_class_f1_averages_over_seeds():
    low = make_result( "E3", "m", seed = 1, errors = 0 )
    high = make_result( "E3", "m", seed = 2, errors = 6 )
    expected = ( low.metrics[ "per_class" ][ "joy" ][ "f1" ] + high.metrics[ "per_class" ][ "joy" ][ "f1" ] ) / 2
    assert mean_per_class_f1( [ low, high ], "m" )[ "joy" ] == pytest.approx( expected )
    assert list( mean_per_class_f1( [ low, high ], "m" ) ) == LABELS

def assert_png( path ):
    assert path.exists()
    assert path.read_bytes()[ :8 ] == PNG_SIGNATURE
    assert path.stat().st_size > 1000

def test_plot_confusion_matrix_writes_a_png_and_creates_the_folder( tmp_path ):
    confusion = compute_metrics( Y_TRUE, predictions( 8 ) )[ "confusion" ]
    path = plot_confusion_matrix( confusion, "Test", tmp_path / "figures" / "cm.png" )
    assert_png( path )

def test_plot_confusion_matrix_copes_with_an_empty_row( tmp_path ):
    confusion = [ [ 0 ] * 6 for _ in range( 6 ) ]
    confusion[ 0 ][ 0 ] = 4
    assert_png( plot_confusion_matrix( confusion, "Sparse", tmp_path / "cm.png" ) )

def test_plot_confusion_matrices_writes_one_png_per_selected_family( results, tmp_path ):
    paths = plot_confusion_matrices( results, tmp_path )
    assert sorted( path.name for path in paths ) == [ "confusion_classical.png", "confusion_transformer.png", "confusion_zero-shot.png" ]
    for path in paths:
        assert_png( path )

def test_plot_confusion_matrices_can_use_an_explicit_model_choice( results, tmp_path ):
    paths = plot_confusion_matrices( results, tmp_path, selected = { "Transformer": "roberta_base" } )
    assert [ path.name for path in paths ] == [ "confusion_transformer.png" ]

def test_plot_per_class_f1_writes_a_png( results, tmp_path ):
    assert_png( plot_per_class_f1( results, tmp_path / "per_class_f1.png" ) )

def test_plot_f1_vs_cost_writes_a_png( results, tmp_path ):
    assert_png( plot_f1_vs_cost( results, tmp_path / "f1_vs_cost.png" ) )

def test_plots_can_be_regenerated_over_existing_files( results, tmp_path ):
    for _ in range( 2 ):
        assert_png( plot_per_class_f1( results, tmp_path / "per_class_f1.png" ) )

def test_make_all_writes_every_figure( results, tmp_path ):
    paths = make_all( results, tmp_path / "figures" )
    assert sorted( path.name for path in paths ) == [
        "confusion_classical.png", "confusion_transformer.png", "confusion_zero-shot.png", "f1_vs_cost.png", "per_class_f1.png",
    ]
    for path in paths:
        assert_png( path )

def test_plots_refuse_to_draw_without_test_results( tmp_path ):
    with pytest.raises( ValueError, match = "test" ):
        make_all( [ make_result( "E1", "tfidf_svm_12", "val" ) ], tmp_path )
