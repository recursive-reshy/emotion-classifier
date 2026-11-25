import pytest

from emotion_classifier.data import LABELS
from emotion_classifier.metrics import compute_metrics
from emotion_classifier.plots import (
    learning_curve_points, make_all, mean_per_class_f1, plot_confusion_matrices, plot_confusion_matrix, plot_f1_vs_cost,
    plot_learning_curve, plot_per_class_f1, pooled_confusion, resolve_selection, select_models,
)
from emotion_classifier.results import RunResult

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
Y_TRUE = list( range( 6 ) ) * 5  # 30 examples, 5 per class

def predictions( errors: int ) -> list[ int ]:
    """Y_TRUE with the first `errors` predictions shifted to the next class (so more errors = lower F1)."""
    return [ ( label + 1 ) % 6 if index < errors else label for index, label in enumerate( Y_TRUE ) ]

def make_result( experiment, model, split = "test", seed = 1, errors = 0, timestamp = "20261124T090000", infer = 1.0, config = None ) -> RunResult:
    return RunResult(
        run_id = f"{ experiment }-{ model }-{ split }-{ seed }-{ timestamp }",
        experiment = experiment, model = model, config = config or {}, split = split,
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

def test_resolve_selection_without_overrides_is_the_validation_choice( results ):
    assert resolve_selection( results, {} ) == select_models( results )

def test_resolve_selection_override_replaces_only_that_family( results ):
    selected = resolve_selection( results, { "Transformer": "roberta_base" } )
    assert selected[ "Transformer" ] == "roberta_base"
    assert selected[ "Classical" ] == "tfidf_svm_12"

def test_resolve_selection_family_names_are_case_insensitive( results ):
    assert resolve_selection( results, { "transformer": "roberta_base", "ZERO-SHOT": "bart_mnli" } )[ "Transformer" ] == "roberta_base"

def test_resolve_selection_rejects_an_unknown_family( results ):
    with pytest.raises( ValueError, match = "Classical" ):
        resolve_selection( results, { "Linear": "tfidf_svm_12" } )

def test_resolve_selection_rejects_a_model_from_another_family_or_missing( results ):
    with pytest.raises( ValueError, match = "roberta_base" ):
        resolve_selection( results, { "Classical": "roberta_base" } )
    with pytest.raises( ValueError, match = "nope" ):
        resolve_selection( results, { "Transformer": "nope" } )

def test_resolve_selection_lets_an_override_settle_an_ambiguous_family( results ):
    extra = make_result( "E4", "other_template", "test", errors = 5 )
    with pytest.raises( ValueError ):
        resolve_selection( [ *results, extra ], {} )
    assert resolve_selection( [ *results, extra ], { "Zero-shot": "other_template" } )[ "Zero-shot" ] == "other_template"

def curve_results() -> list[ RunResult ]:
    """E6: a classical and a transformer series at four train fractions (errors fall as data grows)."""
    results = []
    for family, name, errors in ( ( "classical", "bow_svm", [ 12, 8, 5, 3 ] ), ( "transformer", "distilbert", [ 8, 4, 3, 2 ] ) ):
        for fraction, percent, error_count in zip( [ 0.1, 0.25, 0.5, 1 ], [ 10, 25, 50, 100 ], errors ):
            results.append( make_result( "E6", f"{ name }_f{ percent }", errors = error_count, seed = 42, config = { "family": family, "train_fraction": fraction } ) )
    return results

def test_learning_curve_points_are_sorted_by_fraction_per_series():
    points = learning_curve_points( curve_results() )

    assert list( points ) == [ "bow_svm", "distilbert" ]
    assert [ fraction for fraction, _ in points[ "bow_svm" ] ] == [ 0.1, 0.25, 0.5, 1 ]
    f1s = [ f1 for _, f1 in points[ "distilbert" ] ]
    assert f1s == sorted( f1s )  # fewer errors with more data
    assert f1s[ 0 ] == make_result( "E6", "x", errors = 8 ).metrics[ "macro_f1" ]

def test_learning_curve_points_use_test_results_only_and_the_latest_run():
    results = curve_results()
    results.append( make_result( "E6", "bow_svm_f10", "val", errors = 30, seed = 42, config = { "family": "classical", "train_fraction": 0.1 } ) )
    rerun = make_result( "E6", "bow_svm_f10", errors = 0, seed = 42, timestamp = "20261125T090000", config = { "family": "classical", "train_fraction": 0.1 } )
    points = dict( learning_curve_points( [ *results, rerun ] )[ "bow_svm" ] )
    assert points[ 0.1 ] == 1.0  # the later test run, not the val run or the earlier test run

def test_learning_curve_treats_a_missing_fraction_as_the_full_train_set():
    result = make_result( "E6", "bow_svm_f100", config = { "family": "classical" } )
    assert learning_curve_points( [ result ] ) == { "bow_svm": [ ( 1.0, result.metrics[ "macro_f1" ] ) ] }

def test_learning_curve_ignores_other_experiments( results ):
    assert learning_curve_points( results ) == {}

def test_plot_learning_curve_writes_a_png( tmp_path ):
    assert_png( plot_learning_curve( curve_results(), tmp_path / "learning_curve.png" ) )

def test_plot_learning_curve_without_e6_results_raises( results, tmp_path ):
    with pytest.raises( ValueError, match = "E6" ):
        plot_learning_curve( results, tmp_path / "learning_curve.png" )

def test_make_all_adds_the_learning_curve_when_e6_results_exist( results, tmp_path ):
    paths = make_all( [ *results, *curve_results() ], tmp_path )
    assert "learning_curve.png" in [ path.name for path in paths ]
    assert len( paths ) == 6

def test_e6_runs_do_not_leak_into_model_selection_or_the_cost_plot( results, tmp_path ):
    assert select_models( [ *results, *curve_results() ] ) == select_models( results )
    assert_png( plot_f1_vs_cost( [ *results, *curve_results() ], tmp_path / "f1_vs_cost.png" ) )

def test_zero_shot_family_covers_nli_and_llm_and_defaults_to_the_nli_model( results ):
    llm = make_result( "E5", "qwen", "test", errors = 10, infer = 9.0 )
    assert select_models( [ *results, llm ] )[ "Zero-shot" ] == "bart_mnli"

def test_zero_shot_llm_can_be_chosen_explicitly_and_gets_its_own_figures( results, tmp_path ):
    llm = make_result( "E5", "qwen", "test", errors = 10, infer = 9.0 )
    selected = resolve_selection( [ *results, llm ], { "Zero-shot": "qwen" } )
    assert selected[ "Zero-shot" ] == "qwen"
    assert_png( plot_per_class_f1( [ *results, llm ], tmp_path / "per_class_f1.png", selected ) )

def test_cost_plot_draws_the_llm_point_next_to_the_nli_point( results, tmp_path ):
    llm = make_result( "E5", "qwen", "test", errors = 10, infer = 9.0 )
    assert_png( plot_f1_vs_cost( [ *results, llm ], tmp_path / "f1_vs_cost.png" ) )
