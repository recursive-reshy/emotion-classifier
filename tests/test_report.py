import csv
import statistics

import pytest

from emotion_classifier.report import SUMMARY_COLUMNS, summarise, to_markdown, write_summary
from emotion_classifier.results import RunResult

def make_result(
    experiment = "E3", model = "distilbert", split = "test", seed = 1, timestamp = "20261124T090000",
    accuracy = 0.9, macro_f1 = 0.8, weighted_f1 = 0.85,
    train_seconds = 100.0, infer = 1.0, n_params = 1000,
) -> RunResult:
    return RunResult(
        run_id = f"{ experiment }-{ model }-{ split }-{ seed }-{ timestamp }",
        experiment = experiment, model = model, config = {}, split = split,
        metrics = { "accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1 },
        train_seconds = train_seconds, infer_seconds_per_1k = infer, n_params = n_params,
        device = "cpu", seed = seed, git_sha = None, predictions_path = None,
    )

def three_seeds( **kwargs ) -> list[ RunResult ]:
    f1s, accs, trains = [ 0.88, 0.87, 0.89 ], [ 0.92, 0.93, 0.94 ], [ 240.0, 260.0, 250.0 ]
    return [
        make_result( seed = seed, macro_f1 = f1, accuracy = acc, train_seconds = train, **kwargs )
        for seed, f1, acc, train in zip( [ 13, 42, 7 ], f1s, accs, trains )
    ]

def test_summarise_gives_mean_and_sample_std_over_seeds():
    ( row, ) = summarise( three_seeds() )

    assert row[ "experiment" ] == "E3" and row[ "model" ] == "distilbert"
    assert row[ "n_seeds" ] == 3
    assert row[ "macro_f1_mean" ] == pytest.approx( 0.88 )
    assert row[ "macro_f1_std" ] == pytest.approx( statistics.stdev( [ 0.88, 0.87, 0.89 ] ) )
    assert row[ "accuracy_mean" ] == pytest.approx( 0.93 )
    assert row[ "train_seconds_mean" ] == pytest.approx( 250.0 )
    assert row[ "n_params" ] == 1000

def test_summarise_uses_test_split_only():
    results = three_seeds() + [ make_result( split = "val", seed = 99, macro_f1 = 0.1 ) ]
    ( row, ) = summarise( results )
    assert row[ "n_seeds" ] == 3
    assert row[ "macro_f1_mean" ] == pytest.approx( 0.88 )

def test_summarise_single_seed_has_no_std():
    ( row, ) = summarise( [ make_result( experiment = "E1", model = "tfidf_svm_12", macro_f1 = 0.7 ) ] )
    assert row[ "n_seeds" ] == 1
    assert row[ "macro_f1_mean" ] == 0.7
    assert row[ "macro_f1_std" ] is None

def test_summarise_keeps_none_for_missing_cost_fields():
    ( row, ) = summarise( [ make_result( experiment = "E4", model = "bart_mnli", train_seconds = None, n_params = None ) ] )
    assert row[ "train_seconds_mean" ] is None
    assert row[ "train_seconds_std" ] is None
    assert row[ "n_params" ] is None
    assert row[ "infer_seconds_per_1k_mean" ] == 1.0

def test_summarise_counts_a_repeated_seed_once_keeping_the_latest_run():
    results = [
        make_result( seed = 1, macro_f1 = 0.50, timestamp = "20261124T090000" ),
        make_result( seed = 1, macro_f1 = 0.60, timestamp = "20261125T090000" ),
        make_result( seed = 2, macro_f1 = 0.70 ),
    ]
    ( row, ) = summarise( results )
    assert row[ "n_seeds" ] == 2
    assert row[ "macro_f1_mean" ] == pytest.approx( 0.65 )

def test_summarise_has_one_row_per_experiment_and_model_sorted_by_experiment_then_f1():
    results = [
        make_result( experiment = "E3", model = "roberta_base", macro_f1 = 0.9 ),
        make_result( experiment = "E1", model = "bow_nb_11", macro_f1 = 0.5 ),
        make_result( experiment = "E1", model = "tfidf_svm_12", macro_f1 = 0.8 ),
        make_result( experiment = "E0", model = "majority", macro_f1 = 0.1 ),
    ]
    rows = summarise( results )
    assert [ ( r[ "experiment" ], r[ "model" ] ) for r in rows ] == [
        ( "E0", "majority" ), ( "E1", "tfidf_svm_12" ), ( "E1", "bow_nb_11" ), ( "E3", "roberta_base" ),
    ]

def test_summarise_of_no_results_is_empty():
    assert summarise( [] ) == []
    assert summarise( [ make_result( split = "val" ) ] ) == []

def test_markdown_shows_mean_plus_minus_std_and_dashes_for_missing():
    rows = summarise( three_seeds() + [ make_result( experiment = "E4", model = "bart_mnli", train_seconds = None, n_params = None ) ] )
    markdown = to_markdown( rows )
    lines = markdown.strip().splitlines()

    assert lines[ 0 ].startswith( "| " ) and "macro-F1" in lines[ 0 ]
    assert set( lines[ 1 ] ) <= set( "|- " )
    e3 = next( line for line in lines if "distilbert" in line )
    assert "0.880 ± 0.010" in e3
    assert "1,000" in e3
    e4 = next( line for line in lines if "bart_mnli" in line )
    assert "—" in e4  # no train time, no params
    assert "±" not in e4  # single seed: no std

def test_write_summary_writes_csv_and_markdown( tmp_path ):
    rows = summarise( three_seeds() )
    csv_path, md_path = tmp_path / "summary.csv", tmp_path / "summary.md"
    write_summary( rows, csv_path, md_path )

    with csv_path.open( newline = "", encoding = "utf-8" ) as handle:
        records = list( csv.DictReader( handle ) )
    assert list( records[ 0 ] ) == SUMMARY_COLUMNS
    assert records[ 0 ][ "model" ] == "distilbert"
    assert float( records[ 0 ][ "macro_f1_mean" ] ) == pytest.approx( 0.88 )
    assert md_path.read_text( encoding = "utf-8" ) == to_markdown( rows )

def test_write_summary_overwrites_because_it_is_regenerated( tmp_path ):
    rows = summarise( three_seeds() )
    for _ in range( 2 ):
        write_summary( rows, tmp_path / "summary.csv", tmp_path / "summary.md" )
    assert ( tmp_path / "summary.csv" ).exists()

def test_markdown_keeps_precision_for_small_times():
    row = summarise( [ make_result( experiment = "E1", model = "tfidf_svm_12", train_seconds = 0.0421, infer = 0.0038 ) ] )
    line = next( line for line in to_markdown( row ).splitlines() if "tfidf_svm_12" in line )
    assert "0.042" in line   # not "0"
    assert "0.004" in line   # not "0.00"

def test_markdown_rounds_large_times_to_whole_seconds():
    line = next( line for line in to_markdown( summarise( three_seeds() ) ).splitlines() if "distilbert" in line )
    assert "250 ± 10" in line

def test_summary_includes_the_invalid_rate_for_llm_runs_only():
    llm = make_result( experiment = "E5", model = "qwen", train_seconds = None )
    llm.metrics[ "invalid_rate" ] = 0.05
    rows = summarise( [ llm, make_result( experiment = "E4", model = "bart_mnli", train_seconds = None ) ] )
    by_model = { row[ "model" ]: row for row in rows }

    assert by_model[ "qwen" ][ "invalid_rate_mean" ] == pytest.approx( 0.05 )
    assert by_model[ "bart_mnli" ][ "invalid_rate_mean" ] is None
    lines = to_markdown( rows ).splitlines()
    assert "Invalid" in lines[ 0 ]
    assert "5.0%" in next( line for line in lines if "qwen" in line )
    assert "5.0%" not in next( line for line in lines if "bart_mnli" in line )
