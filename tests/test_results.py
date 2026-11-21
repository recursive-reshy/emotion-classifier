import re

import pytest

from emotion_classifier.results import RunResult, load_results, make_run_id, save_predictions, save_result

def _result( run_id: str = "E1-tfidf_logreg_12-test-42-20261121T101500", **overrides ) -> RunResult:
    fields = dict(
        run_id = run_id,
        experiment = "E1",
        model = "tfidf_logreg_12",
        config = { "seed": 42, "classifier": { "kind": "logreg" } },
        split = "test",
        metrics = { "macro_f1": 0.5, "confusion": [ [ 1, 0 ], [ 0, 1 ] ] },
        train_seconds = 1.5,
        infer_seconds_per_1k = None,
        n_params = None,
        device = "cpu",
        seed = 42,
        git_sha = None,
        predictions_path = None,
    )
    fields.update( overrides )
    return RunResult( **fields )

def test_run_id_format():
    run_id = make_run_id( "E1", "tfidf_logreg_12", "test", 42, "20261121T101500" )
    assert run_id == "E1-tfidf_logreg_12-test-42-20261121T101500"

def test_run_id_slugs_unsafe_model_names():
    run_id = make_run_id( "E4", "facebook/bart-large-mnli", "test", 42, "20261121T101500" )
    assert run_id == "E4-facebook-bart-large-mnli-test-42-20261121T101500"
    assert "/" not in run_id
    assert re.fullmatch( r"[A-Za-z0-9._-]+", run_id )

def test_run_id_slug_is_lowercased():
    assert make_run_id( "E3", "DistilBERT Base", "val", 7, "t" ) == "E3-distilbert-base-val-7-t"

def test_run_id_differs_by_split_and_model():
    ids = {
        make_run_id( "E1", "a", "val", 42, "t" ),
        make_run_id( "E1", "a", "test", 42, "t" ),
        make_run_id( "E1", "b", "val", 42, "t" ),
    }
    assert len( ids ) == 3

def test_save_load_round_trip( tmp_path ):
    original = _result()
    path = save_result( original, tmp_path )
    assert path == tmp_path / f"{ original.run_id }.json"
    assert load_results( tmp_path ) == [ original ]

def test_save_result_never_overwrites( tmp_path ):
    save_result( _result(), tmp_path )
    with pytest.raises( FileExistsError ):
        save_result( _result( metrics = { "macro_f1": 0.9 } ), tmp_path )
    assert load_results( tmp_path )[ 0 ].metrics[ "macro_f1" ] == 0.5

def test_save_result_creates_directory( tmp_path ):
    save_result( _result(), tmp_path / "nested" / "results" )
    assert len( load_results( tmp_path / "nested" / "results" ) ) == 1

def test_load_ignores_non_json_files_and_subdirs( tmp_path ):
    save_result( _result(), tmp_path )
    ( tmp_path / ".gitkeep" ).write_text( "" )
    ( tmp_path / "summary.md" ).write_text( "# summary" )
    ( tmp_path / "preds" ).mkdir()
    ( tmp_path / "preds" / "x.csv" ).write_text( "a,b\n" )
    assert [ r.run_id for r in load_results( tmp_path ) ] == [ _result().run_id ]

def test_load_results_sorted_by_run_id( tmp_path ):
    save_result( _result( run_id = "E1-b" ), tmp_path )
    save_result( _result( run_id = "E1-a" ), tmp_path )
    assert [ r.run_id for r in load_results( tmp_path ) ] == [ "E1-a", "E1-b" ]

def test_load_missing_directory_returns_empty( tmp_path ):
    assert load_results( tmp_path / "nope" ) == []

def test_save_predictions_writes_text_true_pred( tmp_path ):
    path = save_predictions( "run1", [ "hello, world", "bye" ], [ 0, 1 ], [ 0, 0 ], tmp_path )
    assert path == tmp_path / "preds" / "run1.csv"
    lines = path.read_text().splitlines()
    assert lines[ 0 ] == "text,true,pred"
    assert lines[ 1 ] == '"hello, world",0,0'
    assert lines[ 2 ] == "bye,1,0"

def test_save_predictions_never_overwrites( tmp_path ):
    save_predictions( "run1", [ "a" ], [ 0 ], [ 0 ], tmp_path )
    with pytest.raises( FileExistsError ):
        save_predictions( "run1", [ "a" ], [ 0 ], [ 1 ], tmp_path )

def test_save_predictions_rejects_length_mismatch( tmp_path ):
    with pytest.raises( ValueError ):
        save_predictions( "run1", [ "a", "b" ], [ 0 ], [ 0 ], tmp_path )
