import pytest
from typer.testing import CliRunner

from emotion_classifier.cli import app
from emotion_classifier.results import load_results

runner = CliRunner()

def test_help_lists_all_three_commands():
    result = runner.invoke( app, [ "--help" ] )
    assert result.exit_code == 0

    for command in [ "run", "grid", "report" ]:
        assert command in result.output

def test_run_help_shows_its_options():
    result = runner.invoke( app, [ "run", "--help" ] )
    assert result.exit_code == 0
    assert "--output-dir" in result.output
    assert "--split" in result.output

def test_unimplemented_command_fails_loudly():
    result = runner.invoke( app, [ "report" ] )
    assert result.exit_code != 0

CLASSICAL_YAML = """
experiment: E1
family: classical
vectorizer: { kind: tfidf, ngram_range: [ 1, 2 ], min_df: 1 }
classifier: { kind: logreg }
tuning: { param: C, values: [ 0.1, 1 ] }
seed: 42
"""

BASELINE_YAML = """
experiment: E0
family: baseline
strategy: most_frequent
seed: 42
"""

TRANSFORMER_YAML = """
experiment: E3
family: transformer
model_name: distilbert-base-uncased
max_length: 88
lr: 5.0e-5
epochs: 4
batch_size: 32
weight_decay: 0.01
warmup_ratio: 0.1
early_stopping_patience: 2
seeds: [ 13, 42, 7 ]
"""

@pytest.fixture
def fake_data( monkeypatch, tiny_splits ):
    monkeypatch.setattr( "emotion_classifier.cli.load_emotion", lambda *args, **kwargs: tiny_splits )

def _invoke( *args ):
    return runner.invoke( app, [ str( arg ) for arg in args ] )

def test_run_classical_writes_val_and_test_results( fake_data, tmp_path ):
    config = tmp_path / "tfidf_logreg_12.yaml"
    config.write_text( CLASSICAL_YAML )
    results_dir = tmp_path / "results"

    result = _invoke( "run", config, "--results-dir", results_dir )

    assert result.exit_code == 0, result.output
    results = load_results( results_dir )
    assert sorted( r.split for r in results ) == [ "test", "val" ]
    assert all( r.model == "tfidf_logreg_12" for r in results )
    assert len( list( ( results_dir / "preds" ).glob( "*.csv" ) ) ) == 2

def test_run_single_split_writes_one_result( fake_data, tmp_path ):
    config = tmp_path / "majority.yaml"
    config.write_text( BASELINE_YAML )
    result = _invoke( "run", config, "--results-dir", tmp_path / "results", "--split", "val" )
    assert result.exit_code == 0, result.output
    assert [ r.split for r in load_results( tmp_path / "results" ) ] == [ "val" ]

def test_grid_runs_every_config_in_the_folder( fake_data, tmp_path ):
    configs = tmp_path / "configs"
    configs.mkdir()
    ( configs / "majority.yaml" ).write_text( BASELINE_YAML )
    ( configs / "tfidf_logreg_12.yaml" ).write_text( CLASSICAL_YAML )
    ( configs / "notes.txt" ).write_text( "ignored" )

    result = _invoke( "grid", configs, "--results-dir", tmp_path / "results" )

    assert result.exit_code == 0, result.output
    assert len( load_results( tmp_path / "results" ) ) == 4

def test_grid_on_folder_without_configs_fails( tmp_path ):
    ( tmp_path / "empty" ).mkdir()
    assert _invoke( "grid", tmp_path / "empty", "--results-dir", tmp_path / "r" ).exit_code != 0

def test_run_transformer_config_is_not_implemented_yet( fake_data, tmp_path ):
    config = tmp_path / "distilbert.yaml"
    config.write_text( TRANSFORMER_YAML )
    result = _invoke( "run", config, "--results-dir", tmp_path / "results" )
    assert result.exit_code != 0
    assert "not implemented" in result.output
    assert load_results( tmp_path / "results" ) == []

def test_run_invalid_config_fails_with_a_clear_message( fake_data, tmp_path ):
    config = tmp_path / "bad.yaml"
    config.write_text( BASELINE_YAML + "learning_rate: 1\n" )
    result = _invoke( "run", config, "--results-dir", tmp_path / "results" )
    assert result.exit_code != 0
    assert "learning_rate" in result.output

def test_run_never_overwrites_existing_results( fake_data, tmp_path, monkeypatch ):
    monkeypatch.setattr( "emotion_classifier.cli._timestamp", lambda: "20261121T101500" )
    config = tmp_path / "majority.yaml"
    config.write_text( BASELINE_YAML )
    args = ( "run", config, "--results-dir", tmp_path / "results" )
    assert _invoke( *args ).exit_code == 0
    second = _invoke( *args )
    assert second.exit_code != 0
    assert len( load_results( tmp_path / "results" ) ) == 2
