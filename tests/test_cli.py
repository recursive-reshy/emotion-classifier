import pytest
from typer.testing import CliRunner

from emotion_classifier.cli import app
from emotion_classifier.metrics import compute_metrics
from emotion_classifier.results import RunResult, load_results, save_result
from emotion_classifier.transformer import FineTuneOutput

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

ZEROSHOT_YAML = """
experiment: E4
family: zeroshot
model_name: facebook/bart-large-mnli
hypothesis_template: "This text expresses {}."
batch_size: 16
seed: 42
"""

def test_run_zeroshot_writes_a_test_result_only( fake_data, tmp_path, monkeypatch ):
    monkeypatch.setattr(
        "emotion_classifier.zeroshot.predict_zeroshot",
        lambda texts, cfg, classifier = None, device = "cpu": ( [ 1 ] * len( texts ), 1.0 ),
    )
    config = tmp_path / "bart_mnli.yaml"
    config.write_text( ZEROSHOT_YAML )

    result = _invoke( "run", config, "--results-dir", tmp_path / "results" )

    assert result.exit_code == 0, result.output
    results = load_results( tmp_path / "results" )
    assert [ ( r.experiment, r.model, r.split ) for r in results ] == [ ( "E4", "bart_mnli", "test" ) ]

def test_run_transformer_loops_over_seeds( fake_data, tmp_path, monkeypatch ):
    def fake_fine_tune( splits, cfg, seed, output_dir ):
        return FineTuneOutput(
            model = None, tokenizer = None, val_pred = list( splits.val_labels ), test_pred = list( splits.test_labels ),
            train_seconds = 1.0, val_infer_seconds_per_1k = 1.0, test_infer_seconds_per_1k = 1.0,
            n_params = 10, device = "cpu",
        )
    monkeypatch.setattr( "emotion_classifier.transformer.fine_tune", fake_fine_tune )
    config = tmp_path / "distilbert.yaml"
    config.write_text( TRANSFORMER_YAML )

    result = _invoke( "run", config, "--results-dir", tmp_path / "results", "--output-dir", tmp_path / "outputs" )

    assert result.exit_code == 0, result.output
    results = load_results( tmp_path / "results" )
    assert len( results ) == 6
    assert sorted( { r.seed for r in results } ) == [ 7, 13, 42 ]
    assert all( r.model == "distilbert" for r in results )

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

def _saved_results( results_dir ):
    """Val + test results for every family (3 seeds of E3), written like a real run would."""
    y_true = list( range( 6 ) ) * 5

    def save( experiment, model, split, seed, errors ):
        y_pred = [ ( label + 1 ) % 6 if index < errors else label for index, label in enumerate( y_true ) ]
        save_result( RunResult(
            run_id = f"{ experiment }-{ model }-{ split }-{ seed }-20261125T100000", experiment = experiment, model = model,
            config = {}, split = split, metrics = compute_metrics( y_true, y_pred ),
            train_seconds = None if experiment == "E4" else 5.0, infer_seconds_per_1k = 0.5, n_params = None,
            device = "cpu", seed = seed, git_sha = None, predictions_path = None,
        ), results_dir )

    for split in ( "val", "test" ):
        save( "E1", "tfidf_svm_12", split, 42, 4 )
        save( "E1", "bow_nb_11", split, 42, 9 )
        for seed in ( 1, 2, 3 ):
            save( "E3", "distilbert", split, seed, 2 if split == "val" else 4 )
            save( "E3", "roberta_base", split, seed, 3 if split == "val" else 2 )
    save( "E4", "bart_mnli", "test", 42, 15 )

@pytest.fixture
def saved_results( tmp_path ):
    _saved_results( tmp_path / "results" )
    return tmp_path / "results"

def _report( tmp_path, saved_results, *extra ):
    return _invoke( "report", "--results-dir", saved_results, "--figures-dir", tmp_path / "figures", *extra )

def test_report_writes_summary_and_figures( saved_results, tmp_path ):
    result = _report( tmp_path, saved_results )

    assert result.exit_code == 0, result.output
    assert "distilbert" in ( saved_results / "summary.md" ).read_text()
    assert ( saved_results / "summary.csv" ).read_text().startswith( "experiment,model,n_seeds" )
    assert sorted( path.name for path in ( tmp_path / "figures" ).glob( "*.png" ) ) == [
        "confusion_classical.png", "confusion_transformer.png", "confusion_zero-shot.png", "f1_vs_cost.png", "per_class_f1.png",
    ]

def test_report_never_loads_the_dataset_or_trains( saved_results, tmp_path, monkeypatch ):
    def forbidden( *args, **kwargs ):
        raise AssertionError( "report must only read results/" )
    monkeypatch.setattr( "emotion_classifier.cli.load_emotion", forbidden )
    assert _report( tmp_path, saved_results ).exit_code == 0

def test_report_says_which_models_the_figures_show( saved_results, tmp_path ):
    output = _report( tmp_path, saved_results ).output
    assert "Transformer: distilbert" in output  # best on validation
    assert "Classical: tfidf_svm_12" in output

def test_report_select_overrides_the_model_shown( saved_results, tmp_path ):
    output = _report( tmp_path, saved_results, "--select", "Transformer=roberta_base" ).output
    assert "Transformer: roberta_base" in output

def test_report_rejects_a_bad_selection( saved_results, tmp_path ):
    for bad in ( "Transformer=nope", "Linear=tfidf_svm_12", "no-equals-sign" ):
        result = _report( tmp_path, saved_results, "--select", bad )
        assert result.exit_code != 0
        assert not ( tmp_path / "figures" ).exists()

def test_report_can_be_rerun_and_overwrites_its_outputs( saved_results, tmp_path ):
    assert _report( tmp_path, saved_results ).exit_code == 0
    assert _report( tmp_path, saved_results ).exit_code == 0

def test_report_without_results_fails_with_a_clear_message( tmp_path ):
    result = _invoke( "report", "--results-dir", tmp_path / "empty", "--figures-dir", tmp_path / "figures" )
    assert result.exit_code != 0
    assert "No test results" in result.output

def test_run_passes_train_fraction_to_the_loader_for_classical( tmp_path, monkeypatch, tiny_splits ):
    calls = []
    monkeypatch.setattr( "emotion_classifier.cli.load_emotion", lambda *args, **kwargs: calls.append( kwargs ) or tiny_splits )
    config = tmp_path / "tfidf_logreg_12_f25.yaml"
    config.write_text( CLASSICAL_YAML + "train_fraction: 0.25\n" )

    assert _invoke( "run", config, "--results-dir", tmp_path / "results" ).exit_code == 0
    assert calls[ 0 ][ "subsample" ] == 0.25
    assert load_results( tmp_path / "results" )[ 0 ].config[ "train_fraction" ] == 0.25

def test_run_passes_train_fraction_to_the_loader_for_transformer( tmp_path, monkeypatch, tiny_splits ):
    calls = []
    monkeypatch.setattr( "emotion_classifier.cli.load_emotion", lambda *args, **kwargs: calls.append( kwargs ) or tiny_splits )
    monkeypatch.setattr(
        "emotion_classifier.transformer.fine_tune",
        lambda splits, cfg, seed, output_dir: FineTuneOutput(
            model = None, tokenizer = None, val_pred = list( splits.val_labels ), test_pred = list( splits.test_labels ),
            train_seconds = 1.0, val_infer_seconds_per_1k = 1.0, test_infer_seconds_per_1k = 1.0, n_params = 10, device = "cpu",
        ),
    )
    config = tmp_path / "distilbert_f10.yaml"
    config.write_text( TRANSFORMER_YAML.replace( "[ 13, 42, 7 ]", "[ 42 ]" ) + "train_fraction: 0.1\n" )

    assert _invoke( "run", config, "--results-dir", tmp_path / "results", "--output-dir", tmp_path / "out" ).exit_code == 0
    assert calls[ 0 ][ "subsample" ] == 0.1
    assert calls[ 0 ][ "seed" ] == 42  # the subset is drawn with the first seed, matching the classical run

def test_run_without_train_fraction_uses_the_full_train_split( tmp_path, monkeypatch, tiny_splits ):
    calls = []
    monkeypatch.setattr( "emotion_classifier.cli.load_emotion", lambda *args, **kwargs: calls.append( kwargs ) or tiny_splits )
    config = tmp_path / "majority.yaml"
    config.write_text( BASELINE_YAML )
    assert _invoke( "run", config, "--results-dir", tmp_path / "results" ).exit_code == 0
    assert calls[ 0 ].get( "subsample" ) is None

LLM_YAML = """
experiment: E5
family: llm
model_name: Qwen/Qwen3-4B-Instruct-2507
batch_size: 16
max_new_tokens: 8
seed: 42
"""

def test_run_llm_config_writes_a_test_result_with_invalid_rate( fake_data, tmp_path, monkeypatch, tiny_splits ):
    from emotion_classifier.data import LABELS

    outputs = iter( [ LABELS[ label ] for label in tiny_splits.test_labels ] )
    monkeypatch.setattr( "emotion_classifier.zeroshot.build_generator", lambda cfg, device: lambda prompts: [ next( outputs ) for _ in prompts ] )
    config = tmp_path / "qwen3_4b_instruct.yaml"
    config.write_text( LLM_YAML )

    result = _invoke( "run", config, "--results-dir", tmp_path / "results" )

    assert result.exit_code == 0, result.output
    ( saved, ) = load_results( tmp_path / "results" )
    assert ( saved.experiment, saved.model, saved.split ) == ( "E5", "qwen3_4b_instruct", "test" )
    assert saved.metrics[ "invalid_rate" ] == 0.0
    assert "invalid" in result.output.lower()
