from typer.testing import CliRunner

from emotion_classifier.cli import app

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

def test_unimplemented_command_fails_loudly( tmp_path ):
    config = tmp_path / "x.yaml"
    config.write_text( "experiment: E0\n" )
    result = runner.invoke( app, [ "run", str( config ) ] )
    assert result.exit_code != 0