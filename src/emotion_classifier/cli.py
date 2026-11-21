import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from emotion_classifier.classical import run_experiment
from emotion_classifier.config import BaselineConfig, ClassicalConfig, ConfigError, load_config
from emotion_classifier.data import load_emotion

app = typer.Typer(
    help = "Emotion classification experiments: classical vs transformer vs zero-shot.",
    no_args_is_help = True
)

class Split( str, Enum ):
    val = "val"
    test = "test"
    both = "both"

def _not_implemented( name: str ) -> None:
    typer.echo( f"'{name}' is not implemented yet", err = True )
    raise typer.Exit( code = 1 )

def _timestamp() -> str:
    return datetime.now( timezone.utc ).strftime( "%Y%m%dT%H%M%S" )

def _git_sha() -> str | None:
    try:
        completed = subprocess.run( [ "git", "rev-parse", "HEAD" ], capture_output = True, text = True, check = True )
    except ( OSError, subprocess.CalledProcessError ):
        return None
    return completed.stdout.strip() or None

def _run_config( config_path: Path, split: Split, results_dir: Path ) -> None:
    try:
        cfg = load_config( config_path )
    except ConfigError as error:
        typer.echo( f"Invalid config: {error}", err = True )
        raise typer.Exit( code = 1 )

    if not isinstance( cfg, ( ClassicalConfig, BaselineConfig ) ):
        _not_implemented( f"family '{cfg.family}'" )

    results = run_experiment(
        cfg,
        load_emotion( seed = cfg.seed ),
        config_path.stem,
        results_dir,
        _timestamp(),
        git_sha = _git_sha(),
        splits_to_save = split.value,
    )
    for result in results:
        typer.echo( f"{result.run_id}: macro_f1 = {result.metrics[ 'macro_f1' ]:.4f}" )

@app.command()
def run(
    config: Annotated[ Path, typer.Argument( exists = True, dir_okay = False, help = "Experiment YAML config." ) ],
    output_dir: Annotated[ Path, typer.Option( help = "Where checkpoints and large artefacts go." ) ] = Path( "outputs" ),
    split: Annotated[ Split, typer.Option( help = "Which split(s) to evaluate on." ) ] = Split.both,
    results_dir: Annotated[ Path, typer.Option( help = "Where RunResult JSON and predictions go." ) ] = Path( "results" ),
) -> None:
    """Run one experiment from a config file"""
    _run_config( config, split, results_dir )

@app.command()
def grid(
    config_dir: Annotated[ Path, typer.Argument( exists = True, file_okay = False, help = "Folder of YAML configs." ) ],
    output_dir: Annotated[ Path, typer.Option( help = "Where checkpoints and large artefacts go." ) ] = Path( "outputs" ),
    results_dir: Annotated[ Path, typer.Option( help = "Where RunResult JSON and predictions go." ) ] = Path( "results" ),
) -> None:
    """Run every config in a folder"""
    configs = sorted( [ *config_dir.glob( "*.yaml" ), *config_dir.glob( "*.yml" ) ] )
    if not configs:
        typer.echo( f"No YAML configs in {config_dir}", err = True )
        raise typer.Exit( code = 1 )

    for config_path in configs:
        typer.echo( f"== {config_path.name}" )
        _run_config( config_path, Split.both, results_dir )

@app.command()
def report() -> None:
    "Rebuild summary tables and figures from results"
    _not_implemented( "report" )
