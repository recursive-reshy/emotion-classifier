import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from emotion_classifier.classical import run_experiment
from emotion_classifier import plots, transformer, zeroshot
from emotion_classifier.config import BaselineConfig, ClassicalConfig, ConfigError, TransformerConfig, ZeroShotConfig, load_config
from emotion_classifier.data import load_emotion
from emotion_classifier.report import summarise, write_summary
from emotion_classifier.results import load_results

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

def _run_config( config_path: Path, split: Split, results_dir: Path, output_dir: Path ) -> None:
    try:
        cfg = load_config( config_path )
    except ConfigError as error:
        typer.echo( f"Invalid config: {error}", err = True )
        raise typer.Exit( code = 1 )

    if isinstance( cfg, TransformerConfig ):
        results = transformer.run_transformer_experiment(
            cfg,
            load_emotion(),
            config_path.stem,
            results_dir,
            output_dir,
            _timestamp(),
            git_sha = _git_sha(),
            splits_to_save = split.value,
        )
    elif isinstance( cfg, ZeroShotConfig ):
        results = zeroshot.run_zeroshot_experiment(  # always test only: no tuning (TRD §4.5), so --split is ignored
            cfg,
            load_emotion(),
            config_path.stem,
            results_dir,
            _timestamp(),
            git_sha = _git_sha(),
        )
    elif isinstance( cfg, ( ClassicalConfig, BaselineConfig ) ):
        results = run_experiment(
            cfg,
            load_emotion( seed = cfg.seed ),
            config_path.stem,
            results_dir,
            _timestamp(),
            git_sha = _git_sha(),
            splits_to_save = split.value,
        )
    else:
        _not_implemented( f"family '{cfg.family}'" )

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
    _run_config( config, split, results_dir, output_dir )

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
        _run_config( config_path, Split.both, results_dir, output_dir )

@app.command()
def report(
    results_dir: Annotated[ Path, typer.Option( help = "Where the RunResult JSON files are read from." ) ] = Path( "results" ),
    figures_dir: Annotated[ Path, typer.Option( help = "Where figures are written." ) ] = Path( "figures" ),
    select: Annotated[ list[ str ] | None, typer.Option(
        help = "Model to show for a family, as Family=model (repeatable), e.g. --select Transformer=roberta_base. "
               "By default each family shows its best model on validation macro-F1."
    ) ] = None,
) -> None:
    """Rebuild summary tables and figures from results/ (no training, no dataset)"""
    overrides = {}
    for item in select or []:
        family, separator, model = item.partition( "=" )
        if not separator or not family or not model:
            typer.echo( f"Invalid --select {item!r}: expected Family=model", err = True )
            raise typer.Exit( code = 1 )
        overrides[ family ] = model

    results = load_results( results_dir )
    if not any( result.split == "test" for result in results ):
        typer.echo( f"No test results in {results_dir}; run some experiments first", err = True )
        raise typer.Exit( code = 1 )

    try:
        selected = plots.resolve_selection( results, overrides )
    except ValueError as error:
        typer.echo( f"Invalid selection: {error}", err = True )
        raise typer.Exit( code = 1 )

    csv_path, md_path = Path( results_dir ) / "summary.csv", Path( results_dir ) / "summary.md"
    write_summary( summarise( results ), csv_path, md_path )
    figures = plots.make_all( results, figures_dir, selected )

    typer.echo( f"Wrote {csv_path} and {md_path}" )
    typer.echo( f"Wrote {len( figures )} figures to {figures_dir}" )
    typer.echo( "Figures show: " + ", ".join( f"{family}: {model}" for family, model in selected.items() ) )
