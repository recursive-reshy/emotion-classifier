from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

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

@app.command()
def run(
    config: Annotated[ Path, typer.Argument( exists = True, dir_okay = False, help = "Experiment YAML config." ) ],
    output_dir: Annotated[ Path, typer.Option( help = "Where checkpoints and large artefacts go." ) ] = Path( "outputs" ),
    split: Annotated[ Split, typer.Option( help = "Which split(s) to evaluate on." ) ] = Split.both,
) -> None:
    """Run one experiment from a config file"""
    _not_implemented( "run" )

@app.command()
def grid(
    config_dir: Annotated[ Path, typer.Argument( exists = True, file_okay = False, help = "Folder of YAML configs." ) ],
    output_dir: Annotated[ Path, typer.Option( help = "Where checkpoints and large artefacts go." ) ] = Path( "outputs" ),
) -> None:
    """Ryb every config in a folder"""
    _not_implemented( "grid" )

@app.command()
def report() -> None:
    "Rebuild summary tables and figures from results"
    _not_implemented( "report" )