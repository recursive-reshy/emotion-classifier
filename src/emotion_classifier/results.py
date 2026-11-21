import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass
class RunResult:
    run_id: str
    experiment: str
    model: str                      # config file stem, e.g. "tfidf_logreg_12"
    config: dict
    split: str                      # "val" or "test"
    metrics: dict
    train_seconds: float | None
    infer_seconds_per_1k: float | None
    n_params: int | None
    device: str
    seed: int
    git_sha: str | None
    predictions_path: str | None    # CSV of text, true, pred

def slugify( name: str ) -> str:
    return re.sub( r"[^a-z0-9._]+", "-", name.lower() ).strip( "-" )

def make_run_id( experiment: str, model: str, split: str, seed: int, timestamp: str ) -> str:
    return f"{ experiment }-{ slugify( model ) }-{ split }-{ seed }-{ timestamp }"

def save_result( result: RunResult, results_dir: Path ) -> Path:
    """Write results/<run_id>.json. Never overwrites: raises FileExistsError."""
    results_dir = Path( results_dir )
    results_dir.mkdir( parents = True, exist_ok = True )
    path = results_dir / f"{ result.run_id }.json"
    with path.open( "x", encoding = "utf-8" ) as handle:
        json.dump( asdict( result ), handle, indent = 2 )
    return path

def load_results( results_dir: Path ) -> list[ RunResult ]:
    results_dir = Path( results_dir )
    if not results_dir.is_dir():
        return []
    results = []
    for path in sorted( results_dir.glob( "*.json" ) ):
        with path.open( encoding = "utf-8" ) as handle:
            results.append( RunResult( **json.load( handle ) ) )
    return sorted( results, key = lambda result: result.run_id )

def save_predictions(
    run_id: str,
    texts: list[ str ],
    y_true: list[ int ],
    y_pred: list[ int ],
    results_dir: Path,
) -> Path:
    """Write results/preds/<run_id>.csv with columns text,true,pred. Never overwrites."""
    if not len( texts ) == len( y_true ) == len( y_pred ):
        raise ValueError( "texts, y_true and y_pred must have the same length" )
    preds_dir = Path( results_dir ) / "preds"
    preds_dir.mkdir( parents = True, exist_ok = True )
    path = preds_dir / f"{ run_id }.csv"
    with path.open( "x", encoding = "utf-8", newline = "" ) as handle:
        writer = csv.writer( handle, lineterminator = "\n" )
        writer.writerow( [ "text", "true", "pred" ] )
        writer.writerows( zip( texts, y_true, y_pred ) )
    return path
