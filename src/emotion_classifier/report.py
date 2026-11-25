import csv
import statistics
from collections import defaultdict
from pathlib import Path

from emotion_classifier.results import RunResult

# ( summary key, how to read it from a RunResult )
MEASURES = {
    "accuracy": lambda result: result.metrics[ "accuracy" ],
    "macro_f1": lambda result: result.metrics[ "macro_f1" ],
    "weighted_f1": lambda result: result.metrics[ "weighted_f1" ],
    "train_seconds": lambda result: result.train_seconds,
    "infer_seconds_per_1k": lambda result: result.infer_seconds_per_1k,
}

SUMMARY_COLUMNS = [
    "experiment", "model", "n_seeds",
    *[ f"{ name }_{ stat }" for name in MEASURES for stat in ( "mean", "std" ) ],
    "n_params",
]

def _mean_std( values: list[ float | None ] ) -> tuple[ float | None, float | None ]:
    """Mean and sample std (None for a single value); ( None, None ) when any value is missing."""
    if not values or any( value is None for value in values ):
        return None, None
    return statistics.fmean( values ), ( statistics.stdev( values ) if len( values ) > 1 else None )

def latest_runs( results: list[ RunResult ], split: str = "test" ) -> list[ RunResult ]:
    """Results for one split, keeping only the latest run of each ( experiment, model, seed ),
    so a re-run replaces the earlier run instead of counting as another seed."""
    latest: dict[ tuple, RunResult ] = {}
    for result in sorted( results, key = lambda result: result.run_id ):
        if result.split == split:
            latest[ ( result.experiment, result.model, result.seed ) ] = result
    return list( latest.values() )

def summarise( results: list[ RunResult ] ) -> list[ dict ]:
    """One row per ( experiment, model), test split only. Mean and sample std over seeds
    (latest run per seed). Rows are ordered by experiment, then best macro-F1 first."""
    groups: dict[ tuple, list[ RunResult ] ] = defaultdict( list )
    for result in latest_runs( results, "test" ):
        groups[ ( result.experiment, result.model ) ].append( result )

    rows = []
    for ( experiment, model ), group in groups.items():
        row = { "experiment": experiment, "model": model, "n_seeds": len( group ) }
        for name, read in MEASURES.items():
            row[ f"{ name }_mean" ], row[ f"{ name }_std" ] = _mean_std( [ read( result ) for result in group ] )
        row[ "n_params" ] = group[ 0 ].n_params
        rows.append( row )

    return sorted( rows, key = lambda row: ( row[ "experiment" ], -row[ "macro_f1_mean" ], row[ "model" ] ) )

def _time_decimals( seconds: float ) -> int:
    """Enough decimals to show about three significant figures, so millisecond-scale costs don't print as 0."""
    for threshold, decimals in ( ( 100, 0 ), ( 10, 1 ), ( 1, 2 ) ):
        if seconds >= threshold:
            return decimals
    return 3

def _cell( row: dict, name: str, decimals: int | None ) -> str:
    """mean ± std; `decimals = None` picks the precision from the mean (for times)."""
    mean, std = row[ f"{ name }_mean" ], row[ f"{ name }_std" ]
    if mean is None:
        return "—"
    if decimals is None:
        decimals = _time_decimals( mean )
    text = f"{ mean:.{ decimals }f}"
    return text if std is None else f"{ text } ± { std:.{ decimals }f}"

def to_markdown( rows: list[ dict ] ) -> str:
    headers = [ "Experiment", "Model", "Seeds", "Accuracy", "macro-F1", "weighted-F1", "Train (s)", "Infer / 1k (s)", "Params" ]
    lines = [ "| " + " | ".join( headers ) + " |", "|" + "|".join( [ "---" ] * len( headers ) ) + "|" ]
    for row in rows:
        cells = [
            row[ "experiment" ], row[ "model" ], str( row[ "n_seeds" ] ),
            _cell( row, "accuracy", 3 ), _cell( row, "macro_f1", 3 ), _cell( row, "weighted_f1", 3 ),
            _cell( row, "train_seconds", None ), _cell( row, "infer_seconds_per_1k", None ),
            "—" if row[ "n_params" ] is None else f"{ row[ 'n_params' ]:,}",
        ]
        lines.append( "| " + " | ".join( cells ) + " |" )
    return "\n".join( lines ) + "\n"

def write_summary( rows: list[ dict ], csv_path: Path, md_path: Path ) -> None:
    """Write the summary as CSV and Markdown. Unlike run results these are regenerated, so they are overwritten."""
    csv_path, md_path = Path( csv_path ), Path( md_path )
    csv_path.parent.mkdir( parents = True, exist_ok = True )
    md_path.parent.mkdir( parents = True, exist_ok = True )

    with csv_path.open( "w", encoding = "utf-8", newline = "" ) as handle:
        writer = csv.DictWriter( handle, fieldnames = SUMMARY_COLUMNS, lineterminator = "\n" )
        writer.writeheader()
        writer.writerows( rows )
    md_path.write_text( to_markdown( rows ), encoding = "utf-8" )
