"""Figures for the report. Everything is drawn from saved RunResults, never by re-training.

Design: light surface, text in ink tokens (never the series colour), recessive grid, one-hue
sequential ramp for the confusion matrices, and the first three slots of the categorical palette
(validated all-pairs) for classical / transformer / zero-shot. The aqua slot is below 3:1 contrast
on the surface, so bars carry value labels and the scatter uses a different marker per family."""
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from emotion_classifier.data import LABELS
from emotion_classifier.metrics import row_normalise
from emotion_classifier.report import latest_runs, summarise
from emotion_classifier.results import RunResult, slugify

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dcdbd6"
BLUE_RAMP = [ "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b" ]

# Family -> experiments, in the order they appear in every figure. E0 (dummy baselines) is not a family.
FAMILIES = { "Classical": ( "E1", "E2" ), "Transformer": ( "E3", ), "Zero-shot": ( "E4", ) }
FAMILY_COLOUR = { "Classical": "#2a78d6", "Transformer": "#eb6834", "Zero-shot": "#1baf7a" }
FAMILY_MARKER = { "Classical": "o", "Transformer": "s", "Zero-shot": "^" }

def _family_of( experiment: str ) -> str | None:
    return next( ( family for family, experiments in FAMILIES.items() if experiment in experiments ), None )

def _test_runs( results: list[ RunResult ], model: str ) -> list[ RunResult ]:
    return [ result for result in latest_runs( results, "test" ) if result.model == model ]

def _require_test_results( results: list[ RunResult ] ) -> None:
    if not latest_runs( results, "test" ):
        raise ValueError( "no test results to plot" )

def select_models( results: list[ RunResult ] ) -> dict[ str, str ]:
    """The model shown per family: best mean validation macro-F1 (never test, TRD §4). Zero-shot has no
    validation run, so it is taken as-is when it is the only model; with several, say which to plot."""
    validation: dict[ str, list[ float ] ] = {}
    for result in latest_runs( results, "val" ):
        validation.setdefault( result.model, [] ).append( result.metrics[ "macro_f1" ] )

    selected = {}
    for family, experiments in FAMILIES.items():
        candidates = sorted( { r.model for r in latest_runs( results, "test" ) if r.experiment in experiments } )
        scores = { model: sum( validation[ model ] ) / len( validation[ model ] ) for model in candidates if model in validation }
        if scores:
            selected[ family ] = max( sorted( scores ), key = scores.get )
        elif len( candidates ) == 1:
            selected[ family ] = candidates[ 0 ]
        elif candidates:
            raise ValueError( f"{ family }: { candidates } have no validation results, so pass `selected` to choose one" )
    return selected

def pooled_confusion( results: list[ RunResult ], model: str ) -> list[ list[ int ] ]:
    """Test-split confusion matrix of `model`, summed over its seeds."""
    runs = _test_runs( results, model )
    if not runs:
        raise ValueError( f"no test results for model { repr( model ) }" )
    return np.sum( [ run.metrics[ "confusion" ] for run in runs ], axis = 0 ).astype( int ).tolist()

def mean_per_class_f1( results: list[ RunResult ], model: str ) -> dict[ str, float ]:
    """Test-split per-class F1 of `model`, averaged over its seeds."""
    runs = _test_runs( results, model )
    if not runs:
        raise ValueError( f"no test results for model { repr( model ) }" )
    return { label: float( np.mean( [ run.metrics[ "per_class" ][ label ][ "f1" ] for run in runs ] ) ) for label in LABELS }

def _save( figure: Figure, path: Path ) -> Path:
    path = Path( path )
    path.parent.mkdir( parents = True, exist_ok = True )
    figure.savefig( path, dpi = 200, facecolor = SURFACE )
    return path

def _style_axes( ax ) -> None:
    ax.set_facecolor( SURFACE )
    for side in ( "top", "right" ):
        ax.spines[ side ].set_visible( False )
    for side in ( "left", "bottom" ):
        ax.spines[ side ].set_color( GRID )
    ax.tick_params( colors = INK_SECONDARY, labelsize = 9 )

def plot_confusion_matrix( confusion: list[ list[ int ] ], title: str, path: Path ) -> Path:
    """Row-normalised confusion matrix (each row sums to 1: the share of that true class predicted as each label)."""
    shares = np.array( row_normalise( confusion ) )
    cmap = LinearSegmentedColormap.from_list( "blue", BLUE_RAMP )

    figure = Figure( figsize = ( 5.8, 5.0 ), facecolor = SURFACE )
    ax = figure.subplots()
    image = ax.imshow( shares, cmap = cmap, vmin = 0.0, vmax = 1.0 )

    ax.set_xticks( range( len( LABELS ) ), LABELS, rotation = 30, ha = "right" )
    ax.set_yticks( range( len( LABELS ) ), LABELS )
    ax.set_xlabel( "Predicted", color = INK_SECONDARY )
    ax.set_ylabel( "True", color = INK_SECONDARY )
    ax.set_title( title, loc = "left", color = INK, fontsize = 11 )
    ax.tick_params( colors = INK_SECONDARY, labelsize = 9, length = 0 )
    for spine in ax.spines.values():
        spine.set_visible( False )

    # 2px surface gaps between cells.
    ax.set_xticks( np.arange( -0.5, len( LABELS ), 1 ), minor = True )
    ax.set_yticks( np.arange( -0.5, len( LABELS ), 1 ), minor = True )
    ax.grid( which = "minor", color = SURFACE, linewidth = 2 )
    ax.tick_params( which = "minor", length = 0 )

    for row in range( len( LABELS ) ):
        for column in range( len( LABELS ) ):
            value = shares[ row, column ]
            ax.text(
                column, row, "0" if value == 0 else f"{ value:.2f}", ha = "center", va = "center", fontsize = 9,
                color = "white" if value >= 0.6 else INK,
            )

    colourbar = figure.colorbar( image, ax = ax, fraction = 0.046, pad = 0.03 )
    colourbar.set_label( "Share of the true class", color = INK_SECONDARY, fontsize = 9 )
    colourbar.outline.set_visible( False )
    colourbar.ax.tick_params( colors = INK_SECONDARY, labelsize = 8 )

    figure.tight_layout()
    return _save( figure, path )

def plot_confusion_matrices( results: list[ RunResult ], figures_dir: Path, selected: dict[ str, str ] | None = None ) -> list[ Path ]:
    """One row-normalised confusion matrix per family: figures/confusion_<family>.png. Seeds are pooled."""
    _require_test_results( results )
    paths = []
    for family, model in ( selected if selected is not None else select_models( results ) ).items():
        seeds = len( _test_runs( results, model ) )
        suffix = f", pooled over { seeds } seeds" if seeds > 1 else ""
        title = f"{ family }: { model } (test{ suffix })"
        paths.append( plot_confusion_matrix( pooled_confusion( results, model ), title, Path( figures_dir ) / f"confusion_{ slugify( family ) }.png" ) )
    return paths

def plot_per_class_f1( results: list[ RunResult ], path: Path, selected: dict[ str, str ] | None = None ) -> Path:
    """Per-class F1 on test, one bar per family's selected model (mean over seeds), classes top to bottom."""
    _require_test_results( results )
    selected = selected if selected is not None else select_models( results )
    families = list( selected )

    figure = Figure( figsize = ( 7.2, 5.6 ), facecolor = SURFACE )
    ax = figure.subplots()
    _style_axes( ax )

    height = 0.24
    centres = np.arange( len( LABELS ) )[ ::-1 ]  # first class at the top
    for index, family in enumerate( families ):
        scores = mean_per_class_f1( results, selected[ family ] )
        offset = ( index - ( len( families ) - 1 ) / 2 ) * -height
        positions = centres + offset
        values = [ scores[ label ] for label in LABELS ]
        ax.barh( positions, values, height = height, color = FAMILY_COLOUR[ family ], edgecolor = SURFACE, linewidth = 1.5, zorder = 3 )
        for position, value in zip( positions, values ):
            ax.text( value + 0.01, position, f"{ value:.2f}", va = "center", ha = "left", fontsize = 8, color = INK_SECONDARY )

    ax.set_yticks( centres, LABELS )
    ax.set_xlim( 0, 1.08 )
    ax.set_xlabel( "F1 (test)", color = INK_SECONDARY )
    ax.xaxis.grid( True, color = GRID, linewidth = 0.8, zorder = 0 )
    ax.set_axisbelow( True )
    ax.tick_params( axis = "y", length = 0 )
    ax.set_title( "Per-class F1 on the test set", loc = "left", color = INK, fontsize = 11 )

    handles = [ Line2D( [], [], marker = "s", linestyle = "", markersize = 9, color = FAMILY_COLOUR[ family ], label = f"{ family }: { selected[ family ] }" ) for family in families ]
    figure.legend( handles = handles, loc = "lower center", ncol = min( len( handles ), 3 ), frameon = False, fontsize = 9, labelcolor = INK )
    figure.tight_layout( rect = ( 0, 0.06, 1, 1 ) )
    return _save( figure, path )

def plot_f1_vs_cost( results: list[ RunResult ], path: Path, selected: dict[ str, str ] | None = None ) -> Path:
    """Test macro-F1 against inference time per 1,000 examples (log x). One marker per model, colour and shape by family;
    bars show ±1 std over seeds. Dummy baselines are left out. Labelled: each family's selected model, plus every
    transformer and zero-shot model."""
    _require_test_results( results )
    selected = selected if selected is not None else select_models( results )
    rows = [ row for row in summarise( results ) if _family_of( row[ "experiment" ] ) and row[ "infer_seconds_per_1k_mean" ] ]

    figure = Figure( figsize = ( 7.2, 5.0 ), facecolor = SURFACE )
    ax = figure.subplots()
    _style_axes( ax )
    ax.set_xscale( "log" )
    ax.grid( True, color = GRID, linewidth = 0.8, zorder = 0 )
    ax.set_axisbelow( True )

    for family in FAMILIES:
        family_rows = [ row for row in rows if _family_of( row[ "experiment" ] ) == family ]
        if not family_rows:
            continue
        ax.errorbar(
            [ row[ "infer_seconds_per_1k_mean" ] for row in family_rows ],
            [ row[ "macro_f1_mean" ] for row in family_rows ],
            yerr = [ row[ "macro_f1_std" ] or 0.0 for row in family_rows ],
            fmt = "none", ecolor = FAMILY_COLOUR[ family ], elinewidth = 1.5, capsize = 3, zorder = 2,
        )
        ax.scatter(
            [ row[ "infer_seconds_per_1k_mean" ] for row in family_rows ],
            [ row[ "macro_f1_mean" ] for row in family_rows ],
            s = 70, marker = FAMILY_MARKER[ family ], color = FAMILY_COLOUR[ family ],
            edgecolors = SURFACE, linewidths = 1.5, zorder = 3,
        )

    labelled = [
        row for row in rows
        if _family_of( row[ "experiment" ] ) != "Classical" or row[ "model" ] == selected.get( "Classical" )
    ]
    low, high = np.log10( ax.get_xlim() )
    for row in labelled:
        x, y = row[ "infer_seconds_per_1k_mean" ], row[ "macro_f1_mean" ]
        near_above = any(
            other is not row and other[ "macro_f1_mean" ] > y and abs( np.log10( other[ "infer_seconds_per_1k_mean" ] ) - np.log10( x ) ) < 1
            for other in labelled
        )
        near_right_edge = ( np.log10( x ) - low ) / ( high - low ) > 0.85
        ax.annotate(
            row[ "model" ], ( x, y ),
            xytext = ( -4 if near_right_edge else 8, -14 if near_above else 8 ), textcoords = "offset points",
            ha = "right" if near_right_edge else "left", fontsize = 8.5, color = INK,
        )

    ax.set_xlabel( "Inference time per 1,000 tweets (seconds, log scale)", color = INK_SECONDARY )
    ax.set_ylabel( "Macro-F1 (test)", color = INK_SECONDARY )
    ax.set_title( "Accuracy against inference cost", loc = "left", color = INK, fontsize = 11 )

    handles = [
        Line2D( [], [], marker = FAMILY_MARKER[ family ], linestyle = "", markersize = 8, color = FAMILY_COLOUR[ family ], label = family )
        for family in FAMILIES if any( _family_of( row[ "experiment" ] ) == family for row in rows )
    ]
    ax.legend( handles = handles, loc = "lower left", frameon = False, fontsize = 9, labelcolor = INK )
    figure.text( 0.01, 0.01, "Dummy baselines omitted. Error bars: ±1 std over seeds.", fontsize = 7.5, color = INK_SECONDARY )
    figure.tight_layout( rect = ( 0, 0.03, 1, 1 ) )
    return _save( figure, path )

def make_all( results: list[ RunResult ], figures_dir: Path, selected: dict[ str, str ] | None = None ) -> list[ Path ]:
    """Regenerate every figure into figures_dir (existing files are overwritten: figures are derived output)."""
    _require_test_results( results )
    figures_dir = Path( figures_dir )
    return [
        *plot_confusion_matrices( results, figures_dir, selected ),
        plot_per_class_f1( results, figures_dir / "per_class_f1.png", selected ),
        plot_f1_vs_cost( results, figures_dir / "f1_vs_cost.png", selected ),
    ]
