import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted( ( Path( __file__ ).parent.parent / "notebooks" ).glob( "*.ipynb" ) )

@pytest.mark.parametrize( "path", NOTEBOOKS, ids = lambda path: path.name )
def test_notebook_is_valid_and_its_code_cells_compile( path ):
    notebook = json.loads( path.read_text( encoding = "utf-8" ) )
    assert notebook[ "nbformat" ] == 4
    for number, cell in enumerate( notebook[ "cells" ] ):
        assert cell[ "cell_type" ] in ( "code", "markdown" )
        if cell[ "cell_type" ] == "code":
            source = "".join( line for line in cell[ "source" ] if not line.lstrip().startswith( ( "%", "!" ) ) )
            compile( source, f"{ path.name }[ { number } ]", "exec" )

def test_there_is_a_results_notebook():
    assert "02_results.ipynb" in [ path.name for path in NOTEBOOKS ]
