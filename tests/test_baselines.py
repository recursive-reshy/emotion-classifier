from pathlib import Path

import pytest

from emotion_classifier.classical import build_baseline
from emotion_classifier.config import BaselineConfig, load_config

BASELINE_DIR = Path( __file__ ).parent.parent / "configs" / "baselines"

def _cfg( strategy: str, seed: int = 42 ) -> BaselineConfig:
    return BaselineConfig( experiment = "E0", family = "baseline", strategy = strategy, seed = seed )

def test_majority_baseline_predicts_a_single_class( tiny_splits ):
    model = build_baseline( _cfg( "most_frequent" ) )
    model.fit( tiny_splits.train_texts, tiny_splits.train_labels )
    assert len( set( model.predict( tiny_splits.test_texts ) ) ) == 1

def test_stratified_baseline_predicts_several_classes_and_is_seeded( tiny_splits ):
    def predictions( seed: int ) -> list[ int ]:
        model = build_baseline( _cfg( "stratified", seed ) )
        model.fit( tiny_splits.train_texts, tiny_splits.train_labels )
        return list( model.predict( tiny_splits.train_texts ) )

    assert len( set( predictions( 42 ) ) ) > 1
    assert predictions( 42 ) == predictions( 42 )

def test_unknown_strategy_is_rejected():
    with pytest.raises( ValueError ):
        build_baseline( _cfg( "uniform" ) )

@pytest.mark.parametrize( "name", [ "majority", "stratified" ] )
def test_baseline_config_files_load( name ):
    cfg = load_config( BASELINE_DIR / f"{ name }.yaml" )
    assert isinstance( cfg, BaselineConfig )
    assert cfg.experiment == "E0"
