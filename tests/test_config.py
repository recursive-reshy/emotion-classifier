from pathlib import Path

import pytest

from emotion_classifier.config import (
    BaselineConfig,
    ClassicalConfig,
    ConfigError,
    LLMConfig,
    TransformerConfig,
    ZeroShotConfig,
    load_config,
)

CLASSICAL = """
experiment: E1
family: classical
vectorizer: { kind: tfidf, ngram_range: [ 1, 2 ], min_df: 2, max_features: null, stop_words: null }
classifier: { kind: logreg, class_weight: null }
tuning: { param: C, values: [ 0.1, 0.3, 1, 3, 10 ] }
seed: 42
"""

TRANSFORMER = """
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

ZEROSHOT = """
experiment: E4
family: zeroshot
model_name: facebook/bart-large-mnli
hypothesis_template: "This text expresses {}."
batch_size: 16
seed: 42
"""

BASELINE = """
experiment: E0
family: baseline
strategy: most_frequent
seed: 42
"""

def _write( tmp_path, text: str, name: str = "cfg.yaml" ):
    path = tmp_path / name
    path.write_text( text )
    return path

def test_valid_classical_config( tmp_path ):
    cfg = load_config( _write( tmp_path, CLASSICAL ) )
    assert isinstance( cfg, ClassicalConfig )
    assert cfg.experiment == "E1"
    assert cfg.vectorizer.kind == "tfidf"
    assert cfg.vectorizer.ngram_range == ( 1, 2 )
    assert cfg.vectorizer.min_df == 2
    assert cfg.vectorizer.max_features is None
    assert cfg.vectorizer.stop_words is None
    assert cfg.classifier.kind == "logreg"
    assert cfg.classifier.class_weight is None
    assert cfg.tuning.param == "C"
    assert cfg.tuning.values == [ 0.1, 0.3, 1, 3, 10 ]
    assert cfg.seed == 42

def test_classical_nullable_keys_may_be_omitted( tmp_path ):
    text = CLASSICAL.replace( ", max_features: null, stop_words: null", "" ).replace( ", class_weight: null", "" )
    cfg = load_config( _write( tmp_path, text ) )
    assert cfg.vectorizer.max_features is None
    assert cfg.vectorizer.stop_words is None
    assert cfg.classifier.class_weight is None

def test_valid_transformer_config( tmp_path ):
    cfg = load_config( _write( tmp_path, TRANSFORMER ) )
    assert isinstance( cfg, TransformerConfig )
    assert cfg.model_name == "distilbert-base-uncased"
    assert cfg.max_length == 88
    assert cfg.lr == 5.0e-5
    assert cfg.epochs == 4
    assert cfg.seeds == [ 13, 42, 7 ]

def test_valid_zeroshot_config( tmp_path ):
    cfg = load_config( _write( tmp_path, ZEROSHOT ) )
    assert isinstance( cfg, ZeroShotConfig )
    assert cfg.model_name == "facebook/bart-large-mnli"
    assert cfg.hypothesis_template == "This text expresses {}."

def test_valid_baseline_config( tmp_path ):
    cfg = load_config( _write( tmp_path, BASELINE ) )
    assert isinstance( cfg, BaselineConfig )
    assert cfg.strategy == "most_frequent"

def test_unknown_top_level_key_raises( tmp_path ):
    with pytest.raises( ConfigError, match = "learning_rate" ):
        load_config( _write( tmp_path, TRANSFORMER + "learning_rate: 1\n" ) )

def test_unknown_nested_key_raises( tmp_path ):
    with pytest.raises( ConfigError, match = "bogus" ):
        load_config( _write( tmp_path, CLASSICAL.replace( "min_df: 2", "min_df: 2, bogus: 1" ) ) )

@pytest.mark.parametrize( "key", [ "model_name", "max_length", "lr", "epochs", "batch_size", "seeds" ] )
def test_missing_required_transformer_key_raises( tmp_path, key ):
    text = "\n".join( line for line in TRANSFORMER.splitlines() if not line.startswith( f"{ key }:" ) )
    with pytest.raises( ConfigError, match = key ):
        load_config( _write( tmp_path, text ) )

def test_missing_required_nested_key_raises( tmp_path ):
    with pytest.raises( ConfigError, match = "kind" ):
        load_config( _write( tmp_path, CLASSICAL.replace( "kind: logreg, ", "" ) ) )

def test_missing_family_raises( tmp_path ):
    with pytest.raises( ConfigError, match = "family" ):
        load_config( _write( tmp_path, BASELINE.replace( "family: baseline\n", "" ) ) )

def test_unknown_family_raises( tmp_path ):
    with pytest.raises( ConfigError, match = "family" ):
        load_config( _write( tmp_path, BASELINE.replace( "baseline", "quantum" ) ) )

@pytest.mark.parametrize( "old, new", [
    ( "kind: tfidf", "kind: word2vec" ),
    ( "kind: logreg", "kind: forest" ),
] )
def test_invalid_choice_raises( tmp_path, old, new ):
    with pytest.raises( ConfigError ):
        load_config( _write( tmp_path, CLASSICAL.replace( old, new ) ) )

def test_bad_ngram_range_raises( tmp_path ):
    with pytest.raises( ConfigError, match = "ngram_range" ):
        load_config( _write( tmp_path, CLASSICAL.replace( "[ 1, 2 ]", "[ 2, 1 ]" ) ) )

def test_non_mapping_yaml_raises( tmp_path ):
    with pytest.raises( ConfigError ):
        load_config( _write( tmp_path, "- just\n- a list\n" ) )

def test_wrong_type_raises( tmp_path ):
    with pytest.raises( ConfigError, match = "epochs" ):
        load_config( _write( tmp_path, TRANSFORMER.replace( "epochs: 4", "epochs: four" ) ) )

def test_train_fraction_defaults_to_none( tmp_path ):
    assert load_config( _write( tmp_path, CLASSICAL ) ).train_fraction is None
    assert load_config( _write( tmp_path, TRANSFORMER ) ).train_fraction is None

@pytest.mark.parametrize( "text", [ CLASSICAL, TRANSFORMER ] )
@pytest.mark.parametrize( "value, expected", [ ( "0.25", 0.25 ), ( "1", 1 ), ( "0.1", 0.1 ) ] )
def test_train_fraction_parses( tmp_path, text, value, expected ):
    assert load_config( _write( tmp_path, text + f"train_fraction: { value }\n" ) ).train_fraction == expected

@pytest.mark.parametrize( "text", [ CLASSICAL, TRANSFORMER ] )
@pytest.mark.parametrize( "value", [ "0", "-0.5", "1.5", "'half'", "true" ] )
def test_train_fraction_outside_zero_one_or_wrong_type_raises( tmp_path, text, value ):
    with pytest.raises( ConfigError, match = "train_fraction" ):
        load_config( _write( tmp_path, text + f"train_fraction: { value }\n" ) )

def test_train_fraction_is_not_allowed_on_zeroshot_or_baseline( tmp_path ):
    for text in ( ZEROSHOT, BASELINE ):
        with pytest.raises( ConfigError, match = "train_fraction" ):
            load_config( _write( tmp_path, text + "train_fraction: 0.5\n" ) )

CONFIG_FILES = sorted( ( Path( __file__ ).parent.parent / "configs" ).rglob( "*.yaml" ) )

@pytest.mark.parametrize( "path", CONFIG_FILES, ids = lambda path: f"{ path.parent.name }/{ path.name }" )
def test_every_shipped_config_loads( path ):
    load_config( path )

def test_empty_seeds_raises( tmp_path ):
    with pytest.raises( ConfigError, match = "seeds" ):
        load_config( _write( tmp_path, TRANSFORMER.replace( "[ 13, 42, 7 ]", "[]" ) ) )

LLM = """
experiment: E5
family: llm
model_name: Qwen/Qwen3-4B-Instruct-2507
batch_size: 16
max_new_tokens: 8
seed: 42
"""

def test_valid_llm_config( tmp_path ):
    cfg = load_config( _write( tmp_path, LLM ) )
    assert isinstance( cfg, LLMConfig )
    assert cfg.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.max_new_tokens == 8

@pytest.mark.parametrize( "key", [ "model_name", "batch_size", "max_new_tokens", "seed" ] )
def test_missing_required_llm_key_raises( tmp_path, key ):
    text = "\n".join( line for line in LLM.splitlines() if not line.startswith( key + ":" ) )
    with pytest.raises( ConfigError, match = key ):
        load_config( _write( tmp_path, text ) )

def test_llm_config_rejects_unknown_key_and_nonpositive_sizes( tmp_path ):
    with pytest.raises( ConfigError, match = "temperature" ):
        load_config( _write( tmp_path, LLM + "temperature: 0.7\n" ) )  # decoding is always greedy
    for line in ( "batch_size: 0", "max_new_tokens: 0" ):
        key = line.split( ":" )[ 0 ]
        with pytest.raises( ConfigError, match = key ):
            load_config( _write( tmp_path, "\n".join( line if row.startswith( key + ":" ) else row for row in LLM.splitlines() ) ) )
