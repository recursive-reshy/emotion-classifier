import types
import typing
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

class ConfigError( ValueError ):
    pass

VECTORIZER_KINDS = ( "bow", "tfidf" )
CLASSIFIER_KINDS = ( "nb", "logreg", "svm" )
BASELINE_STRATEGIES = ( "most_frequent", "stratified" )

@dataclass( frozen = True )
class VectorizerConfig:
    kind: str
    ngram_range: tuple[ int, int ]
    min_df: int
    max_features: int | None = None
    stop_words: str | None = None

@dataclass( frozen = True )
class ClassifierConfig:
    kind: str
    class_weight: str | None = None

@dataclass( frozen = True )
class TuningConfig:
    param: str
    values: list[ float ]

@dataclass( frozen = True )
class ClassicalConfig:
    experiment: str
    family: str
    vectorizer: VectorizerConfig
    classifier: ClassifierConfig
    tuning: TuningConfig
    seed: int
    train_fraction: float | None = None  # stratified share of train (E6 learning curve); None = all

@dataclass( frozen = True )
class BaselineConfig:
    experiment: str
    family: str
    strategy: str
    seed: int

@dataclass( frozen = True )
class TransformerConfig:
    experiment: str
    family: str
    model_name: str
    max_length: int
    lr: float
    epochs: int
    batch_size: int
    weight_decay: float
    warmup_ratio: float
    early_stopping_patience: int
    seeds: list[ int ]
    train_fraction: float | None = None  # stratified share of train (E6 learning curve); None = all

@dataclass( frozen = True )
class ZeroShotConfig:
    experiment: str
    family: str
    model_name: str
    hypothesis_template: str
    batch_size: int
    seed: int

@dataclass( frozen = True )
class LLMConfig:
    """Zero-shot instruct LLM (E5). Decoding is always greedy (temperature 0), so there is no temperature key."""
    experiment: str
    family: str
    model_name: str
    batch_size: int
    max_new_tokens: int
    seed: int

Config = ClassicalConfig | BaselineConfig | TransformerConfig | ZeroShotConfig | LLMConfig

FAMILIES = {
    "classical": ClassicalConfig,
    "baseline": BaselineConfig,
    "transformer": TransformerConfig,
    "zeroshot": ZeroShotConfig,
    "llm": LLMConfig,
}

# Nested dataclasses, by field name.
NESTED = {
    "vectorizer": VectorizerConfig,
    "classifier": ClassifierConfig,
    "tuning": TuningConfig,
}

def _matches( value: Any, expected: Any ) -> bool:
    origin = typing.get_origin( expected )
    if expected is Any:
        return True
    if origin in ( typing.Union, types.UnionType ):
        return any( _matches( value, option ) for option in typing.get_args( expected ) )
    if expected is type( None ):
        return value is None
    if isinstance( value, bool ):
        return False
    if expected is float:
        return isinstance( value, ( int, float ) )
    if origin is list:
        ( item_type, ) = typing.get_args( expected )
        return isinstance( value, list ) and all( _matches( item, item_type ) for item in value )
    if origin is tuple:
        return isinstance( value, ( list, tuple ) ) and len( value ) == len( typing.get_args( expected ) ) \
            and all( _matches( item, item_type ) for item, item_type in zip( value, typing.get_args( expected ) ) )
    return isinstance( value, expected )

def _build( cls: type, data: Any, where: str ):
    if not isinstance( data, dict ):
        raise ConfigError( f"{ where }: expected a mapping, got { type( data ).__name__ }" )

    known = { field.name: field for field in fields( cls ) }
    unknown = sorted( set( data ) - set( known ) )
    if unknown:
        raise ConfigError( f"{ where }: unknown key(s) { unknown }; allowed: { sorted( known ) }" )

    hints = typing.get_type_hints( cls )
    kwargs = {}
    for name, field in known.items():
        has_default = field.default is not MISSING
        if name not in data:
            if not has_default:
                raise ConfigError( f"{ where }: missing required key '{ name }'" )
            continue

        value = data[ name ]
        if name in NESTED:
            value = _build( NESTED[ name ], value, f"{ where }.{ name }" )
        elif not _matches( value, hints[ name ] ):
            raise ConfigError( f"{ where }: '{ name }' has wrong type ({ repr( value ) }), expected { hints[ name ] }" )
        kwargs[ name ] = value

    if "ngram_range" in kwargs:
        low, high = kwargs[ "ngram_range" ]
        if not 1 <= low <= high:
            raise ConfigError( f"{ where }: ngram_range must satisfy 1 <= low <= high, got { [ low, high ] }" )
        kwargs[ "ngram_range" ] = ( low, high )

    fraction = kwargs.get( "train_fraction" )
    if fraction is not None and not 0 < fraction <= 1:
        raise ConfigError( f"{ where }: train_fraction must be in (0, 1], got { fraction }" )
    for name in ( "batch_size", "max_new_tokens" ):
        if name in kwargs and kwargs[ name ] < 1:
            raise ConfigError( f"{ where }: { name } must be >= 1, got { kwargs[ name ] }" )
    if "seeds" in kwargs and not kwargs[ "seeds" ]:
        raise ConfigError( f"{ where }: seeds must not be empty" )

    _check_choice( cls, kwargs, where )
    return cls( **kwargs )

def _check_choice( cls: type, kwargs: dict, where: str ) -> None:
    choices = {
        VectorizerConfig: ( "kind", VECTORIZER_KINDS ),
        ClassifierConfig: ( "kind", CLASSIFIER_KINDS ),
        BaselineConfig: ( "strategy", BASELINE_STRATEGIES ),
    }
    if cls in choices:
        name, allowed = choices[ cls ]
        if kwargs[ name ] not in allowed:
            raise ConfigError( f"{ where }: '{ name }' must be one of { list( allowed ) }, got { repr( kwargs[ name ] ) }" )

def load_config( path: Path ) -> Config:
    path = Path( path )
    with path.open( encoding = "utf-8" ) as handle:
        data = yaml.safe_load( handle )

    if not isinstance( data, dict ):
        raise ConfigError( f"{ path }: expected a mapping at the top level" )
    family = data.get( "family" )
    if family not in FAMILIES:
        raise ConfigError( f"{ path }: 'family' must be one of { list( FAMILIES ) }, got { repr( family ) }" )

    return _build( FAMILIES[ family ], data, str( path ) )
