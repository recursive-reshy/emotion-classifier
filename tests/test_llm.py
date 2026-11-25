import csv

import pytest

from emotion_classifier.config import LLMConfig
from emotion_classifier.data import LABELS
from emotion_classifier.results import load_results
from emotion_classifier.zeroshot import INVALID, build_prompt, parse_label, predict_llm, run_llm_experiment

def make_cfg( **overrides ) -> LLMConfig:
    values = dict( experiment = "E5", family = "llm", model_name = "Qwen/Qwen3-4B-Instruct-2507", batch_size = 2, max_new_tokens = 8, seed = 42 )
    values.update( overrides )
    return LLMConfig( **values )

# ---- build_prompt

def test_build_prompt_contains_the_tweet_and_every_label_in_order():
    prompt = build_prompt( "i feel so alone tonight", LABELS )
    assert "i feel so alone tonight" in prompt
    positions = [ prompt.index( label ) for label in LABELS ]
    assert positions == sorted( positions )

def test_build_prompt_is_deterministic_and_depends_on_the_text():
    assert build_prompt( "a", LABELS ) == build_prompt( "a", LABELS )
    assert build_prompt( "a", LABELS ) != build_prompt( "b", LABELS )

def test_build_prompt_leaves_braces_in_the_tweet_alone():
    assert "{label} {0} {}" in build_prompt( "{label} {0} {}", LABELS )

def test_build_prompt_asks_for_the_label_only():
    assert "only" in build_prompt( "x", LABELS ).lower()

# ---- parse_label

@pytest.mark.parametrize( "raw, expected", [ ( label, index ) for index, label in enumerate( LABELS ) ] )
def test_parse_label_exact_label( raw, expected ):
    assert parse_label( raw, LABELS ) == expected

@pytest.mark.parametrize( "raw, expected", [
    ( "Joy.", 1 ), ( " ANGER!\n", 3 ), ( '"fear"', 4 ), ( "Sadness\n", 0 ), ( "**love**", 2 ), ( "Surprise", 5 ),
] )
def test_parse_label_ignores_case_punctuation_and_whitespace( raw, expected ):
    assert parse_label( raw, LABELS ) == expected

def test_parse_label_accepts_a_single_label_inside_a_sentence():
    assert parse_label( "The emotion is love.", LABELS ) == 2
    assert parse_label( "joy joy", LABELS ) == 1  # one distinct label

@pytest.mark.parametrize( "raw", [ "happiness", "joyful", "anxious", "surprised", "unjoy", "excited" ] )
def test_parse_label_synonyms_and_word_forms_are_invalid( raw ):
    assert parse_label( raw, LABELS ) is None

@pytest.mark.parametrize( "raw", [ "joy or love", "sadness, anger", "fear and surprise", "Label: joy / love" ] )
def test_parse_label_several_labels_is_invalid( raw ):
    assert parse_label( raw, LABELS ) is None

@pytest.mark.parametrize( "raw", [ "", "   ", "\n", "I cannot determine the emotion." ] )
def test_parse_label_empty_or_label_free_output_is_invalid( raw ):
    assert parse_label( raw, LABELS ) is None

def test_parse_label_uses_the_labels_it_is_given():
    assert parse_label( "b", [ "a", "b" ] ) == 1
    assert parse_label( "joy", [ "a", "b" ] ) is None

# ---- predict_llm

class FakeGenerator:
    def __init__( self, outputs: list[ str ] ):
        self.outputs, self.batches = outputs, []

    def __call__( self, prompts: list[ str ] ) -> list[ str ]:
        start = sum( len( batch ) for batch in self.batches )
        self.batches.append( prompts )
        return self.outputs[ start:start + len( prompts ) ]

def test_predict_llm_batches_prompts_in_order_and_returns_raw_outputs():
    fake = FakeGenerator( [ "joy", "fear", "love", "anger", "sadness" ] )
    raws, per_1k = predict_llm( [ "a", "b", "c", "d", "e" ], make_cfg( batch_size = 2 ), generator = fake )

    assert raws == [ "joy", "fear", "love", "anger", "sadness" ]
    assert [ len( batch ) for batch in fake.batches ] == [ 2, 2, 1 ]
    assert fake.batches[ 0 ] == [ build_prompt( "a", LABELS ), build_prompt( "b", LABELS ) ]
    assert per_1k >= 0.0

def test_predict_llm_rejects_a_generator_that_drops_outputs():
    with pytest.raises( ValueError, match = "outputs" ):
        predict_llm( [ "a", "b" ], make_cfg(), generator = lambda prompts: [ "joy" ] )

def test_predict_llm_handles_empty_input():
    assert predict_llm( [], make_cfg(), generator = FakeGenerator( [] ) ) == ( [], 0.0 )

# ---- run_llm_experiment

def run_fake( tiny_splits, tmp_path, outputs = None, **kwargs ):
    outputs = outputs if outputs is not None else [ LABELS[ label ] for label in tiny_splits.test_labels ]
    return run_llm_experiment(
        make_cfg(), tiny_splits, "qwen3_4b_instruct", tmp_path / "results", "20261125T140000",
        git_sha = "abc123", generator = FakeGenerator( outputs ), device = "cpu", **kwargs,
    )

def test_run_evaluates_on_test_only_and_scores_a_perfect_model( tiny_splits, tmp_path ):
    ( result, ) = run_fake( tiny_splits, tmp_path )
    assert result.split == "test"
    assert result.metrics[ "macro_f1" ] == 1.0
    assert result.metrics[ "invalid_rate" ] == 0.0
    assert [ r.split for r in load_results( tmp_path / "results" ) ] == [ "test" ]

def test_run_records_provenance_and_the_exact_prompt( tiny_splits, tmp_path ):
    ( result, ) = run_fake( tiny_splits, tmp_path )
    assert result.experiment == "E5"
    assert result.model == "qwen3_4b_instruct"
    assert result.run_id.startswith( "E5-qwen3_4b_instruct-test-42-" )
    assert result.train_seconds is None
    assert result.infer_seconds_per_1k >= 0.0
    assert result.n_params is None  # the fake generator has no model to count
    assert result.git_sha == "abc123"
    assert "<tweet>" in result.config[ "prompt_example" ]
    assert "greedy" in result.config[ "decoding" ]

def test_invalid_outputs_count_as_wrong_and_are_reported_separately( tiny_splits, tmp_path ):
    n = len( tiny_splits.test_labels )
    outputs = [ LABELS[ label ] for label in tiny_splits.test_labels ]
    outputs[ : n // 2 ] = [ "no idea" ] * ( n // 2 )
    ( result, ) = run_fake( tiny_splits, tmp_path, outputs = outputs )

    assert result.metrics[ "invalid_rate" ] == pytest.approx( ( n // 2 ) / n )
    assert result.metrics[ "n_invalid" ] == n // 2
    assert result.metrics[ "accuracy" ] == pytest.approx( ( n - n // 2 ) / n )  # invalid = wrong, not dropped
    with ( tmp_path / "results" / result.predictions_path ).open( newline = "" ) as handle:
        preds = [ int( row[ "pred" ] ) for row in csv.DictReader( handle ) ]
    assert preds[ : n // 2 ] == [ INVALID ] * ( n // 2 )

def test_all_invalid_outputs_do_not_crash_the_metrics( tiny_splits, tmp_path ):
    ( result, ) = run_fake( tiny_splits, tmp_path, outputs = [ "???" ] * len( tiny_splits.test_labels ) )
    assert result.metrics[ "invalid_rate" ] == 1.0
    assert result.metrics[ "accuracy" ] == 0.0
    assert result.metrics[ "macro_f1" ] == 0.0

def test_raw_outputs_are_saved_for_auditing_the_parser( tiny_splits, tmp_path ):
    outputs = [ "Joy." ] * len( tiny_splits.test_labels )
    ( result, ) = run_fake( tiny_splits, tmp_path, outputs = outputs )
    raw_path = tmp_path / "results" / "preds" / f"{ result.run_id }.raw.csv"
    with raw_path.open( newline = "" ) as handle:
        rows = list( csv.DictReader( handle ) )
    assert [ row[ "raw" ] for row in rows ] == outputs
    assert [ row[ "text" ] for row in rows ] == tiny_splits.test_texts

def test_run_counts_params_when_the_generator_has_a_model( tiny_splits, tmp_path ):
    import torch

    generator = FakeGenerator( [ LABELS[ label ] for label in tiny_splits.test_labels ] )
    generator.model = torch.nn.Linear( 3, 2 )
    ( result, ) = run_llm_experiment( make_cfg(), tiny_splits, "q", tmp_path / "results", "20261125T140000", generator = generator, device = "cpu" )
    assert result.n_params == 8

def test_run_never_overwrites( tiny_splits, tmp_path ):
    run_fake( tiny_splits, tmp_path )
    with pytest.raises( FileExistsError ):
        run_fake( tiny_splits, tmp_path )

@pytest.mark.slow
def test_hf_generator_smoke_on_a_tiny_random_model():
    from emotion_classifier.zeroshot import build_generator

    cfg = make_cfg( model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM", max_new_tokens = 3, batch_size = 2 )
    generator = build_generator( cfg, "cpu" )
    raws, per_1k = predict_llm( [ "i feel fine", "a much longer tweet about how i feel today" ], cfg, generator = generator )

    assert len( raws ) == 2 and all( isinstance( raw, str ) for raw in raws )
    assert per_1k > 0.0
    assert sum( p.numel() for p in generator.model.parameters() ) > 0
