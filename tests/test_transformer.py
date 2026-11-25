import pytest
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

from emotion_classifier.config import TransformerConfig
from emotion_classifier.data import LABELS
from emotion_classifier.results import load_results
from emotion_classifier.transformer import (
    FineTuneOutput, build_model, count_params, label_maps, run_transformer_experiment, tokenize,
)

@pytest.fixture
def tokenizer( tmp_path ) -> BertTokenizerFast:
    """Offline WordPiece tokenizer over a toy vocab (no network)."""
    words = [ "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "feel", "happy", "sad", "today" ]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text( "\n".join( words ), encoding = "utf-8" )
    return BertTokenizerFast( vocab_file = str( vocab_file ) )

@pytest.fixture
def tiny_model_dir( tmp_path ) -> str:
    """A random 2-layer DistilBERT saved locally, so loading needs no network."""
    config = DistilBertConfig( vocab_size = 10, dim = 16, hidden_dim = 32, n_layers = 2, n_heads = 2, max_position_embeddings = 64 )
    path = tmp_path / "tiny-distilbert"
    DistilBertForSequenceClassification( config ).save_pretrained( path )
    return str( path )

def make_cfg( model_name: str, **overrides ) -> TransformerConfig:
    values = dict(
        experiment = "E3", family = "transformer", model_name = model_name, max_length = 16,
        lr = 5.0e-5, epochs = 1, batch_size = 8, weight_decay = 0.01, warmup_ratio = 0.1,
        early_stopping_patience = 2, seeds = [ 1 ],
    )
    values.update( overrides )
    return TransformerConfig( **values )

def test_tokenize_truncates_to_max_length( tokenizer ):
    long_text = " ".join( [ "happy" ] * 50 )
    encoded = tokenize( tokenizer, [ long_text ], max_length = 8 )
    assert len( encoded[ "input_ids" ][ 0 ] ) == 8

def test_tokenize_does_not_pad( tokenizer ):
    encoded = tokenize( tokenizer, [ "i feel happy", "i feel happy today" ], max_length = 16 )
    assert [ len( ids ) for ids in encoded[ "input_ids" ] ] == [ 5, 6 ]  # [CLS] + words + [SEP]; padding is dynamic

def test_label_maps_match_labels():
    id2label, label2id = label_maps()
    assert [ id2label[ i ] for i in range( len( LABELS ) ) ] == LABELS
    assert all( id2label[ label2id[ name ] ] == name for name in LABELS )

def test_build_model_sets_six_way_head_and_label_maps( tiny_model_dir ):
    model = build_model( tiny_model_dir )
    id2label, label2id = label_maps()
    assert model.config.num_labels == 6
    assert model.config.id2label == id2label
    assert model.config.label2id == label2id

def test_count_params_sums_all_parameters( tiny_model_dir ):
    model = build_model( tiny_model_dir )
    assert count_params( model ) == sum( p.numel() for p in model.parameters() )
    assert count_params( model ) > 0

def fake_fine_tuner( calls: list ):
    def fine_tuner( splits, cfg, seed, output_dir ) -> FineTuneOutput:
        calls.append( ( seed, output_dir ) )
        return FineTuneOutput(
            model = None, tokenizer = None,
            val_pred = list( splits.val_labels ),  # perfect on val
            test_pred = [ 0 ] * len( splits.test_labels ),
            train_seconds = 12.5, val_infer_seconds_per_1k = 1.5, test_infer_seconds_per_1k = 2.5,
            n_params = 1234, device = "cpu",
        )
    return fine_tuner

def run_fake( tiny_splits, tmp_path, cfg = None, **kwargs ):
    calls: list = []
    results = run_transformer_experiment(
        cfg or make_cfg( "distilbert-base-uncased", seeds = [ 13, 42, 7 ] ),
        tiny_splits, "distilbert", tmp_path / "results", tmp_path / "outputs", "20261124T090000",
        git_sha = "abc123", fine_tuner = fake_fine_tuner( calls ), **kwargs,
    )
    return results, calls

def test_experiment_saves_one_result_per_seed_per_split( tiny_splits, tmp_path ):
    results, calls = run_fake( tiny_splits, tmp_path )
    assert [ seed for seed, _ in calls ] == [ 13, 42, 7 ]
    assert len( results ) == 6
    assert sorted( ( r.seed, r.split ) for r in results ) == sorted( ( s, p ) for s in ( 13, 42, 7 ) for p in ( "val", "test" ) )
    assert len( load_results( tmp_path / "results" ) ) == 6
    assert len( list( ( tmp_path / "results" / "preds" ).glob( "*.csv" ) ) ) == 6

def test_experiment_records_metrics_cost_and_provenance( tiny_splits, tmp_path ):
    results, _ = run_fake( tiny_splits, tmp_path )
    by_split = { r.split: r for r in results if r.seed == 13 }
    assert by_split[ "val" ].metrics[ "macro_f1" ] == 1.0
    assert by_split[ "test" ].metrics[ "macro_f1" ] < 1.0
    assert by_split[ "val" ].infer_seconds_per_1k == 1.5
    assert by_split[ "test" ].infer_seconds_per_1k == 2.5
    for result in by_split.values():
        assert result.experiment == "E3"
        assert result.model == "distilbert"
        assert result.train_seconds == 12.5
        assert result.n_params == 1234
        assert result.device == "cpu"
        assert result.git_sha == "abc123"
        assert result.config[ "model_name" ] == "distilbert-base-uncased"
        assert result.run_id.startswith( "E3-distilbert-" )

def test_experiment_gives_each_seed_its_own_output_dir( tiny_splits, tmp_path ):
    _, calls = run_fake( tiny_splits, tmp_path )
    dirs = [ directory for _, directory in calls ]
    assert len( set( dirs ) ) == 3
    assert all( ( tmp_path / "outputs" ) in directory.parents for directory in dirs )

def test_experiment_single_split( tiny_splits, tmp_path ):
    results, _ = run_fake( tiny_splits, tmp_path, splits_to_save = "val" )
    assert { r.split for r in results } == { "val" }
    assert len( results ) == 3

def test_experiment_rejects_unknown_split( tiny_splits, tmp_path ):
    with pytest.raises( ValueError ):
        run_fake( tiny_splits, tmp_path, splits_to_save = "train" )

@pytest.mark.slow
def test_fine_tune_smoke_gives_six_way_predictions( tiny_splits, tmp_path ):
    from emotion_classifier.transformer import fine_tune

    cfg = make_cfg( "hf-internal-testing/tiny-random-distilbert", batch_size = 32 )
    output = fine_tune( tiny_splits, cfg, seed = 1, output_dir = tmp_path )

    assert len( output.val_pred ) == len( tiny_splits.val_labels )
    assert len( output.test_pred ) == len( tiny_splits.test_labels )
    assert set( output.val_pred + output.test_pred ) <= set( range( 6 ) )
    assert output.n_params > 0
    assert output.train_seconds > 0
    assert output.val_infer_seconds_per_1k > 0
    assert output.device in ( "cuda", "mps", "cpu" )
