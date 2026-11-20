import pytest

from emotion_classifier.data import LABELS, class_distribution, load_emotion, stratified_subsample

# class_distribution
def test_class_distribution_counts_every_label_by_name( tiny_splits ):
    counts = class_distribution( [ 1, 1, 3 ] )
    
    assert list( counts.keys() ) == LABELS
    assert counts[ "joy" ] == 2
    assert counts[ "anger" ] == 1
    assert counts[ "surprise" ] == 0

# stratified_subsample
def _imbalanced() -> tuple[ list[ str ], list[ int ] ]:
    labels = [ 0 ] * 80 + [ 5 ] * 20
    texts = [ f"text {i}" for i in range( len( labels ) ) ]
    return texts, labels

def test_subsample_keeps_class_proportions():
    texts, labels = _imbalanced()
    _, sub_labels = stratified_subsample( texts, labels, fraction = 0.5, seed = 42 )

    assert len( sub_labels ) == 50
    assert sub_labels.count( 0 ) == 40
    assert sub_labels.count( 5 ) == 10

def test_subsample_keeps_texts_paired_with_their_labels():
    texts, labels = _imbalanced()
    sub_texts, sub_labels = stratified_subsample( texts, labels, fraction = 0.5, seed = 42 )
    original = dict( zip( texts, labels ) )

    assert all( original[ text ] == label for text, label in zip( sub_texts, sub_labels ) )

def test_subsample_is_deterministic_for_a_seed():
    texts, labels = _imbalanced()
    first = stratified_subsample( texts, labels, fraction = 0.25, seed = 7 )
    second = stratified_subsample( texts, labels, fraction = 0.25, seed = 7 )

    assert first == second

def test_subsample_fraction_one_returns_everything():
    texts, labels = _imbalanced()

    assert stratified_subsample( texts, labels, fraction = 1.0, seed = 42 ) == ( texts, labels )

@pytest.mark.parametrize( "fraction", [ 0, -0.1, 1.5 ] )
def test_subsample_rejects_invalid_fractions( fraction ):
    texts, labels = _imbalanced()

    with pytest.raises( ValueError ):
        stratified_subsample( texts, labels, fraction = fraction, seed = 42 )

# real data (network)
@pytest.mark.network
def test_real_dataset_has_expected_splits_and_labels():
    splits = load_emotion()

    assert len( splits.train_texts ) == 16_000
    assert len( splits.val_texts ) == 2_000
    assert len( splits.test_texts ) == 2_000
    assert set( splits.train_labels ) == set( range( 6 ) )