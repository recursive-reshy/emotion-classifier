from emotion_classifier.data import LABELS, DatasetSplits

def test_labels_have_dataset_emotions_in_order():
    assert LABELS == [ "sadness", "joy", "love", "anger", "fear", "surprise" ]

def test_fixture_is_a_dataset_splits( tiny_splits ):
    assert isinstance( tiny_splits, DatasetSplits )

def test_texts_and_labels_align_in_every_split( tiny_splits ):
    for texts, labels in tiny_splits.pairs():
        assert len( texts ) == len( labels )

def test_every_split_contains_all_six_labels( tiny_splits ):
    for _, labels in tiny_splits.pairs():
        assert set( labels ) == set( range( len( LABELS ) ) )