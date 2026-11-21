import numpy as np
import pytest

from emotion_classifier.features import build_vectorizer
from emotion_classifier.preprocess import remove_stop_words, resolve_stop_words

DOCS = [ "the cat sat on the mat", "i am not happy", "happy happy joy" ]

def _vocab( vectorizer ) -> set[ str ]:
    return set( vectorizer.get_feature_names_out() )

def test_bow_gives_counts():
    vectorizer = build_vectorizer( "bow", ( 1, 1 ), 1, None, None )
    matrix = vectorizer.fit_transform( [ "happy happy joy" ] ).toarray()[ 0 ]
    counts = dict( zip( vectorizer.get_feature_names_out(), matrix ) )
    assert counts == { "happy": 2, "joy": 1 }

def test_tfidf_rows_are_l2_normalised():
    matrix = build_vectorizer( "tfidf", ( 1, 2 ), 1, None, None ).fit_transform( DOCS )
    norms = np.sqrt( matrix.multiply( matrix ).sum( axis = 1 ) ).A1
    assert norms == pytest.approx( [ 1.0 ] * len( DOCS ) )

def test_unigram_range_has_no_bigrams():
    vectorizer = build_vectorizer( "bow", ( 1, 1 ), 1, None, None ).fit( DOCS )
    assert not any( " " in term for term in _vocab( vectorizer ) )

def test_bigrams_appear_when_ngram_range_is_1_2():
    vectorizer = build_vectorizer( "bow", ( 1, 2 ), 1, None, None ).fit( DOCS )
    assert { "cat sat", "not happy" } <= _vocab( vectorizer )

def test_min_df_drops_rare_terms():
    vocab = _vocab( build_vectorizer( "bow", ( 1, 1 ), 2, None, None ).fit( DOCS ) )
    assert vocab == { "happy" }

def test_max_features_caps_vocabulary():
    vectorizer = build_vectorizer( "bow", ( 1, 1 ), 1, 3, None ).fit( DOCS )
    assert len( _vocab( vectorizer ) ) == 3

def test_stop_word_removal_drops_the():
    vocab = _vocab( build_vectorizer( "bow", ( 1, 1 ), 1, None, "english" ).fit( DOCS ) )
    assert "the" not in vocab
    assert "cat" in vocab

def test_stop_word_removal_also_drops_not():
    # Trade-off the report discusses: the stock English list removes negation,
    # so "not happy" and "happy" become indistinguishable.
    vocab = _vocab( build_vectorizer( "bow", ( 1, 1 ), 1, None, "english" ).fit( DOCS ) )
    assert "not" not in vocab
    assert "happy" in vocab

def test_unknown_kind_raises():
    with pytest.raises( ValueError ):
        build_vectorizer( "word2vec", ( 1, 1 ), 1, None, None )

def test_resolve_stop_words():
    assert resolve_stop_words( None ) is None
    words = resolve_stop_words( "english" )
    assert "the" in words and "not" in words
    with pytest.raises( ValueError ):
        resolve_stop_words( "klingon" )

def test_remove_stop_words_is_a_pure_text_function():
    stop = resolve_stop_words( "english" )
    assert remove_stop_words( "the cat is not happy", stop ) == "cat happy"
    assert remove_stop_words( "the cat", None ) == "the cat"
