from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from emotion_classifier.config import VECTORIZER_KINDS
from emotion_classifier.preprocess import resolve_stop_words

def build_vectorizer(
    kind: str,
    ngram_range: tuple[ int, int ],
    min_df: int,
    max_features: int | None,
    stop_words: str | None,
) -> CountVectorizer:
    if kind not in VECTORIZER_KINDS:
        raise ValueError( f"Unknown vectorizer kind { repr( kind ) }; expected one of { list( VECTORIZER_KINDS ) }" )

    vectorizer_class = CountVectorizer if kind == "bow" else TfidfVectorizer
    return vectorizer_class(
        ngram_range = tuple( ngram_range ),
        min_df = min_df,
        max_features = max_features,
        stop_words = resolve_stop_words( stop_words ),
    )
