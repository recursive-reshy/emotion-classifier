from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORD_LISTS = { "english": ENGLISH_STOP_WORDS }

def resolve_stop_words( name: str | None ) -> list[ str ] | None:
    """Config value -> word list. Note the stock English list contains negations such as "not" and "no"."""
    if name is None:
        return None
    if name not in STOP_WORD_LISTS:
        raise ValueError( f"Unknown stop-word list { repr( name ) }; expected one of { list( STOP_WORD_LISTS ) }" )
    return sorted( STOP_WORD_LISTS[ name ] )

def remove_stop_words( text: str, stop_words: list[ str ] | None ) -> str:
    if stop_words is None:
        return text
    stop = set( stop_words )
    return " ".join( word for word in text.split() if word not in stop )
