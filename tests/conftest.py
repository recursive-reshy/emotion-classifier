import pytest

from emotion_classifier.data import DatasetSplits

EXAMPLES: dict[ int, list[ str ] ] = {
    0: [  # sadness
        "i feel so alone and miserable tonight",
        "i feel heartbroken and empty inside",
        "i feel gloomy and hopeless about everything",
        "i feel lonely since everyone left",
        "i feel so sad and down today",
        "i feel defeated and miserable after the news",
        "i feel empty and sad without you",
        "i feel hopeless and lonely again",
        "i feel so unhappy and gloomy this week",
        "i feel heartbroken after saying goodbye",
    ],
    1: [  # joy
        "i feel so happy and cheerful today",
        "i feel delighted with how the day went",
        "i feel joyful and excited about the weekend",
        "i feel glad that everything worked out",
        "i feel cheerful and content this morning",
        "i feel happy to be home again",
        "i feel delighted and pleased with my results",
        "i feel joyful after a great workout",
        "i feel content and glad right now",
        "i feel so pleased and happy with life",
    ],
    2: [  # love
        "i feel so loving towards my partner",
        "i feel affectionate and tender with my family",
        "i feel devoted to the people i adore",
        "i feel romantic and caring tonight",
        "i feel passionate about you my darling",
        "i feel tender and loving when we cuddle",
        "i feel adored and cherished by my wife",
        "i feel caring and affectionate towards him",
        "i feel so much love for my sweetheart",
        "i feel devoted and romantic on our anniversary",
    ],
    3: [  # anger
        "i feel so angry and furious right now",
        "i feel irritated by their rude comments",
        "i feel annoyed and mad at the delay",
        "i feel furious that they lied again",
        "i feel resentful and bitter about the decision",
        "i feel enraged by the unfair treatment",
        "i feel so annoyed and irritated today",
        "i feel mad and angry at my landlord",
        "i feel bitter and furious about the bill",
        "i feel outraged and resentful at work",
    ],
    4: [  # fear
        "i feel scared and anxious about tomorrow",
        "i feel terrified of being alone in the dark",
        "i feel nervous and afraid before the exam",
        "i feel frightened by the strange noises",
        "i feel anxious and worried about my health",
        "i feel so scared of what might happen",
        "i feel afraid and uneasy walking home",
        "i feel nervous and terrified of the interview",
        "i feel worried and frightened all night",
        "i feel uneasy and anxious in crowds",
    ],
    5: [  # surprise
        "i feel so surprised by the sudden news",
        "i feel amazed at how fast it happened",
        "i feel shocked and stunned by the result",
        "i feel astonished at the unexpected gift",
        "i feel surprised and amazed by the ending",
        "i feel stunned that they actually came",
        "i feel shocked by the strange turn of events",
        "i feel amazed and astonished by the view",
        "i feel so surprised at the curious twist",
        "i feel stunned and shocked by the announcement",
    ],
}

def _split( start: int, stop: int ) -> tuple[ list[ str ], list[ int ] ]:
    texts: list[ str ] = []
    labels: list[ int ] = []

    for label, sentences in EXAMPLES.items():
        texts.extend( sentences[ start:stop ] )
        labels.extend( [ label ] * ( stop - start ) )
    return texts, labels

@pytest.fixture
def tiny_splits() -> DatasetSplits:
    train_texts, train_labels = _split( 0, 6 )
    val_texts, val_labels = _split( 6, 8 )
    test_texts, test_labels = _split( 8, 10 )

    return DatasetSplits(
        train_texts = train_texts, train_labels = train_labels,
        val_texts = val_texts, val_labels = val_labels,
        test_texts = test_texts, test_labels = test_labels
    )