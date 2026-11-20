from dataclasses import dataclass
from collections import Counter

from sklearn.model_selection import train_test_split

DATASET_NAME = "dair-ai/emotion"
DATASET_CONFIG = "split"

LABELS: list[ str ] = [ "saddness", "joy", "love", "anger", "fear", "surprise" ]

@dataclass( frozen = True )
class DatasetSplits:
    train_texts: list[ str ]
    train_labels: list[ int ]
    val_texts: list[ str ]
    val_labels: list[ int ]
    test_texts: list[ str ]
    test_labels: list[ int ]

    def pairs( self ) -> list[ tuple[ list[ str ], list[ int ] ] ]:
        return [
            ( self.train_texts, self.train_labels ),
            ( self.val_texts, self.val_labels ),
            ( self.test_texts, self.test_labels )
        ]

def class_distribution( labels: list[ int ] ) -> dict[ str, int ]:
    counts = Counter( labels )
    return { name: counts.get( index, 0 ) for index, name in enumerate( LABELS ) }

def stratified_subsample(
    texts: list[ str ],
    labels: list[ int ],
    fraction: float,
    seed: int
) -> tuple[ list[ str ], list[ int ] ]:
    if not 0 < fraction <= 1:
        raise ValueError( f"Fraction must be in (0, 1], got {fraction}" )

    if fraction == 1:
        return texts, labels

    sub_texts, _, sub_labels, _ = train_test_split(
        texts,
        labels,
        train_size = fraction,
        stratify = labels,
        random_state = seed
    )

    return list( sub_texts ), list( sub_labels )

def load_emotion(
    subsample: float | None = None,
    seed: int = 42
) -> DatasetSplits:
    from datasets import load_dataset

    dataset = load_dataset( DATASET_NAME, DATASET_CONFIG )
    names = dataset[ "train" ].features[ "label" ].names

    if names != LABELS:
        raise ValueError( f"Label order mismatch: dataset has {names}, code expects {LABELS}" )
    
    train_texts, train_labels = list( dataset[ "train" ][ "text" ] ), list( dataset[ "train" ][ "label" ] )
    if subsample is not None:
        train_texts, train_labels = stratified_subsample( train_texts, train_labels, subsample, seed )

    return DatasetSplits(
        train_texts = train_texts, train_labels = train_labels,
        val_texts = list( dataset[ "validation" ][ "text" ] ), val_labels = list( dataset[ "validation" ][ "label" ] ),
        test_texts = list( dataset[ "test" ][ "text" ] ), test_labels = list( dataset[ "test" ][ "label" ] ),
    )