from dataclasses import dataclass

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
    