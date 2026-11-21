from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from emotion_classifier.data import LABELS

def compute_metrics( y_true: list[ int ], y_prep: list[ int ], labels: list[ str ] = LABELS ) -> dict:
    label_ids = list( range( len( labels ) ) )

    precison, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_prep,
        labels = label_ids,
        zero_division = 0
    )

    per_class = {
        name: {
            "precision": float( precison[ index ] ),
            "recall": float( recall[ index ] ),
            "f1": float( f1[ index ] ),
            "support": int( support[ index ] )
        }
        for index, name in enumerate( labels )
    }

    return {
        "accuracy": float( accuracy_score( y_true, y_prep ) ),
        "macro_f1": float( f1_score( y_true, y_prep, labels = label_ids, average = "macro", zero_division = 0 ) ),
        "weighted_f1": float( f1_score( y_true, y_prep, labels = label_ids, average = "weighted", zero_division = 0 ) ),
        "per_class": per_class,
        "confusion": confusion_matrix( y_true, y_prep, labels = label_ids ).tolist()
    }

def row_normalise( confusion: list[ list[ int ] ] ) -> list[ list[ float ] ]:
    normalised = []
    for row in confusion:
        total = sum( row )
        normalised.append( [ value / total if total else 0.0 for value in row ] )
    return normalised