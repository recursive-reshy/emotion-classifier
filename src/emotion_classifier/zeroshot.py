from typing import Any, Callable

from emotion_classifier.config import ZeroShotConfig
from emotion_classifier.data import LABELS
from emotion_classifier.timing import Timer

def labels_to_ids( names: list[ str ] ) -> list[ int ]:
    """Label names (pipeline output) -> ids (index in data.LABELS)."""
    ids = []
    for name in names:
        if name not in LABELS:
            raise ValueError( f"Unknown label { repr( name ) }; expected one of { LABELS }" )
        ids.append( LABELS.index( name ) )
    return ids

def top_label_ids( outputs: dict | list[ dict ] ) -> list[ int ]:
    """Highest-scoring label id per pipeline result (picked by score, not by list position)."""
    if isinstance( outputs, dict ):
        outputs = [ outputs ]
    top = [ max( zip( output[ "scores" ], output[ "labels" ] ) )[ 1 ] for output in outputs ]
    return labels_to_ids( top )

def build_pipeline( cfg: ZeroShotConfig, device: str ) -> Callable[ ..., Any ]:
    """HF zero-shot NLI pipeline (imported lazily: transformers is slow to import)."""
    from transformers import pipeline

    return pipeline( "zero-shot-classification", model = cfg.model_name, device = device )

def predict_zeroshot(
    texts: list[ str ],
    cfg: ZeroShotConfig,
    classifier: Callable[ ..., Any ] | None = None,
    device: str = "cpu",
) -> tuple[ list[ int ], float ]:
    """Single-label zero-shot predictions, and inference seconds per 1,000 examples.
    Labels are the six emotion names and the hypothesis template is fixed by the config."""
    if not texts:
        return [], 0.0

    classifier = classifier or build_pipeline( cfg, device )
    with Timer() as timer:
        outputs = classifier(
            list( texts ),
            candidate_labels = LABELS,
            hypothesis_template = cfg.hypothesis_template,
            batch_size = cfg.batch_size,
            multi_label = False,
        )
    return top_label_ids( outputs ), timer.seconds / len( texts ) * 1000
