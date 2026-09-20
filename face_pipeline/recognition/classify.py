"""Step 3 helper: classify a single face embedding into a person name, or
'unknown' when the confidence is below the given threshold.

The threshold is a run-time argument rather than something baked into the
trained classifier, so it can be retuned from config.yaml between
`run-inference` runs without retraining."""
from __future__ import annotations

import numpy as np

from face_pipeline.recognition.train_classifier import PersonClassifier

UNKNOWN = "unknown"


def classify(classifier: PersonClassifier, embedding: np.ndarray, min_confidence: float) -> tuple[str, float]:
    name, confidence = classifier.predict(embedding)
    if confidence < min_confidence:
        return UNKNOWN, confidence
    return name, confidence
