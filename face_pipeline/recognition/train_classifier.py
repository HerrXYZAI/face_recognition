"""Step 2: train a person classifier on top of InsightFace embeddings, using
the faces exported from Lightroom (face_pipeline.lightroom.catalog_export) as
labeled training data.

For each labeled image we run full-image face detection once (an image can
have several labeled faces) and match each labeled bounding box to the
closest detected box by IoU, rather than cropping-and-re-detecting -- a crop
tight to Lightroom's box can cut off context the detector needs.
"""
from __future__ import annotations

import csv
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from face_pipeline.imaging import fractional_to_pixel_bbox, iou, load_image_bgr
from face_pipeline.recognition.embeddings import FaceEmbedder

logger = logging.getLogger(__name__)

MIN_IOU_MATCH = 0.3
MIN_EXAMPLES_PER_PERSON = 3


@dataclass
class TrainingExample:
    person_name: str
    embedding: np.ndarray


def _load_labels_by_image(csv_path: Path) -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_image[row["image_path"]].append(row)
    return by_image


def collect_training_examples(
    labels_csv: Path, embedder: FaceEmbedder, skip_missing: bool = True
) -> list[TrainingExample]:
    by_image = _load_labels_by_image(labels_csv)
    examples: list[TrainingExample] = []
    unmatched = 0

    for image_path, rows in by_image.items():
        path = Path(image_path)
        if not path.exists():
            if skip_missing:
                logger.warning("Skipping missing image: %s", image_path)
                continue
            raise FileNotFoundError(image_path)

        try:
            image_bgr = load_image_bgr(path)
        except Exception:
            logger.exception("Failed to load image, skipping: %s", image_path)
            continue

        height, width = image_bgr.shape[:2]
        detections = embedder.detect(image_bgr)

        for row in rows:
            target = fractional_to_pixel_bbox(
                float(row["left"]), float(row["top"]), float(row["right"]), float(row["bottom"]),
                width, height,
            )
            best, best_iou = None, 0.0
            for det in detections:
                score = iou(target, (det.left, det.top, det.right, det.bottom))
                if score > best_iou:
                    best, best_iou = det, score
            if best is None or best_iou < MIN_IOU_MATCH:
                unmatched += 1
                logger.debug(
                    "No detector match for %s / %s (best IoU=%.2f)",
                    image_path, row["person_name"], best_iou,
                )
                continue
            examples.append(TrainingExample(person_name=row["person_name"], embedding=best.embedding))

    if unmatched:
        logger.warning(
            "%d labeled faces had no matching detection (IoU < %.2f) and were "
            "skipped. If this is a large fraction of your labels, run "
            "`face-pipeline verify-crops` to sanity-check bbox alignment.",
            unmatched, MIN_IOU_MATCH,
        )
    logger.info("Collected %d training examples across %d people",
                len(examples), len({e.person_name for e in examples}))
    return examples


@dataclass
class PersonClassifier:
    svm: SVC
    labels: list[str]
    centroids: dict[str, np.ndarray] = field(default_factory=dict)

    def predict(self, embedding: np.ndarray) -> tuple[str, float]:
        proba = self.svm.predict_proba(embedding.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        return self.labels[idx], float(proba[idx])


def train(examples: list[TrainingExample]) -> PersonClassifier:
    if not examples:
        raise ValueError("No training examples collected -- nothing to train on.")

    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.person_name] += 1
    too_few = {name: n for name, n in counts.items() if n < MIN_EXAMPLES_PER_PERSON}
    if too_few:
        logger.warning(
            "These people have fewer than %d labeled examples and may classify "
            "poorly: %s", MIN_EXAMPLES_PER_PERSON, too_few,
        )

    x = np.stack([ex.embedding for ex in examples])
    y = [ex.person_name for ex in examples]

    svm = SVC(kernel="linear", probability=True, class_weight="balanced")
    svm.fit(x, y)

    centroids: dict[str, np.ndarray] = {}
    for name in set(y):
        vecs = np.stack([ex.embedding for ex in examples if ex.person_name == name])
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[name] = centroid / norm if norm > 0 else centroid

    return PersonClassifier(svm=svm, labels=list(svm.classes_), centroids=centroids)


def save(classifier: PersonClassifier, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(classifier, f)
    logger.info("Saved classifier (%d people) to %s", len(classifier.labels), out_path)


def load(path: Path) -> PersonClassifier:
    with path.open("rb") as f:
        return pickle.load(f)
