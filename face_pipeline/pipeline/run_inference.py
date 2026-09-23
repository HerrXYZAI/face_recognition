"""Step 3: batch face detection + classification over the full image library,
storing every detected face (bounding box, embedding, predicted person) into
faces.db. Resumable -- images already processed at their current mtime are
skipped, so a re-run after adding new photos, or after being stopped
partway through, only processes what's left. Pass restart=True to ignore
that and reprocess the whole library from scratch instead.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from face_pipeline.imaging import load_image_bgr
from face_pipeline.pipeline import db
from face_pipeline.recognition.classify import classify
from face_pipeline.recognition.embeddings import FaceEmbedder
from face_pipeline.recognition.train_classifier import PersonClassifier

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng",
    ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".pef", ".heic",
}
MODEL_VERSION = "buffalo_l+svm-v1"


def iter_images(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def run(
    images_root: Path,
    faces_db_path: Path,
    embedder: FaceEmbedder,
    classifier: PersonClassifier,
    min_confidence: float,
    limit: Optional[int] = None,
    restart: bool = False,
) -> dict:
    """restart=False (default) skips any image already processed at its
    current mtime -- the mechanism that makes this resumable: each image is
    committed to faces.db (image row + all its faces) as soon as it's done,
    so stopping the run anytime loses at most the one image in flight, and
    re-running just picks up with whatever isn't committed yet. restart=True
    ignores that check and reprocesses every image regardless of mtime,
    which -- like any mtime change today -- replaces its existing faces and
    resets their review decisions back to pending (see db.upsert_image)."""
    stats = {"images_processed": 0, "images_skipped": 0, "faces_found": 0, "errors": 0}
    paths = list(iter_images(images_root))
    if limit:
        paths = paths[:limit]
    logger.info("Found %d candidate images under %s", len(paths), images_root)

    with db.connect(faces_db_path) as conn:
        for i, path in enumerate(tqdm(paths, desc="Processing images")):
            mtime = path.stat().st_mtime
            if not restart:
                existing_mtime = db.get_processed_mtime(conn, str(path))
                if existing_mtime is not None and existing_mtime == mtime:
                    stats["images_skipped"] += 1
                    continue

            try:
                image_bgr = load_image_bgr(path)
            except Exception:
                logger.exception("Failed to load image: %s", path)
                stats["errors"] += 1
                continue

            height, width = image_bgr.shape[:2]
            try:
                detections = embedder.detect(image_bgr)
            except Exception:
                logger.exception("Detection failed for: %s", path)
                stats["errors"] += 1
                continue

            image_id = db.upsert_image(
                conn, str(path), width, height, mtime,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            for det in detections:
                name, confidence = classify(classifier, det.embedding, min_confidence)
                db.insert_face(
                    conn, image_id,
                    det.left, det.top, det.right, det.bottom,
                    det.embedding, name, confidence, det.detector_score, MODEL_VERSION,
                )
            stats["images_processed"] += 1
            stats["faces_found"] += len(detections)

            # Committed per image (not batched) so an interrupted run can
            # resume from exactly here instead of redoing a whole batch.
            conn.commit()
            if (i + 1) % 500 == 0:
                logger.info("Progress: %s", stats)

    logger.info("Done: %s", stats)
    return stats
