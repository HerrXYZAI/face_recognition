"""Thin wrapper around insightface.app.FaceAnalysis for face detection +
embedding extraction (ArcFace, via the buffalo_l model pack by default)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    left: float
    top: float
    right: float
    bottom: float
    embedding: np.ndarray  # 512-d, L2-normalized
    detector_score: float


class FaceEmbedder:
    """Lazily loads the InsightFace model on first use, since model init is
    slow and not every CLI command needs it (e.g. `inspect-catalog`)."""

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0, det_size=(640, 640)):
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._det_size = tuple(det_size)
        self._app = None

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        logger.info(
            "Loading InsightFace model '%s' (ctx_id=%d, det_size=%s)",
            self._model_name, self._ctx_id, self._det_size,
        )
        self._app = FaceAnalysis(name=self._model_name)
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """image_bgr: HxWx3 uint8 array in OpenCV BGR order."""
        self._ensure_loaded()
        faces = self._app.get(image_bgr)
        results = []
        for f in faces:
            x1, y1, x2, y2 = (float(v) for v in f.bbox)
            embedding = np.asarray(f.normed_embedding, dtype=np.float32)
            results.append(
                DetectedFace(
                    left=x1, top=y1, right=x2, bottom=y2,
                    embedding=embedding,
                    detector_score=float(f.det_score),
                )
            )
        return results
