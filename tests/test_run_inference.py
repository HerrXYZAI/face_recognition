from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_pipeline.pipeline import db, run_inference
from face_pipeline.recognition.embeddings import DetectedFace


class _FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def detect(self, image_bgr):
        self.calls += 1
        return [
            DetectedFace(
                left=0, top=0, right=10, bottom=10,
                embedding=np.zeros(512, dtype=np.float32), detector_score=0.9,
            )
        ]


class _FakeClassifier:
    def predict(self, embedding):
        return "Alice", 0.9


class _CrashingClassifier:
    """Raises on its 2nd call -- simulates the process being killed/crashing
    partway through a run, to check what survives in faces.db afterward."""

    def __init__(self):
        self.calls = 0

    def predict(self, embedding):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated abrupt interruption")
        return "Alice", 0.9


def _make_images(images_dir: Path, n: int) -> list[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = images_dir / f"img{i}.jpg"
        p.write_bytes(b"fake")
        paths.append(p)
    return paths


@pytest.fixture(autouse=True)
def _fake_image_loading(monkeypatch):
    # run() only needs image dimensions here, not real decoded pixels --
    # avoids depending on real, decodable JPEG bytes in this test.
    monkeypatch.setattr(run_inference, "load_image_bgr", lambda p: np.zeros((20, 20, 3), dtype=np.uint8))


def test_default_run_skips_unchanged_images_on_rerun(tmp_path: Path):
    images_dir = tmp_path / "images"
    _make_images(images_dir, 3)
    db_path = tmp_path / "faces.db"
    embedder = _FakeEmbedder()
    classifier = _FakeClassifier()

    stats = run_inference.run(images_dir, db_path, embedder, classifier, min_confidence=0.5)
    assert stats["images_processed"] == 3
    assert embedder.calls == 3

    # Re-running without restart: every image's mtime is unchanged, so all
    # three are skipped and detect() isn't called again for any of them.
    stats2 = run_inference.run(images_dir, db_path, embedder, classifier, min_confidence=0.5)
    assert stats2 == {"images_processed": 0, "images_skipped": 3, "faces_found": 0, "errors": 0}
    assert embedder.calls == 3


def test_restart_reprocesses_every_image_regardless_of_mtime(tmp_path: Path):
    images_dir = tmp_path / "images"
    _make_images(images_dir, 3)
    db_path = tmp_path / "faces.db"
    embedder = _FakeEmbedder()
    classifier = _FakeClassifier()

    run_inference.run(images_dir, db_path, embedder, classifier, min_confidence=0.5)
    assert embedder.calls == 3

    stats = run_inference.run(images_dir, db_path, embedder, classifier, min_confidence=0.5, restart=True)
    assert stats["images_processed"] == 3
    assert stats["images_skipped"] == 0
    assert embedder.calls == 6  # detect() re-ran for every image, not skipped


def test_run_commits_each_image_so_earlier_work_survives_an_interruption(tmp_path: Path):
    images_dir = tmp_path / "images"
    _make_images(images_dir, 3)
    db_path = tmp_path / "faces.db"
    embedder = _FakeEmbedder()
    classifier = _CrashingClassifier()

    # classify() raising is not caught anywhere in run(), so this propagates
    # out -- simulating the process dying mid-run -- before all 3 images are
    # processed.
    with pytest.raises(RuntimeError):
        run_inference.run(images_dir, db_path, embedder, classifier, min_confidence=0.5)

    # Exactly one image (whichever ran first) reached a full image+face
    # commit before the crash on the second one. Committing per image
    # (rather than batching many images per commit) is what makes that
    # first image's work survive instead of being rolled back along with
    # the second, incomplete image.
    with db.connect(db_path) as conn:
        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        n_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    assert n_images == 1
    assert n_faces == 1
