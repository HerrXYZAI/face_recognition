from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_pipeline.pipeline import db


def test_upsert_and_query_roundtrip(tmp_path: Path):
    db_path = tmp_path / "faces.db"

    with db.connect(db_path) as conn:
        image_id = db.upsert_image(conn, "/photos/a.jpg", 800, 600, 12345.0, "2024-01-01T00:00:00")
        db.insert_face(
            conn, image_id, 10, 20, 110, 220,
            np.ones(512, dtype=np.float32), "Alice", 0.9, 0.99, "test-v1",
        )
        db.insert_face(
            conn, image_id, 200, 20, 300, 220,
            np.zeros(512, dtype=np.float32), "unknown", 0.3, 0.95, "test-v1",
        )

    with db.connect(db_path) as conn:
        assert db.get_processed_mtime(conn, "/photos/a.jpg") == 12345.0
        assert db.get_processed_mtime(conn, "/photos/missing.jpg") is None

        rows = list(db.iter_faces_for_xmp(conn, min_confidence=0.5))
        assert len(rows) == 1
        path, width, height, faces = rows[0]
        assert path == "/photos/a.jpg"
        assert (width, height) == (800, 600)
        # iter_faces_for_xmp converts the pixel bboxes stored in `faces` to
        # fractional (0..1) coordinates, since that's what the MWG region
        # writer (Step 4) expects.
        assert len(faces) == 1
        name, left, top, right, bottom = faces[0]
        assert name == "Alice"
        assert (left, top, right, bottom) == pytest.approx((10 / 800, 20 / 600, 110 / 800, 220 / 600))


def test_upsert_image_replaces_and_cascades_faces(tmp_path: Path):
    db_path = tmp_path / "faces.db"

    with db.connect(db_path) as conn:
        image_id = db.upsert_image(conn, "/photos/a.jpg", 800, 600, 1.0, "t0")
        db.insert_face(
            conn, image_id, 0, 0, 10, 10, np.ones(512, dtype=np.float32), "Alice", 0.9, 0.9, "v1"
        )

    with db.connect(db_path) as conn:
        # Simulate a re-run after the file changed (new mtime): old faces
        # for this path must be gone after the re-upsert.
        db.upsert_image(conn, "/photos/a.jpg", 800, 600, 2.0, "t1")
        n_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        assert n_faces == 0
