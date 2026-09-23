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

        # Freshly inserted faces default to review_status='pending' and
        # aren't written until a human approves them.
        assert list(db.iter_faces_for_xmp(conn)) == []

        alice_id = conn.execute("SELECT id FROM faces WHERE person_name = 'Alice'").fetchone()["id"]
        db.set_review_status(conn, alice_id, "approved", reviewed_at="2024-01-02T00:00:00")

        rows = list(db.iter_faces_for_xmp(conn))
        assert len(rows) == 1
        path, width, height, faces = rows[0]
        assert path == "/photos/a.jpg"
        assert (width, height) == (800, 600)
        # iter_faces_for_xmp converts the pixel bboxes stored in `faces` to
        # fractional (0..1) coordinates, since that's what the MWG region
        # writer (Step 5) expects.
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
        # for this path -- and their review decisions -- must be gone after
        # the re-upsert.
        db.upsert_image(conn, "/photos/a.jpg", 800, 600, 2.0, "t1")
        n_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        assert n_faces == 0


def test_set_review_status_rejects_approving_unknown(tmp_path: Path):
    db_path = tmp_path / "faces.db"

    with db.connect(db_path) as conn:
        image_id = db.upsert_image(conn, "/photos/a.jpg", 800, 600, 1.0, "t0")
        db.insert_face(
            conn, image_id, 0, 0, 10, 10, np.zeros(512, dtype=np.float32), "unknown", 0.2, 0.9, "v1"
        )
        face_id = conn.execute("SELECT id FROM faces").fetchone()["id"]
        with pytest.raises(ValueError):
            db.set_review_status(conn, face_id, "approved", person_name="unknown")


def test_set_review_status_relabels_and_approves(tmp_path: Path):
    db_path = tmp_path / "faces.db"

    with db.connect(db_path) as conn:
        image_id = db.upsert_image(conn, "/photos/a.jpg", 800, 600, 1.0, "t0")
        db.insert_face(
            conn, image_id, 0, 0, 10, 10, np.zeros(512, dtype=np.float32), "unknown", 0.2, 0.9, "v1"
        )
        face_id = conn.execute("SELECT id FROM faces").fetchone()["id"]
        db.set_review_status(conn, face_id, "approved", person_name="Carol", reviewed_at="t")

    with db.connect(db_path) as conn:
        row = conn.execute("SELECT person_name, review_status FROM faces WHERE id = ?", (face_id,)).fetchone()
        assert row["person_name"] == "Carol"
        assert row["review_status"] == "approved"

        rows = list(db.iter_faces_for_xmp(conn))
        assert rows[0][3][0][0] == "Carol"


def test_iter_faces_for_review_filters_by_confidence_and_orders_ascending(tmp_path: Path):
    db_path = tmp_path / "faces.db"

    with db.connect(db_path) as conn:
        image_id = db.upsert_image(conn, "/photos/a.jpg", 800, 600, 1.0, "t0")
        db.insert_face(conn, image_id, 0, 0, 10, 10, np.zeros(512, dtype=np.float32), "Bob", 0.4, 0.9, "v1")
        db.insert_face(conn, image_id, 20, 20, 30, 30, np.zeros(512, dtype=np.float32), "Alice", 0.8, 0.9, "v1")
        db.insert_face(conn, image_id, 40, 40, 50, 50, np.zeros(512, dtype=np.float32), "Alice", 0.6, 0.9, "v1")

    with db.connect(db_path) as conn:
        rows = db.iter_faces_for_review(conn, review_status="pending", min_confidence=0.5)
        names = [row["person_name"] for row in rows]
        assert names == ["Alice", "Alice"]  # Bob (0.4) excluded by min_confidence
        confidences = [row["confidence"] for row in rows]
        assert confidences == sorted(confidences)  # lowest-confidence first

        alice_only = db.iter_faces_for_review(conn, review_status="pending", min_confidence=0.0, person_name="Alice")
        assert len(alice_only) == 2


def test_review_counts(tmp_path: Path):
    db_path = tmp_path / "faces.db"

    with db.connect(db_path) as conn:
        image_id = db.upsert_image(conn, "/photos/a.jpg", 800, 600, 1.0, "t0")
        db.insert_face(conn, image_id, 0, 0, 10, 10, np.zeros(512, dtype=np.float32), "Alice", 0.9, 0.9, "v1")
        db.insert_face(conn, image_id, 20, 20, 30, 30, np.zeros(512, dtype=np.float32), "Bob", 0.9, 0.9, "v1")
        ids = [row["id"] for row in conn.execute("SELECT id FROM faces ORDER BY id")]
        db.set_review_status(conn, ids[0], "approved", reviewed_at="t")
        db.set_review_status(conn, ids[1], "rejected", reviewed_at="t")

    with db.connect(db_path) as conn:
        assert db.review_counts(conn) == {"pending": 0, "approved": 1, "rejected": 1}


def test_iter_faces_for_xmp_skip_exported_and_clear_marks(tmp_path: Path):
    db_path = tmp_path / "faces.db"

    with db.connect(db_path) as conn:
        img_a = db.upsert_image(conn, "/photos/a.jpg", 800, 600, 1.0, "t0")
        db.insert_face(conn, img_a, 0, 0, 10, 10, np.ones(512, dtype=np.float32), "Alice", 0.9, 0.9, "v1")
        alice_id = conn.execute("SELECT id FROM faces WHERE person_name = 'Alice'").fetchone()["id"]
        db.set_review_status(conn, alice_id, "approved", reviewed_at="t")

        img_b = db.upsert_image(conn, "/photos/b.jpg", 800, 600, 1.0, "t0")
        db.insert_face(conn, img_b, 0, 0, 10, 10, np.ones(512, dtype=np.float32), "Bob", 0.9, 0.9, "v1")
        bob_id = conn.execute("SELECT id FROM faces WHERE person_name = 'Bob'").fetchone()["id"]
        db.set_review_status(conn, bob_id, "approved", reviewed_at="t")

    with db.connect(db_path) as conn:
        assert len(list(db.iter_faces_for_xmp(conn))) == 2

        # Marking a.jpg as already exported (simulating an interrupted
        # write-xmp run) hides it from a skip_exported=True query -- what a
        # resumed run uses to avoid redoing it -- but not from the default.
        db.mark_xmp_exported(conn, "/photos/a.jpg", "2024-01-01T00:00:00")
        assert len(list(db.iter_faces_for_xmp(conn))) == 2
        remaining = list(db.iter_faces_for_xmp(conn, skip_exported=True))
        assert len(remaining) == 1
        assert remaining[0][0] == "/photos/b.jpg"

        # Clearing marks (done once a write-xmp pass finishes) makes both
        # images visible again, as if nothing had ever been exported.
        db.clear_xmp_exported_marks(conn)
        assert len(list(db.iter_faces_for_xmp(conn, skip_exported=True))) == 2


def test_distinct_person_names_excludes_unknown(tmp_path: Path):
    db_path = tmp_path / "faces.db"

    with db.connect(db_path) as conn:
        image_id = db.upsert_image(conn, "/photos/a.jpg", 800, 600, 1.0, "t0")
        db.insert_face(conn, image_id, 0, 0, 10, 10, np.zeros(512, dtype=np.float32), "Alice", 0.9, 0.9, "v1")
        db.insert_face(conn, image_id, 20, 20, 30, 30, np.zeros(512, dtype=np.float32), "unknown", 0.2, 0.9, "v1")

    with db.connect(db_path) as conn:
        assert db.distinct_person_names(conn) == ["Alice"]
