"""SQLite schema + helpers for faces.db -- this project's own database of
every detected face across the photo library.

Kept entirely separate from Lightroom's .lrcat: nothing in this project ever
writes into the Lightroom catalog itself, so there's no risk to it.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from face_pipeline.imaging import pixel_to_fractional_bbox

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    mtime REAL NOT NULL,
    processed_at TEXT NOT NULL,
    xmp_exported_at TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    left REAL NOT NULL,
    top REAL NOT NULL,
    right REAL NOT NULL,
    bottom REAL NOT NULL,
    embedding BLOB NOT NULL,
    person_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    detector_score REAL NOT NULL,
    model_version TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',
    reviewed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_name ON faces(person_name);
"""

REVIEW_STATUSES = {"pending", "approved", "rejected"}
UNKNOWN_NAME = "unknown"


def _migrate(conn: sqlite3.Connection) -> None:
    """Adds columns introduced after a faces.db may already have been
    created, so existing databases pick them up instead of erroring."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(faces)")}
    if "review_status" not in cols:
        conn.execute("ALTER TABLE faces ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'")
    if "reviewed_at" not in cols:
        conn.execute("ALTER TABLE faces ADD COLUMN reviewed_at TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_review_status ON faces(review_status)")

    image_cols = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
    if "xmp_exported_at" not in image_cols:
        conn.execute("ALTER TABLE images ADD COLUMN xmp_exported_at TEXT")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_processed_mtime(conn: sqlite3.Connection, path: str) -> Optional[float]:
    row = conn.execute("SELECT mtime FROM images WHERE path = ?", (path,)).fetchone()
    return row["mtime"] if row else None


def mark_xmp_exported(conn: sqlite3.Connection, image_path: str, exported_at: str) -> None:
    """Records that `image_path`'s approved faces were just written out by
    write-xmp, so a later resumed run can skip it via
    `iter_faces_for_xmp(..., skip_exported=True)`."""
    conn.execute("UPDATE images SET xmp_exported_at = ? WHERE path = ?", (exported_at, image_path))


def clear_xmp_exported_marks(conn: sqlite3.Connection) -> None:
    """Clears every xmp-export mark. Called once a write-xmp pass has gone
    through every candidate image (whether resumed or not), so the marks
    only ever reflect an in-progress/interrupted run -- the next invocation
    always starts from a clean slate."""
    conn.execute("UPDATE images SET xmp_exported_at = NULL")


def upsert_image(
    conn: sqlite3.Connection, path: str, width: int, height: int, mtime: float, processed_at: str
) -> int:
    """Replaces any existing row (and, via ON DELETE CASCADE, its faces) for
    this path -- this is what makes a re-run after a file changes idempotent."""
    conn.execute("DELETE FROM images WHERE path = ?", (path,))
    cur = conn.execute(
        "INSERT INTO images (path, width, height, mtime, processed_at) VALUES (?, ?, ?, ?, ?)",
        (path, width, height, mtime, processed_at),
    )
    return cur.lastrowid


def insert_face(
    conn: sqlite3.Connection,
    image_id: int,
    left: float, top: float, right: float, bottom: float,
    embedding: np.ndarray,
    person_name: str,
    confidence: float,
    detector_score: float,
    model_version: str,
) -> None:
    conn.execute(
        """INSERT INTO faces
           (image_id, left, top, right, bottom, embedding, person_name,
            confidence, detector_score, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            image_id, left, top, right, bottom,
            np.asarray(embedding, dtype=np.float32).tobytes(),
            person_name, confidence, detector_score, model_version,
        ),
    )


def iter_faces_for_xmp(conn: sqlite3.Connection, skip_exported: bool = False):
    """Yields (image_path, width, height, [(person_name, left, top, right, bottom), ...])
    for every image that has at least one *approved* face -- used by the
    write-xmp step. Only faces a human has approved via `review-faces` are
    written; a face's automatic classification confidence no longer gates
    this on its own, since approval is a strictly stronger signal (it can
    also rescue a low-confidence face the classifier under-scored). Bounding
    boxes are converted from the pixel coordinates stored in `faces` to the
    fractional (0..1) coordinates the MWG region writer expects.

    skip_exported=True excludes images already marked via
    `mark_xmp_exported` -- used to resume a write-xmp run that was
    interrupted partway through without redoing what it already wrote."""
    query = (
        """SELECT img.path AS path, img.width AS width, img.height AS height,
                  f.person_name AS person_name, f.left AS left, f.top AS top,
                  f.right AS right, f.bottom AS bottom, f.confidence AS confidence
           FROM faces f JOIN images img ON img.id = f.image_id
           WHERE f.review_status = 'approved' AND f.person_name != 'unknown'"""
    )
    if skip_exported:
        query += " AND img.xmp_exported_at IS NULL"
    query += " ORDER BY img.path"
    rows = conn.execute(query).fetchall()

    current_path = None
    current_dims = None
    current_faces: list[tuple[str, float, float, float, float]] = []
    for row in rows:
        if row["path"] != current_path:
            if current_path is not None:
                yield current_path, current_dims[0], current_dims[1], current_faces
            current_path = row["path"]
            current_dims = (row["width"], row["height"])
            current_faces = []
        left, top, right, bottom = pixel_to_fractional_bbox(
            row["left"], row["top"], row["right"], row["bottom"], current_dims[0], current_dims[1]
        )
        current_faces.append((row["person_name"], left, top, right, bottom))
    if current_path is not None:
        yield current_path, current_dims[0], current_dims[1], current_faces


def iter_faces_for_review(
    conn: sqlite3.Connection,
    review_status: str = "pending",
    min_confidence: float = 0.0,
    person_name: Optional[str] = None,
) -> list[sqlite3.Row]:
    """Returns candidate faces for the review UI: id, image path/dimensions,
    bounding box, predicted name, and both confidence scores. Ordered lowest
    classifier-confidence first, since those predictions are the most likely
    to be wrong and so the most worth a human's attention first."""
    query = (
        "SELECT f.id AS id, img.path AS path, img.width AS width, img.height AS height, "
        "f.left AS left, f.top AS top, f.right AS right, f.bottom AS bottom, "
        "f.person_name AS person_name, f.confidence AS confidence, "
        "f.detector_score AS detector_score "
        "FROM faces f JOIN images img ON img.id = f.image_id "
        "WHERE f.review_status = ? AND f.confidence >= ?"
    )
    params: list = [review_status, min_confidence]
    if person_name is not None:
        query += " AND f.person_name = ?"
        params.append(person_name)
    query += " ORDER BY f.confidence ASC, img.path"
    return conn.execute(query, params).fetchall()


def set_review_status(
    conn: sqlite3.Connection,
    face_id: int,
    review_status: str,
    person_name: Optional[str] = None,
    reviewed_at: Optional[str] = None,
) -> None:
    """Records a human review decision for one face. Passing `person_name`
    also corrects the classifier's predicted label (used when the reviewer
    relabels a misidentified or previously-unknown face)."""
    if review_status not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review_status: {review_status!r} (expected one of {REVIEW_STATUSES})")
    if review_status == "approved" and (person_name or "").strip().lower() == UNKNOWN_NAME:
        raise ValueError("Cannot approve a face labeled 'unknown' -- assign a real name first, or reject it instead.")

    if person_name is not None:
        conn.execute(
            "UPDATE faces SET review_status = ?, person_name = ?, reviewed_at = ? WHERE id = ?",
            (review_status, person_name, reviewed_at, face_id),
        )
    else:
        conn.execute(
            "UPDATE faces SET review_status = ?, reviewed_at = ? WHERE id = ?",
            (review_status, reviewed_at, face_id),
        )


def get_review_state(conn: sqlite3.Connection, face_id: int) -> Optional[sqlite3.Row]:
    """Returns a face's current (review_status, person_name, reviewed_at) --
    used by the review UI to snapshot a face's state before changing it, so
    an "undo" can restore exactly what was there before."""
    return conn.execute(
        "SELECT review_status, person_name, reviewed_at FROM faces WHERE id = ?", (face_id,)
    ).fetchone()


def distinct_person_names(conn: sqlite3.Connection) -> list[str]:
    """Every real (non-'unknown') person name currently in faces.db, for
    populating the review UI's relabel dropdown."""
    rows = conn.execute(
        "SELECT DISTINCT person_name FROM faces WHERE person_name != 'unknown' ORDER BY person_name"
    ).fetchall()
    return [row["person_name"] for row in rows]


def review_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Face counts per review_status, e.g. for the `status` CLI command and
    the review UI's progress readout."""
    counts = {status: 0 for status in REVIEW_STATUSES}
    for row in conn.execute("SELECT review_status, COUNT(*) AS n FROM faces GROUP BY review_status"):
        counts[row["review_status"]] = row["n"]
    return counts
