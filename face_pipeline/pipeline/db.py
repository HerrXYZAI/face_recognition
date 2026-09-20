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
    processed_at TEXT NOT NULL
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
    model_version TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_name ON faces(person_name);
"""


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_processed_mtime(conn: sqlite3.Connection, path: str) -> Optional[float]:
    row = conn.execute("SELECT mtime FROM images WHERE path = ?", (path,)).fetchone()
    return row["mtime"] if row else None


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


def iter_faces_for_xmp(conn: sqlite3.Connection, min_confidence: float):
    """Yields (image_path, width, height, [(person_name, left, top, right, bottom), ...])
    for every image that has at least one face at or above min_confidence --
    used by Step 4's XMP writer. Bounding boxes are converted from the pixel
    coordinates stored in `faces` to the fractional (0..1) coordinates the
    MWG region writer expects."""
    rows = conn.execute(
        """SELECT img.path AS path, img.width AS width, img.height AS height,
                  f.person_name AS person_name, f.left AS left, f.top AS top,
                  f.right AS right, f.bottom AS bottom, f.confidence AS confidence
           FROM faces f JOIN images img ON img.id = f.image_id
           WHERE f.person_name != 'unknown' AND f.confidence >= ?
           ORDER BY img.path""",
        (min_confidence,),
    ).fetchall()

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
