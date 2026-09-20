"""Step 1: export person-tagged faces (name + bounding box) from a Lightroom
Classic catalog.

Lightroom's .lrcat format is an undocumented SQLite database, and the exact
column names for face bounding boxes have varied across versions. Rather than
hardcode one guess, this module inspects the live schema of the catalog copy
it's given, tries a small set of known column-naming schemes, and fails loudly
with the actual columns found if none match -- see `_resolve_bbox_columns`.

Always point this at a *copy* of the .lrcat file. Lightroom locks the live
catalog while running, and this keeps the operation strictly read-only by
construction.
"""
from __future__ import annotations

import csv
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

# Tables this export depends on. If Adobe renames one of these in a future
# catalog version, `export_faces` raises a clear error naming which lookup
# failed rather than silently returning nothing.
REQUIRED_TABLES = [
    "AgLibraryFace",
    "AgLibraryKeywordFace",
    "AgLibraryKeyword",
    "Adobe_images",
    "AgLibraryFile",
    "AgLibraryFolder",
    "AgLibraryRootFolder",
]


@dataclass
class LabeledFace:
    image_path: str
    person_name: str
    # Fractional bounding box, 0..1, left/top/right/bottom relative to the
    # image as Lightroom stores it (before we know the pixel dimensions).
    left: float
    top: float
    right: float
    bottom: float
    confirmed: bool  # AgLibraryKeywordFace.userPick


class CatalogSchemaError(RuntimeError):
    """Raised when the catalog doesn't match any known Lightroom schema."""


def _open_readonly(catalog_path: Path) -> sqlite3.Connection:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")
    uri = f"file:{catalog_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """Returns {lowercase_name: actual_name} for a table's columns."""
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {row["name"].lower(): row["name"] for row in rows}


def _check_required_tables(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    missing = [t for t in REQUIRED_TABLES if t not in existing]
    if missing:
        raise CatalogSchemaError(
            "This doesn't look like a Lightroom Classic catalog (or its schema "
            f"has changed): missing table(s) {missing}. Run "
            "`face-pipeline inspect-catalog` for a full table dump."
        )


# Candidate (name, columns, to_ltrb) schemes for AgLibraryFace's bounding box,
# tried in order. Each `to_ltrb` takes the resolved row (a dict keyed by the
# scheme's own column names) and returns fractional (left, top, right, bottom).
_BBOX_SCHEMES: list[tuple[str, tuple[str, ...], Callable[[dict], tuple[float, float, float, float]]]] = [
    (
        "top/left/bottom/right",
        ("top", "left", "bottom", "right"),
        lambda r: (r["left"], r["top"], r["right"], r["bottom"]),
    ),
    (
        "bl_x/bl_y/bl_width/bl_height",
        ("bl_x", "bl_y", "bl_width", "bl_height"),
        lambda r: (r["bl_x"], r["bl_y"], r["bl_x"] + r["bl_width"], r["bl_y"] + r["bl_height"]),
    ),
    (
        "tl_x/tl_y/br_x/br_y",
        ("tl_x", "tl_y", "br_x", "br_y"),
        lambda r: (r["tl_x"], r["tl_y"], r["br_x"], r["br_y"]),
    ),
]


def _resolve_bbox_columns(conn: sqlite3.Connection):
    cols = _table_columns(conn, "AgLibraryFace")
    for scheme_name, needed, to_ltrb in _BBOX_SCHEMES:
        if all(n in cols for n in needed):
            logger.info("AgLibraryFace bounding box scheme detected: %s", scheme_name)
            actual = {n: cols[n] for n in needed}
            return actual, to_ltrb
    raise CatalogSchemaError(
        "Could not find a recognized bounding-box column scheme in "
        f"AgLibraryFace. Actual columns: {sorted(cols.values())}. Add a new "
        "entry to _BBOX_SCHEMES in face_pipeline/lightroom/catalog_export.py "
        "matching these column names."
    )


def inspect_catalog(catalog_path: Path) -> str:
    """Returns a human-readable dump of the face-related tables' schemas, for
    `face-pipeline inspect-catalog` -- run this first against a real catalog
    copy to confirm/extend the column-scheme detection above."""
    conn = _open_readonly(catalog_path)
    try:
        lines = []
        existing = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in REQUIRED_TABLES:
            lines.append(f"== {table} ==")
            if table not in existing:
                lines.append("  MISSING")
                continue
            for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall():
                lines.append(f"  {row['name']} ({row['type']})")
        return "\n".join(lines)
    finally:
        conn.close()


def _iter_labeled_faces(conn: sqlite3.Connection) -> Iterator[LabeledFace]:
    bbox_cols, to_ltrb = _resolve_bbox_columns(conn)

    query = f"""
        SELECT
            face.image AS image_id,
            face.{bbox_cols[list(bbox_cols)[0]]} AS c0,
            face.{bbox_cols[list(bbox_cols)[1]]} AS c1,
            face.{bbox_cols[list(bbox_cols)[2]]} AS c2,
            face.{bbox_cols[list(bbox_cols)[3]]} AS c3,
            kw.name AS person_name,
            kwface.userPick AS user_pick,
            kwface.userReject AS user_reject,
            rootfolder.absolutePath AS root_path,
            folder.pathFromRoot AS folder_path,
            file.baseName AS base_name,
            file.extension AS extension
        FROM AgLibraryFace face
        JOIN AgLibraryKeywordFace kwface ON kwface.face = face.id_local
        JOIN AgLibraryKeyword kw ON kw.id_local = kwface.tag
        JOIN Adobe_images img ON img.id_local = face.image
        JOIN AgLibraryFile file ON file.id_local = img.rootFile
        JOIN AgLibraryFolder folder ON folder.id_local = file.folder
        JOIN AgLibraryRootFolder rootfolder ON rootfolder.id_local = folder.rootFolder
        WHERE COALESCE(kwface.userReject, 0) = 0
    """
    # The 4 bbox columns are aliased positionally (c0..c3) in the order the
    # matched scheme declared them, so `to_ltrb` can stay scheme-agnostic here.
    keys = list(bbox_cols)
    for row in conn.execute(query):
        raw = {keys[i]: row[f"c{i}"] for i in range(4)}
        left, top, right, bottom = to_ltrb(raw)
        path = str(Path(row["root_path"]) / row["folder_path"] / f"{row['base_name']}.{row['extension']}")
        yield LabeledFace(
            image_path=path,
            person_name=row["person_name"],
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            confirmed=bool(row["user_pick"]),
        )


def export_faces(catalog_path: Path, out_csv: Path) -> int:
    """Exports labeled faces to `out_csv`. Returns the number of rows written."""
    conn = _open_readonly(catalog_path)
    try:
        _check_required_tables(conn)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["image_path", "person_name", "left", "top", "right", "bottom", "confirmed"]
            )
            for face in _iter_labeled_faces(conn):
                writer.writerow(
                    [
                        face.image_path,
                        face.person_name,
                        face.left,
                        face.top,
                        face.right,
                        face.bottom,
                        int(face.confirmed),
                    ]
                )
                count += 1
        logger.info("Exported %d labeled faces to %s", count, out_csv)
        return count
    finally:
        conn.close()
