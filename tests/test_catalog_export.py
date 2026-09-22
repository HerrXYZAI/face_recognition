"""Exercises catalog_export against a small synthetic SQLite file mimicking
the relevant Lightroom Classic tables -- doesn't require a real catalog."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from face_pipeline.lightroom.catalog_export import export_faces, inspect_catalog


def _build_fixture_catalog(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE AgLibraryFace (
            id_local INTEGER PRIMARY KEY, image INTEGER,
            top REAL, left REAL, bottom REAL, right REAL
        );
        CREATE TABLE AgLibraryKeywordFace (
            id_local INTEGER PRIMARY KEY, face INTEGER, tag INTEGER,
            userPick INTEGER, userReject INTEGER
        );
        CREATE TABLE AgLibraryKeyword (id_local INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE Adobe_images (id_local INTEGER PRIMARY KEY, rootFile INTEGER);
        CREATE TABLE AgLibraryFile (
            id_local INTEGER PRIMARY KEY, folder INTEGER, baseName TEXT, extension TEXT
        );
        CREATE TABLE AgLibraryFolder (
            id_local INTEGER PRIMARY KEY, rootFolder INTEGER, pathFromRoot TEXT
        );
        CREATE TABLE AgLibraryRootFolder (id_local INTEGER PRIMARY KEY, absolutePath TEXT);
        """
    )
    conn.executescript(
        """
        INSERT INTO AgLibraryRootFolder VALUES (1, '/photos/');
        INSERT INTO AgLibraryFolder VALUES (1, 1, '2024/vacation/');
        INSERT INTO AgLibraryFile VALUES (1, 1, 'IMG_0001', 'jpg');
        INSERT INTO Adobe_images VALUES (1, 1);
        INSERT INTO AgLibraryKeyword VALUES (1, 'Alice');
        INSERT INTO AgLibraryFace VALUES (1, 1, 0.1, 0.2, 0.4, 0.5);
        -- confirmed match for Alice
        INSERT INTO AgLibraryKeywordFace VALUES (1, 1, 1, 1, 0);
        -- a rejected suggestion that must be excluded from export
        INSERT INTO AgLibraryKeyword VALUES (2, 'Bob');
        INSERT INTO AgLibraryFace VALUES (2, 1, 0.5, 0.6, 0.8, 0.9);
        INSERT INTO AgLibraryKeywordFace VALUES (2, 2, 2, 0, 1);
        """
    )
    conn.commit()
    conn.close()


def test_export_faces_reads_confirmed_only(tmp_path: Path):
    catalog = tmp_path / "catalog.lrcat"
    _build_fixture_catalog(catalog)
    out_csv = tmp_path / "labeled_faces.csv"

    count = export_faces(catalog, out_csv, Path("/photos/"))
    assert count == 1

    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["person_name"] == "Alice"
    assert Path(row["image_path"]) == Path("/photos/2024/vacation/IMG_0001.jpg")
    assert float(row["left"]) == 0.2
    assert float(row["top"]) == 0.1
    assert float(row["right"]) == 0.5
    assert float(row["bottom"]) == 0.4
    assert row["confirmed"] == "1"


def test_export_faces_resolves_under_configured_images_root(tmp_path: Path):
    """The catalog's own recorded root_path ('/photos/') only matches the
    real filesystem when running on the same machine/mount layout Lightroom
    saw at import time (e.g. bare-metal via run.bat). Under Docker the host
    folder is bind-mounted elsewhere (e.g. /data/images), so export_faces
    must rebuild paths under the *configured* images_root instead."""
    catalog = tmp_path / "catalog.lrcat"
    _build_fixture_catalog(catalog)
    out_csv = tmp_path / "labeled_faces.csv"

    export_faces(catalog, out_csv, Path("/data/images"))

    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert Path(rows[0]["image_path"]) == Path("/data/images/2024/vacation/IMG_0001.jpg")


def test_inspect_catalog_lists_columns(tmp_path: Path):
    catalog = tmp_path / "catalog.lrcat"
    _build_fixture_catalog(catalog)

    output = inspect_catalog(catalog)

    assert "AgLibraryFace" in output
    assert "left" in output
