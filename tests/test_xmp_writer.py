from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from face_pipeline.lightroom.xmp_writer import (
    _build_region_info,
    _escape,
    _merge_region_items,
    _region_struct_item,
    require_exiftool,
    target_path_for,
    write_approved_regions,
    write_regions,
)
from face_pipeline.pipeline import db

HAS_EXIFTOOL = shutil.which("exiftool") is not None


def test_target_path_for_embed_format():
    path, is_sidecar = target_path_for(Path("photo.jpg"))
    assert path == Path("photo.jpg")
    assert is_sidecar is False


def test_target_path_for_raw_format_uses_sidecar():
    path, is_sidecar = target_path_for(Path("photo.CR2"))
    assert path == Path("photo.xmp")
    assert is_sidecar is True


def test_target_path_for_unknown_extension_raises():
    with pytest.raises(ValueError):
        target_path_for(Path("photo.weird"))


def test_escape_special_characters():
    assert _escape("Smith, John") == "Smith|, John"
    assert _escape("{Bob}") == "|{Bob|}"


def test_region_struct_item_uses_center_coordinates():
    # A face spanning the full image (0,0)-(1,1) should end up centered at
    # (0.5, 0.5) with width/height 1 -- MWG regions are center-based, unlike
    # Lightroom's own left/top/right/bottom storage.
    item = _region_struct_item("Alice", 0.0, 0.0, 1.0, 1.0)
    assert "X=0.5,Y=0.5,W=1.0,H=1.0" in item
    assert "Name=Alice" in item
    assert "Type=Face" in item


def test_build_region_info_wraps_items_with_dimensions():
    info = _build_region_info(1000, 800, [_region_struct_item("Alice", 0.0, 0.0, 1.0, 1.0)])
    assert "W=1000,H=800" in info
    assert "Name=Alice" in info


def test_merge_region_items_keeps_non_overlapping_existing():
    # A region for "Bob" already in the file, nowhere near the new "Alice"
    # face, must survive the merge -- this is the "attach, don't overwrite"
    # behavior: only spatially-matching regions get replaced.
    existing = [{"Name": "Bob", "Type": "Face", "Area": {"X": 0.8, "Y": 0.8, "W": 0.1, "H": 0.1, "Unit": "normalized"}}]
    items = _merge_region_items(existing, [("Alice", 0.0, 0.0, 0.2, 0.2)], Path("irrelevant.xmp"))
    assert any("Name=Bob" in item for item in items)
    assert any("Name=Alice" in item for item in items)
    assert len(items) == 2


def test_merge_region_items_replaces_overlapping_existing():
    # An existing "Bob" region that overlaps the new "Alice" region closely
    # is the same physical face relabeled/reprocessed -- it should be
    # replaced in place, not kept alongside the new one.
    existing = [{"Name": "Bob", "Type": "Face", "Area": {"X": 0.1, "Y": 0.1, "W": 0.2, "H": 0.2, "Unit": "normalized"}}]
    items = _merge_region_items(existing, [("Alice", 0.0, 0.0, 0.2, 0.2)], Path("irrelevant.xmp"))
    assert len(items) == 1
    assert "Name=Alice" in items[0]


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool not installed")
def test_write_regions_roundtrip_sidecar(tmp_path: Path):
    image_path = tmp_path / "photo.cr2"
    image_path.write_bytes(b"not a real raw file, just needs to exist as a path")

    write_regions(image_path, 1000, 800, [("Alice", 0.1, 0.1, 0.3, 0.3)], dry_run=False)

    sidecar = tmp_path / "photo.xmp"
    assert sidecar.exists()

    result = subprocess.run(
        [require_exiftool(), "-struct", "-j", "-XMP-mwg-rs:RegionInfo", str(sidecar)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Alice" in result.stdout


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool not installed")
def test_write_regions_second_call_attaches_instead_of_overwriting(tmp_path: Path):
    image_path = tmp_path / "photo.cr2"
    image_path.write_bytes(b"not a real raw file, just needs to exist as a path")

    write_regions(image_path, 1000, 800, [("Alice", 0.1, 0.1, 0.3, 0.3)], dry_run=False)
    # A second, later write-xmp run approving a different person in the same
    # photo must not clobber Alice's region from the first run.
    write_regions(image_path, 1000, 800, [("Bob", 0.6, 0.6, 0.9, 0.9)], dry_run=False)

    sidecar = tmp_path / "photo.xmp"
    result = subprocess.run(
        [require_exiftool(), "-struct", "-j", "-XMP-mwg-rs:RegionInfo", str(sidecar)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Alice" in result.stdout
    assert "Bob" in result.stdout


def _setup_two_approved_faces(db_path: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Two images, each with one approved face, as write_approved_regions
    would find them via db.iter_faces_for_xmp."""
    image_a = tmp_path / "a.cr2"
    image_a.write_bytes(b"not a real raw file, just needs to exist as a path")
    image_b = tmp_path / "b.cr2"
    image_b.write_bytes(b"not a real raw file, just needs to exist as a path")

    with db.connect(db_path) as conn:
        img_a = db.upsert_image(conn, str(image_a), 1000, 800, 1.0, "t0")
        db.insert_face(conn, img_a, 100, 100, 300, 300, np.zeros(512, dtype=np.float32), "Alice", 0.9, 0.9, "v1")
        alice_id = conn.execute("SELECT id FROM faces WHERE person_name = 'Alice'").fetchone()["id"]
        db.set_review_status(conn, alice_id, "approved", reviewed_at="t")

        img_b = db.upsert_image(conn, str(image_b), 1000, 800, 1.0, "t0")
        db.insert_face(conn, img_b, 100, 100, 300, 300, np.zeros(512, dtype=np.float32), "Bob", 0.9, 0.9, "v1")
        bob_id = conn.execute("SELECT id FROM faces WHERE person_name = 'Bob'").fetchone()["id"]
        db.set_review_status(conn, bob_id, "approved", reviewed_at="t")

    return image_a, image_b


def _exported_marks(db_path: Path) -> list:
    with db.connect(db_path) as conn:
        return [row["xmp_exported_at"] for row in conn.execute("SELECT xmp_exported_at FROM images ORDER BY path")]


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool not installed")
def test_write_approved_regions_full_run_clears_marks_on_completion(tmp_path: Path):
    db_path = tmp_path / "faces.db"
    image_a, image_b = _setup_two_approved_faces(db_path, tmp_path)

    stats = write_approved_regions(db_path)
    assert stats == {"written": 2, "failed": 0, "skipped_already_exported": 0}
    assert image_a.with_suffix(".xmp").exists()
    assert image_b.with_suffix(".xmp").exists()

    # A run that reaches every candidate image clears all marks on the way
    # out, even though nothing here was interrupted -- the next invocation
    # (resumed or not) should see nothing pre-exported.
    assert _exported_marks(db_path) == [None, None]


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool not installed")
def test_write_approved_regions_resume_skips_already_exported(tmp_path: Path):
    db_path = tmp_path / "faces.db"
    image_a, image_b = _setup_two_approved_faces(db_path, tmp_path)

    # Simulate an interrupted first run: a.cr2 already marked exported,
    # b.cr2 never got there.
    with db.connect(db_path) as conn:
        db.mark_xmp_exported(conn, str(image_a), "2024-01-01T00:00:00")

    stats = write_approved_regions(db_path, resume=True)
    assert stats["written"] == 1
    assert stats["skipped_already_exported"] == 1
    assert not image_a.with_suffix(".xmp").exists()  # skipped, untouched
    assert image_b.with_suffix(".xmp").exists()

    # Completing the resumed run clears every mark, a.cr2's included.
    assert _exported_marks(db_path) == [None, None]


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool not installed")
def test_write_approved_regions_dry_run_previews_without_marking(tmp_path: Path):
    db_path = tmp_path / "faces.db"
    image_a, image_b = _setup_two_approved_faces(db_path, tmp_path)

    stats = write_approved_regions(db_path, dry_run=True)
    assert stats["written"] == 2
    assert not image_a.with_suffix(".xmp").exists()
    assert not image_b.with_suffix(".xmp").exists()
    assert _exported_marks(db_path) == [None, None]  # dry-run marks/clears nothing


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool not installed")
def test_write_approved_regions_default_ignores_stale_marks_from_a_prior_interrupted_run(tmp_path: Path):
    db_path = tmp_path / "faces.db"
    image_a, image_b = _setup_two_approved_faces(db_path, tmp_path)

    with db.connect(db_path) as conn:
        db.mark_xmp_exported(conn, str(image_a), "2024-01-01T00:00:00")

    # resume=False (the default, i.e. "start again") reprocesses everything,
    # a.cr2 included, even though a previous interrupted run had marked it.
    stats = write_approved_regions(db_path, resume=False)
    assert stats["written"] == 2
    assert image_a.with_suffix(".xmp").exists()
    assert image_b.with_suffix(".xmp").exists()
