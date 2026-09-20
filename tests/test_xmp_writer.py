from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from face_pipeline.lightroom.xmp_writer import (
    _build_region_info,
    _escape,
    target_path_for,
    write_regions,
)

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


def test_build_region_info_uses_center_coordinates():
    # A face spanning the full image (0,0)-(1,1) should end up centered at
    # (0.5, 0.5) with width/height 1 -- MWG regions are center-based, unlike
    # Lightroom's own left/top/right/bottom storage.
    info = _build_region_info(1000, 800, [("Alice", 0.0, 0.0, 1.0, 1.0)])
    assert "X=0.5,Y=0.5,W=1.0,H=1.0" in info
    assert "Name=Alice" in info
    assert "W=1000,H=800" in info


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="exiftool not installed")
def test_write_regions_roundtrip_sidecar(tmp_path: Path):
    image_path = tmp_path / "photo.cr2"
    image_path.write_bytes(b"not a real raw file, just needs to exist as a path")

    write_regions(image_path, 1000, 800, [("Alice", 0.1, 0.1, 0.3, 0.3)], dry_run=False)

    sidecar = tmp_path / "photo.xmp"
    assert sidecar.exists()

    result = subprocess.run(
        ["exiftool", "-struct", "-j", "-XMP-mwg-rs:RegionInfo", str(sidecar)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Alice" in result.stdout
