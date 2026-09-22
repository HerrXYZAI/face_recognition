"""Step 4: write detected face regions back out as Metadata Working Group
(MWG) face regions, so Lightroom Classic can display them as if it had
detected them itself.

Shells out to `exiftool` (bundled in the Docker images) since it has the most
reliable, well-tested support for writing the MWG `RegionInfo` structure of
any tool available from Python.

Where the data goes matches Lightroom Classic's own convention:
- Formats Lightroom can embed XMP into directly (DNG, JPEG, TIFF, PSD) get
  the region written straight into the file.
- Proprietary raw formats (CR2, NEF, ARW, ...) get an external `<name>.xmp`
  sidecar next to the raw file, which is what Lightroom itself reads/writes
  for those formats.

IMPORTANT (documented, not automatable): Lightroom only auto-imports MWG
regions from XMP when a file is first *imported*. For photos already in your
catalog, select them in Lightroom and run Metadata > Read Metadata from Files
after running this step, so the new regions and names appear.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from face_pipeline.imaging import iou

logger = logging.getLogger(__name__)

EMBED_EXTENSIONS = {".dng", ".jpg", ".jpeg", ".tif", ".tiff", ".psd", ".heic"}
SIDECAR_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".pef"}

# Two regions are treated as "the same physical face" (update in place) when
# their boxes overlap by at least this much; below it, both are kept side by
# side. Matches the threshold train_classifier.py uses for the same kind of
# bounding-box matching.
REGION_MATCH_IOU = 0.3

# ExifTool's struct-value syntax escapes special characters with a leading
# "|", not a backslash (see https://exiftool.org/struct.html) -- e.g. a
# literal comma in a value is written as "|,".
_SPECIAL_CHARS = set(",{}[]|")


def _escape(value: str) -> str:
    return "".join(f"|{c}" if c in _SPECIAL_CHARS else c for c in value)


def require_exiftool() -> str:
    exe = shutil.which("exiftool")
    if exe is None:
        raise RuntimeError(
            "exiftool was not found on PATH. It's included in the project's "
            "Docker images; if running outside Docker, install it from "
            "https://exiftool.org and ensure it's on PATH."
        )
    return exe


def target_path_for(image_path: Path) -> tuple[Path, bool]:
    """Returns (path_to_write, is_sidecar) for an image path."""
    ext = image_path.suffix.lower()
    if ext in EMBED_EXTENSIONS:
        return image_path, False
    if ext in SIDECAR_EXTENSIONS:
        return image_path.with_suffix(".xmp"), True
    raise ValueError(
        f"Don't know whether {image_path.suffix!r} should use an XMP sidecar "
        f"or embedded XMP -- add it to EMBED_EXTENSIONS or SIDECAR_EXTENSIONS "
        f"in {__file__}."
    )


def _region_struct_item(
    name: str, left: float, top: float, right: float, bottom: float, region_type: str = "Face"
) -> str:
    """left/top/right/bottom: fractional 0..1. MWG regions use a CENTER x/y +
    width/height, normalized 0..1 -- distinct from Lightroom's own
    top/left/bottom/right internal storage."""
    cx, cy = (left + right) / 2, (top + bottom) / 2
    w, h = right - left, bottom - top
    return "{Area={X=%s,Y=%s,W=%s,H=%s,Unit=normalized},Name=%s,Type=%s}" % (
        cx, cy, w, h, _escape(name), _escape(region_type),
    )


def _build_region_info(width: int, height: int, region_items: list[str]) -> str:
    region_list = "[" + ",".join(region_items) + "]"
    return (
        "{AppliedToDimensions={W=%d,H=%d,Unit=pixel},RegionList=%s}"
        % (width, height, region_list)
    )


def _existing_region_list(exiftool: str, target: Path) -> list[dict]:
    """Reads the MWG face regions already present at `target`, if any, so a
    re-run can update/attach to them instead of clobbering the whole list.
    Returns [] if the file/sidecar doesn't exist yet or has no regions."""
    if not target.exists():
        return []
    result = subprocess.run(
        [exiftool, "-j", "-struct", "-XMP-mwg-rs:RegionInfo", str(target)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning(
            "Could not read existing regions from %s (unparseable exiftool output) -- "
            "writing without merging.", target,
        )
        return []
    if not parsed:
        return []
    region_info = parsed[0].get("RegionInfo")
    if not region_info:
        return []
    region_list = region_info.get("RegionList") or []
    if isinstance(region_list, dict):
        region_list = [region_list]  # exiftool collapses a single-item list to a bare dict
    return region_list


def _region_bbox(region: dict) -> Optional[tuple[float, float, float, float]]:
    area = region.get("Area")
    if not isinstance(area, dict) or str(area.get("Unit", "normalized")).lower() != "normalized":
        return None
    try:
        cx, cy, w, h = float(area["X"]), float(area["Y"]), float(area["W"]), float(area["H"])
    except (KeyError, TypeError, ValueError):
        return None
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _merge_region_items(
    existing: list[dict], new_faces: list[tuple[str, float, float, float, float]], target: Path,
) -> list[str]:
    """Existing regions are kept as-is unless a new region's box overlaps
    them closely enough to be the same physical face, in which case the new
    one replaces it. This preserves anything else already in the file --
    regions from an earlier write-xmp run, or from another tool entirely --
    instead of overwriting the whole region list."""
    new_boxes = [(left, top, right, bottom) for _name, left, top, right, bottom in new_faces]
    kept_items = []
    unparseable = 0
    for region in existing:
        bbox = _region_bbox(region)
        name = region.get("Name")
        if bbox is None or not name:
            unparseable += 1
            continue
        if any(iou(bbox, new_box) >= REGION_MATCH_IOU for new_box in new_boxes):
            continue  # superseded by one of the new regions below
        kept_items.append(_region_struct_item(str(name), *bbox, region_type=str(region.get("Type") or "Face")))
    if unparseable:
        logger.warning(
            "%d existing region(s) in %s were in an unexpected format and could not "
            "be preserved -- they were dropped.", unparseable, target,
        )
    new_items = [_region_struct_item(name, left, top, right, bottom) for name, left, top, right, bottom in new_faces]
    return kept_items + new_items


def write_regions(
    image_path: Path,
    width: int,
    height: int,
    faces: list[tuple[str, float, float, float, float]],
    keep_backup: bool = True,
    dry_run: bool = False,
) -> Path:
    """Writes the MWG face-region list for one image, merged with whatever
    regions are already there. Returns the path actually written (the image
    itself, or its .xmp sidecar).

    For embed-capable formats this modifies the original file; exiftool
    keeps a `<file>_original` backup copy by default (`keep_backup=True`).
    For sidecars, exiftool creates the .xmp file from scratch if it doesn't
    already exist. Either way, existing regions are read first: one that
    spatially overlaps a face passed in here is updated in place, everything
    else already there (from an earlier run, or another tool) is kept as-is
    -- the target is never blindly overwritten. See `_merge_region_items`.
    """
    exiftool = require_exiftool()
    target, _is_sidecar = target_path_for(image_path)
    existing = _existing_region_list(exiftool, target)
    region_items = _merge_region_items(existing, faces, target)
    region_info = _build_region_info(width, height, region_items)

    args = [exiftool, "-struct"]
    if not keep_backup:
        args.append("-overwrite_original")
    args += [f"-XMP-mwg-rs:RegionInfo={region_info}", str(target)]

    if dry_run:
        logger.info("[dry-run] %s", " ".join(args))
        return target

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"exiftool failed for {target}: {result.stderr.strip()}")
    logger.debug(
        "Wrote %d region(s) to %s (%d new/updated, %d preserved from before)",
        len(region_items), target, len(faces), len(region_items) - len(faces),
    )
    return target
