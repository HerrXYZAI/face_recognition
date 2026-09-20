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

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

EMBED_EXTENSIONS = {".dng", ".jpg", ".jpeg", ".tif", ".tiff", ".psd", ".heic"}
SIDECAR_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".pef"}

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


def _build_region_info(
    width: int, height: int, faces: list[tuple[str, float, float, float, float]]
) -> str:
    """faces: list of (person_name, left, top, right, bottom), all fractional
    0..1. MWG regions use a CENTER x/y + width/height, normalized 0..1 --
    distinct from Lightroom's own top/left/bottom/right internal storage."""
    region_items = []
    for name, left, top, right, bottom in faces:
        cx, cy = (left + right) / 2, (top + bottom) / 2
        w, h = right - left, bottom - top
        region_items.append(
            "{Area={X=%s,Y=%s,W=%s,H=%s,Unit=normalized},Name=%s,Type=Face}"
            % (cx, cy, w, h, _escape(name))
        )
    region_list = "[" + ",".join(region_items) + "]"
    return (
        "{AppliedToDimensions={W=%d,H=%d,Unit=pixel},RegionList=%s}"
        % (width, height, region_list)
    )


def write_regions(
    image_path: Path,
    width: int,
    height: int,
    faces: list[tuple[str, float, float, float, float]],
    keep_backup: bool = True,
    dry_run: bool = False,
) -> Path:
    """Writes/replaces the MWG face-region list for one image. Returns the
    path actually written (the image itself, or its .xmp sidecar).

    For embed-capable formats this modifies the original file; exiftool
    keeps a `<file>_original` backup copy by default (`keep_backup=True`).
    For sidecars, exiftool creates the .xmp file from scratch if it doesn't
    already exist, or merges into it (preserving other tags) if it does.
    """
    exiftool = require_exiftool()
    target, _is_sidecar = target_path_for(image_path)
    region_info = _build_region_info(width, height, faces)

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
    logger.debug("Wrote %d region(s) to %s", len(faces), target)
    return target
