"""Shared image loading helpers used by every pipeline step.

Images are always loaded EXIF-orientation-corrected (so a phone photo shot in
portrait and stored with an EXIF rotation tag comes out right-side-up as a
pixel array), and every fractional bounding box in this project -- whether
read from Lightroom's catalog or produced by the recognition model -- is
treated as relative to that corrected orientation. Keeping this consistent
everywhere is what makes Lightroom's face boxes line up with the model's.

This assumption (Lightroom's stored face fractions are relative to the
EXIF-corrected/displayed orientation, not the raw stored pixel grid) matches
how Lightroom displays and edits face regions, but hasn't been verified
against Adobe source. Use `face-pipeline verify-crops` to sanity-check it
against your own catalog before trusting a full training run.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_bgr(path: Path | str) -> np.ndarray:
    """Loads an image as an EXIF-orientation-corrected BGR uint8 array
    (OpenCV's native channel order)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        rgb = np.array(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fractional_to_pixel_bbox(
    left: float, top: float, right: float, bottom: float, width: int, height: int
) -> tuple[float, float, float, float]:
    return (left * width, top * height, right * width, bottom * height)


def pixel_to_fractional_bbox(
    left: float, top: float, right: float, bottom: float, width: int, height: int
) -> tuple[float, float, float, float]:
    return (left / width, top / height, right / width, bottom / height)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two (left, top, right, bottom) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
