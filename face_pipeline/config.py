"""Loads config.yaml (paths, model settings) shared by every pipeline step."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("FACE_PIPELINE_CONFIG", "config.yaml"))


@dataclass
class RecognitionConfig:
    model_name: str = "buffalo_l"
    ctx_id: int = 0
    detector_size: tuple[int, int] = (640, 640)
    min_confidence: float = 0.55


@dataclass
class Config:
    catalog_copy_path: Path
    images_root: Path
    labeled_faces_csv: Path
    faces_db: Path
    classifier_path: Path
    recognition: RecognitionConfig
    xmp_min_confidence: float
    raw: dict

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}. Copy config.example.yaml to "
                f"{path.name} and fill in your paths, or set FACE_PIPELINE_CONFIG."
            )
        raw = yaml.safe_load(path.read_text()) or {}

        lightroom = raw.get("lightroom", {})
        images = raw.get("images", {})
        data = raw.get("data", {})
        recognition = raw.get("recognition", {})
        xmp = raw.get("xmp", {})

        rec_cfg = RecognitionConfig(
            model_name=recognition.get("model_name", "buffalo_l"),
            ctx_id=int(recognition.get("ctx_id", 0)),
            detector_size=tuple(recognition.get("detector_size", [640, 640])),
            min_confidence=float(recognition.get("min_confidence", 0.55)),
        )

        return cls(
            catalog_copy_path=Path(lightroom["catalog_copy_path"]),
            images_root=Path(images["root"]),
            labeled_faces_csv=Path(data.get("labeled_faces_csv", "data/labeled_faces.csv")),
            faces_db=Path(data.get("faces_db", "data/faces.db")),
            classifier_path=Path(data.get("classifier_path", "models/classifier.pkl")),
            recognition=rec_cfg,
            xmp_min_confidence=float(xmp.get("min_confidence", 0.6)),
            raw=raw,
        )
