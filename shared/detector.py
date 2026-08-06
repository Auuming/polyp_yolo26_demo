from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


_INFERENCE_LOCK = threading.RLock()

MODEL_OPTIONS = {
    "V1 - positives only (baseline)": "best.pt",
    "V2 - positives + 400 negatives": "best_v2.pt",
}
DEFAULT_MODEL = next(iter(MODEL_OPTIONS))


def default_device() -> int | str:
    return 0 if torch.cuda.is_available() else "cpu"


def find_model(*directories: Path, filename: str = "best.pt") -> Path:
    for directory in directories:
        for candidate in (directory / filename, directory / "models" / filename):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"models/{filename} was not found.")


def find_models(*directories: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for display_name, filename in MODEL_OPTIONS.items():
        try:
            found[display_name] = find_model(*directories, filename=filename)
        except FileNotFoundError:
            continue
    if not found:
        expected = ", ".join(MODEL_OPTIONS.values())
        raise FileNotFoundError(f"No model weights were found. Expected: {expected}.")
    return found


def load_model(model_path: Path | str) -> YOLO:
    return YOLO(str(model_path))


def predict(
    model: YOLO,
    source: Any,
    confidence: float,
    device: int | str | None = None,
):
    with _INFERENCE_LOCK:
        return model.predict(
            source=source,
            conf=confidence,
            imgsz=640,
            device=default_device() if device is None else device,
            verbose=False,
        )[0]
