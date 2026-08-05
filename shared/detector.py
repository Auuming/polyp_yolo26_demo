from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


_INFERENCE_LOCK = threading.RLock()


def default_device() -> int | str:
    return 0 if torch.cuda.is_available() else "cpu"


def find_model(*directories: Path) -> Path:
    for directory in directories:
        for candidate in (directory / "best.pt", directory / "models" / "best.pt"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("models/best.pt was not found.")


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
