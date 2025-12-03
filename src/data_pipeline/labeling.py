"""Utilities that transform captured Playwright traces into training labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class Annotation:
    screenshot: Path
    action_type: str
    coords: np.ndarray
    text: str | None

    def as_dict(self) -> Dict:
        return {
            "screenshot": str(self.screenshot),
            "action_type": self.action_type,
            "coords": self.coords.tolist(),
            "text": self.text,
        }


def _normalize_box(box: Dict[str, float], width: int, height: int) -> np.ndarray:
    cx = (box["x"] + box["width"] / 2.0) / width
    cy = (box["y"] + box["height"] / 2.0) / height
    return np.array([cx, cy], dtype=np.float32)


def build_annotations_from_capture(
    capture_dir: Path, *, viewport_width: int, viewport_height: int
) -> List[Annotation]:
    log_path = capture_dir / "trajectory.json"
    with log_path.open("r", encoding="utf-8") as fp:
        trajectory = json.load(fp)
    annotations: List[Annotation] = []
    for record in trajectory:
        box = record.get("bounding_box")
        if not box:
            continue
        coords = _normalize_box(box, viewport_width, viewport_height)
        annotations.append(
            Annotation(
                screenshot=capture_dir / record["screenshot"],
                action_type=record["action_type"],
                coords=coords,
                text=record.get("typed_text"),
            )
        )
    return annotations


def merge_annotation_runs(
    capture_roots: Sequence[Path], *, viewport_width: int, viewport_height: int, output_path: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged: List[Dict] = []
    for root in capture_roots:
        merged.extend(
            [a.as_dict() for a in build_annotations_from_capture(root, viewport_width=viewport_width, viewport_height=viewport_height)]
        )
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(merged, fp, indent=2)
    return output_path
