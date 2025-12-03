#!/usr/bin/env python3
"""Visualize captured bounding boxes on screenshots for sanity checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay capture annotations on screenshots")
    parser.add_argument("capture_dir", type=Path, help="Directory containing trajectory.json and screenshots")
    parser.add_argument("--index", type=int, default=-1, help="Entry index to visualize (default all)")
    parser.add_argument("--output", type=Path, default=Path("overlay.png"))
    return parser.parse_args()


def overlay(capture_dir: Path, index: int, dest: Path) -> None:
    log = capture_dir / "trajectory.json"
    with log.open("r", encoding="utf-8") as fp:
        records = json.load(fp)
    if index >= 0:
        records = [records[index]]
    layers = []
    for record in records:
        img_path = capture_dir / record["screenshot"]
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        box = record.get("bounding_box")
        if box:
            x0, y0 = box["x"], box["y"]
            x1, y1 = x0 + box["width"], y0 + box["height"]
            draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
            draw.text((x0 + 6, y0 + 6), record["step"], fill="yellow")
        layers.append(image)
    if not layers:
        raise ValueError("No records found to visualize")
    result = layers[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(layers) == 1:
        result.save(dest)
        print(f"Saved overlay to {dest}")
        return
    for idx, layer in enumerate(layers, start=0):
        target = dest.with_name(f"{dest.stem}_{idx}{dest.suffix}")
        layer.save(target)
        print(f"Saved overlay to {target}")


def main() -> None:
    args = parse_args()
    overlay(args.capture_dir, args.index, args.output)


if __name__ == "__main__":
    main()
