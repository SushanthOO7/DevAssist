#!/usr/bin/env python3
"""CLI for collecting a single Vercel trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path

from screeninfo import get_monitors

from src.data_pipeline.capture import CaptureConfig, capture_vercel_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one automation run on Vercel")
    parser.add_argument("repo", help="GitHub repository name to search")
    parser.add_argument("output", type=Path, help="Directory to store frames and log")
    parser.add_argument("--storage-state", type=Path, dest="storage_state", help="Playwright login state JSON")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--monitor-index", type=int, default=0, help="Monitor index for viewport auto-detect")
    parser.add_argument("--viewport-width", type=int, default=1920)
    parser.add_argument("--viewport-height", type=int, default=1080)
    parser.add_argument("--use-system-viewport", action="store_true", help="Let Playwright control viewport via window size")
    return parser.parse_args()


def _resolve_viewport(args: argparse.Namespace) -> tuple[int, int]:
    if args.headless:
        return args.viewport_width, args.viewport_height
    try:
        monitors = get_monitors()
        if monitors:
            idx = max(0, min(args.monitor_index, len(monitors) - 1))
            monitor = monitors[idx]
            return monitor.width, monitor.height
    except Exception:
        pass
    return args.viewport_width, args.viewport_height


def main() -> None:
    args = parse_args()
    viewport_w, viewport_h = _resolve_viewport(args)
    config = CaptureConfig(
        repo_name=args.repo,
        output_dir=args.output,
        storage_state=args.storage_state,
        headless=args.headless,
        viewport_width=viewport_w,
        viewport_height=viewport_h,
        use_system_viewport=args.use_system_viewport,
    )
    log_path = capture_vercel_flow(config)
    print(f"Saved capture to {log_path}")


if __name__ == "__main__":
    main()
