#!/usr/bin/env python3
"""Execute the learned policy inside a real Vercel session."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.runner import AutomationRunner, RunnerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Vercel deployment flow")
    parser.add_argument("checkpoint", type=Path, help="Path to trained checkpoint")
    parser.add_argument("repo", help="Repository to deploy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--chrome", action="store_true", help="Use the locally installed Google Chrome instead of bundled Chromium")
    parser.add_argument("--maximize", action="store_true", help="Launch headed browser maximized")
    parser.add_argument("--fullscreen", action="store_true", help="Launch headed browser in fullscreen (Chromium Browser.setWindowBounds)")
    parser.add_argument("--viewport-width", type=int, default=1920)
    parser.add_argument("--viewport-height", type=int, default=1080)
    parser.add_argument("--storage-state", type=Path)
    parser.add_argument("--debug", action="store_true", help="Print policy decisions each step")
    parser.add_argument("--save-screenshots", type=Path, dest="screenshot_dir", help="Directory to store inference frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunnerConfig(
        checkpoint=args.checkpoint,
        repo_name=args.repo,
        headless=args.headless,
        use_chrome=args.chrome,
        start_maximized=args.maximize,
        fullscreen=args.fullscreen,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        storage_state=args.storage_state,
        debug=args.debug,
        screenshot_dir=args.screenshot_dir,
    )
    runner = AutomationRunner(cfg)
    runner.execute()


if __name__ == "__main__":
    main()
