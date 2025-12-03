#!/usr/bin/env python3
"""Train DevAssist on collected trajectories."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DevAssist agent")
    parser.add_argument("manifest", type=Path, help="Path to merged annotations JSON")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainerConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    trainer = Trainer(args.manifest, cfg)
    ckpt = trainer.fit()
    print(f"Best checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()
