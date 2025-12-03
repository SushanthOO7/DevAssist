"""Training loop orchestration."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data_pipeline.dataset import TrajectoryDataset
from src.training.losses import total_loss
from src.training.model import DevAssistAgent, ModelConfig


@dataclass
class TrainerConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, manifest: Path, config: TrainerConfig) -> None:
        dataset = TrajectoryDataset(manifest)
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        self.train_set, self.val_set = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size)
        model_cfg = ModelConfig(action_dim=4, hidden_dim=192, text_vocab=len(dataset.tokenizer.alphabet) + 1)
        self.model = DevAssistAgent(model_cfg).to(config.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = math.ceil(len(self.train_loader) * config.epochs)
        self.scheduler = CosineAnnealingLR(self.optimizer, total_steps)
        self.cfg = config
        self.best_val = float("inf")
        self.ckpt_dir = Path("artifacts/checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> Path:
        best_ckpt = self.ckpt_dir / "best.pt"
        for epoch in range(self.cfg.epochs):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            if val_loss < self.best_val:
                self.best_val = val_loss
                torch.save({"model": self.model.state_dict()}, best_ckpt)
            print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")
        return best_ckpt

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> float:
        stage = "train" if train else "val"
        running = 0.0
        steps = 0
        self.model.train(train)
        for batch in tqdm(loader, desc=stage):
            batch = {k: v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(batch["image"])
            loss = total_loss(outputs, batch)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
            running += loss.item()
            steps += 1
        return running / max(steps, 1)
