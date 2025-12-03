"""Neural network definition for DevAssist."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _build_encoder():
    if hasattr(tv_models, "vit_tiny_patch16_224"):
        model = tv_models.vit_tiny_patch16_224(weights=None)
        model.heads = nn.Identity()
        hidden_dim = model.hidden_dim
        return model, hidden_dim
    if hasattr(tv_models, "vit_b_16"):
        model = tv_models.vit_b_16(weights=None)
        model.heads = nn.Identity()
        hidden_dim = model.hidden_dim
        return model, hidden_dim
    # Fallback to ResNet if ViT unavailable
    backbone = tv_models.resnet18(weights=None)
    hidden_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, hidden_dim


@dataclass
class ModelConfig:
    action_dim: int = 4
    hidden_dim: int = 192
    text_vocab: int = 64
    text_max_len: int = 32


class CoordHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, max_len: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.project = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        repeated = self.project(x).unsqueeze(1).repeat(1, self.max_len, 1)
        logits = self.classifier(repeated)
        return logits


class DevAssistAgent(nn.Module):
    """Vision encoder + multi-head decoders for action planning."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder, hidden_dim = _build_encoder()
        self.coord_head = CoordHead(hidden_dim)
        self.action_head = ActionHead(hidden_dim, config.action_dim)
        self.text_head = TextHead(hidden_dim, config.text_vocab, config.text_max_len)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(images)
        coords = self.coord_head(features)
        action_logits = self.action_head(features)
        text_logits = self.text_head(features)
        return {
            "coords": coords,
            "action_logits": action_logits,
            "text_logits": text_logits,
        }
