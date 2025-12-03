"""Loss helpers for multi-task supervision."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def coordinate_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def action_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def text_loss(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    flat_logits = logits.view(-1, vocab)
    flat_target = tokens.view(-1)
    flat_mask = mask.view(-1)
    loss = F.cross_entropy(flat_logits, flat_target, reduction="none")
    masked = loss * flat_mask.float()
    denom = flat_mask.float().sum().clamp(min=1.0)
    return masked.sum() / denom


def total_loss(outputs, batch) -> torch.Tensor:
    loss_coords = coordinate_loss(outputs["coords"], batch["coords"])
    loss_action = action_loss(outputs["action_logits"], batch["action_label"])
    loss_text = text_loss(outputs["text_logits"], batch["text_tokens"], batch["text_mask"])
    return loss_coords + loss_action + loss_text
