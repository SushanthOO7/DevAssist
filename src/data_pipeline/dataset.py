"""Dataset objects for supervising DevAssist."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


DEFAULT_ACTION_VOCAB = ["click", "scroll", "type", "wait"]


@dataclass
class TextTokenizer:
    alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789-_/.:" " \n"
    max_length: int = 32

    def __post_init__(self) -> None:
        self.char_to_idx = {ch: idx + 1 for idx, ch in enumerate(self.alphabet)}
        self.pad_idx = 0

    def encode(self, text: str | None) -> torch.LongTensor:
        tokens = [self.char_to_idx.get(ch.lower(), self.pad_idx) for ch in (text or "")]
        tokens = tokens[: self.max_length]
        padded = tokens + [self.pad_idx] * (self.max_length - len(tokens))
        return torch.LongTensor(padded)

    def mask(self, tokens: torch.LongTensor) -> torch.BoolTensor:
        return tokens != self.pad_idx


class TrajectoryDataset(Dataset):
    """Loads screenshot-action pairs for supervised training."""

    def __init__(
        self,
        manifest: Path,
        *,
        image_size: int = 224,
        action_vocab: Sequence[str] = DEFAULT_ACTION_VOCAB,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        self.samples = self._load_manifest(manifest)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.tokenizer = tokenizer or TextTokenizer()
        self.action_to_idx = {name: idx for idx, name in enumerate(action_vocab)}

    def _load_manifest(self, manifest: Path) -> List[Dict]:
        with manifest.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        image = Image.open(sample["screenshot"]).convert("RGB")
        image_tensor = self.transform(image)
        coords = torch.tensor(sample["coords"], dtype=torch.float32)
        action_idx = torch.tensor(self.action_to_idx[sample["action_type"]], dtype=torch.long)
        text_tokens = self.tokenizer.encode(sample.get("text"))
        text_mask = self.tokenizer.mask(text_tokens)
        return {
            "image": image_tensor,
            "coords": coords,
            "action_label": action_idx,
            "text_tokens": text_tokens,
            "text_mask": text_mask,
        }
