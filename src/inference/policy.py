"""Policy module that maps screenshots to UI actions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from src.training.model import DevAssistAgent, ModelConfig
from src.data_pipeline.dataset import TextTokenizer


@dataclass
class PolicyOutput:
    action: str
    coords: tuple[float, float]
    text: str | None


class PixelPolicy:
    def __init__(self, checkpoint: Path, action_vocab: list[str]) -> None:
        self.action_vocab = action_vocab
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = TextTokenizer()
        vocab_size = len(self.tokenizer.alphabet) + 1
        config = ModelConfig(action_dim=len(action_vocab), hidden_dim=192, text_vocab=vocab_size, text_max_len=self.tokenizer.max_length)
        self.model = DevAssistAgent(config)
        state = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.model.to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def act(self, screenshot: Path) -> PolicyOutput:
        image = Image.open(screenshot).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        action_idx = outputs["action_logits"].argmax(dim=-1).item()
        coords = outputs["coords"].squeeze(0).cpu().tolist()
        text_tokens = outputs["text_logits"].argmax(dim=-1).squeeze(0).cpu().tolist()
        decoded_text = self._decode_text(text_tokens)
        return PolicyOutput(action=self.action_vocab[action_idx], coords=(coords[0], coords[1]), text=decoded_text)

    def _decode_text(self, tokens: list[int]) -> str | None:
        inv = {idx + 1: ch for idx, ch in enumerate(self.tokenizer.alphabet)}
        chars = [inv.get(idx, "") for idx in tokens if idx != 0]
        text = "".join(chars).strip()
        return text or None
