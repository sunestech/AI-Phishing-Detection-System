"""Data preparation helpers for the character-level URL LSTM."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_INDEX = 0
UNK_INDEX = 1


def normalize_url(url: object) -> str:
    """Convert a URL value into the consistent text form used by the model."""
    return str(url).strip().lower()


def build_character_vocabulary(urls: Iterable[object], min_frequency: int = 1) -> Dict[str, int]:
    """Build a deterministic character-to-integer lookup from training URLs only."""
    counts: Counter[str] = Counter()
    for url in urls:
        counts.update(normalize_url(url))

    character_to_index: Dict[str, int] = {
        PAD_TOKEN: PAD_INDEX,
        UNK_TOKEN: UNK_INDEX,
    }

    # Frequency-first ordering is deterministic and puts common characters near the front.
    ordered = sorted(
        ((character, count) for character, count in counts.items() if count >= min_frequency),
        key=lambda item: (-item[1], item[0]),
    )
    for character, _ in ordered:
        character_to_index[character] = len(character_to_index)

    return character_to_index


def encode_url(
    url: object,
    character_to_index: Dict[str, int],
    max_length: int,
) -> Tuple[torch.Tensor, int]:
    """Convert one URL into a right-padded tensor and return its true length.

    Very long URLs keep both their beginning and ending. The beginning preserves the
    scheme/domain; the ending preserves late path/query patterns.
    """
    text = normalize_url(url)
    if not text:
        text = UNK_TOKEN

    if len(text) > max_length:
        head_length = max_length // 2
        tail_length = max_length - head_length
        text = text[:head_length] + text[-tail_length:]

    encoded = [character_to_index.get(character, UNK_INDEX) for character in text]
    true_length = max(1, len(encoded))
    encoded.extend([PAD_INDEX] * (max_length - len(encoded)))

    return torch.tensor(encoded, dtype=torch.long), true_length


class URLDataset(Dataset):
    """PyTorch dataset returning encoded URL, sequence length, and binary label."""

    def __init__(
        self,
        urls: Iterable[object],
        labels: Iterable[int],
        character_to_index: Dict[str, int],
        max_length: int,
    ) -> None:
        self.urls = list(urls)
        self.labels = [int(label) for label in labels]
        self.character_to_index = character_to_index
        self.max_length = max_length

        if len(self.urls) != len(self.labels):
            raise ValueError("The URL and label collections must have the same length.")

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, index: int):
        encoded, true_length = encode_url(
            self.urls[index], self.character_to_index, self.max_length
        )
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return encoded, torch.tensor(true_length, dtype=torch.long), label
