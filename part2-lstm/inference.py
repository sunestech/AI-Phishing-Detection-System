"""Reusable model-loading and single-URL prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from data_utils import encode_url
from model import URLLSTMClassifier


def load_predictor(model_path: Path) -> Tuple[URLLSTMClassifier, Dict[str, Any], torch.device]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train the LSTM before running predictions."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(model_path, map_location=device)

    model = URLLSTMClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


def predict_url(
    url: str,
    model: URLLSTMClassifier,
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    encoded, length = encode_url(
        url,
        checkpoint["character_to_index"],
        int(checkpoint["max_length"]),
    )
    token_ids = encoded.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)

    with torch.no_grad():
        logit = model(token_ids, lengths)
        probability = float(torch.sigmoid(logit).item())

    threshold = float(checkpoint.get("threshold", 0.5))
    predicted_label = int(probability >= threshold)
    label_mapping = checkpoint.get(
        "label_mapping", {0: "Legitimate", 1: "Phishing"}
    )
    # JSON/pickle may preserve integer keys, but this also tolerates string keys.
    label_name = label_mapping.get(
        predicted_label, label_mapping.get(str(predicted_label), str(predicted_label))
    )

    return {
        "url": url,
        "predicted_label": predicted_label,
        "prediction": label_name,
        "phishing_probability": probability,
        "legitimate_probability": 1.0 - probability,
        "threshold": threshold,
        "characters_analyzed": length,
    }
