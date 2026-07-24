"""Classify one URL string from the command line without visiting the website."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference import load_predictor, predict_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="The URL text to analyze")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/lstm_model.pt"),
        help="Saved PyTorch checkpoint path",
    )
    args = parser.parse_args()

    model, checkpoint, device = load_predictor(args.model)
    result = predict_url(args.url, model, checkpoint, device)

    print(f"URL: {result['url']}")
    print(f"Prediction: {result['prediction']} ({result['predicted_label']})")
    print(f"Phishing probability: {result['phishing_probability']:.2%}")
    print(f"Decision threshold: {result['threshold']:.2f}")
    print("Safety note: the URL text was analyzed locally; the website was not opened.")


if __name__ == "__main__":
    main()
