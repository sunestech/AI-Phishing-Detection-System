"""Train, evaluate, and save a character-level PyTorch LSTM URL classifier."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_utils import URLDataset, build_character_vocabulary
from model import URLLSTMClassifier

RANDOM_STATE = 42
LABEL_NAMES = ["Legitimate", "Phishing"]


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # This favors repeatability over maximum speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_validate_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run prepare_raw_url_data.py before training."
        )

    data = pd.read_csv(path)
    required_columns = {"url", "label"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(
            f"The LSTM CSV must contain url and label columns. Missing: {sorted(missing)}"
        )

    data = data[["url", "label"]].dropna().copy()
    data["url"] = data["url"].astype(str).str.strip()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data = data[data["label"].isin([0, 1])]
    data["label"] = data["label"].astype(int)
    data = data[data["url"].str.len() > 0]
    data = data.drop_duplicates(subset="url").reset_index(drop=True)

    if data["label"].nunique() != 2:
        raise ValueError("The dataset must contain both label 0 and label 1.")

    return data


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_validation, test = train_test_split(
        data,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    validation_fraction_of_remaining = 0.15 / 0.85
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction_of_remaining,
        random_state=RANDOM_STATE,
        stratify=train_validation["label"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def make_loader(
    frame: pd.DataFrame,
    character_to_index: Dict[str, int],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = URLDataset(
        frame["url"].tolist(),
        frame["label"].tolist(),
        character_to_index,
        max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Reliable default for Windows and beginner projects.
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for token_ids, lengths, labels in loader:
        token_ids = token_ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(token_ids, lengths)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        total_loss += loss.item() * labels.size(0)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().long().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    return float(average_loss), float(accuracy)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []

    with torch.no_grad():
        for token_ids, lengths, batch_labels in loader:
            token_ids = token_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = model(token_ids, lengths)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().astype(int).tolist())

    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], LABEL_NAMES)
    axis.set_yticks([0, 1], LABEL_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Part 2 LSTM Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_training_history(history: list[dict], csv_path: Path, image_path: Path) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame["epoch"], frame["train_loss"], label="Training loss")
    axis.plot(frame["epoch"], frame["validation_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("LSTM Training History")
    axis.legend()
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    set_reproducible_seeds(RANDOM_STATE)
    output_directory = args.output_dir
    artifact_directory = args.artifact_dir
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading and validating raw URL data...")
    data = load_and_validate_data(args.data)
    train_frame, validation_frame, test_frame = split_data(data)
    print(f"      Total rows:      {len(data):,}")
    print(f"      Training rows:   {len(train_frame):,}")
    print(f"      Validation rows: {len(validation_frame):,}")
    print(f"      Test rows:       {len(test_frame):,}")

    print("[2/7] Building the character vocabulary from training URLs only...")
    character_to_index = build_character_vocabulary(train_frame["url"])
    print(f"      Vocabulary size: {len(character_to_index):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3/7] Computing device: {device}")

    train_loader = make_loader(
        train_frame,
        character_to_index,
        args.max_length,
        args.batch_size,
        True,
        device,
    )
    validation_loader = make_loader(
        validation_frame,
        character_to_index,
        args.max_length,
        args.batch_size,
        False,
        device,
    )
    test_loader = make_loader(
        test_frame,
        character_to_index,
        args.max_length,
        args.batch_size,
        False,
        device,
    )

    model_config = {
        "vocab_size": len(character_to_index),
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": 1,
        "dropout": args.dropout,
        "bidirectional": True,
        "padding_index": 0,
    }
    model = URLLSTMClassifier(**model_config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print("[4/7] Training the character-level LSTM...")
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"      Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.4f} | val acc {validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"      Early stopping: validation loss did not improve for "
                    f"{args.patience} epochs."
                )
                break

    if best_state is None:
        raise RuntimeError("Training ended without producing a model state.")
    model.load_state_dict(best_state)

    print("[5/7] Evaluating the best model on untouched test URLs...")
    true_labels, probabilities = collect_predictions(model, test_loader, device)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision_phishing": float(
            precision_score(true_labels, predictions, pos_label=1, zero_division=0)
        ),
        "recall_phishing": float(
            recall_score(true_labels, predictions, pos_label=1, zero_division=0)
        ),
        "f1_phishing": float(
            f1_score(true_labels, predictions, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(true_labels, probabilities)),
        "threshold": 0.5,
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "vocabulary_size": int(len(character_to_index)),
        "max_length": int(args.max_length),
        "epochs_completed": int(len(history)),
        "device": str(device),
    }
    report = classification_report(
        true_labels,
        predictions,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])

    print(f"      Test accuracy:     {metrics['accuracy']:.4%}")
    print(f"      Phishing precision:{metrics['precision_phishing']:9.4%}")
    print(f"      Phishing recall:   {metrics['recall_phishing']:9.4%}")
    print(f"      Phishing F1-score: {metrics['f1_phishing']:9.4%}")
    print(f"      ROC-AUC:           {metrics['roc_auc']:.4f}")
    print("\n" + report)

    print("[6/7] Saving the PyTorch model and character vocabulary...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "character_to_index": character_to_index,
        "max_length": args.max_length,
        "threshold": 0.5,
        "label_mapping": {0: "Legitimate", 1: "Phishing"},
        "metrics": metrics,
        "normalization": "strip and lowercase",
        "long_url_strategy": "keep first half and last half",
    }
    model_path = artifact_directory / "lstm_model.pt"
    torch.save(checkpoint, model_path)
    with (artifact_directory / "character_vocabulary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(character_to_index, file, indent=2, ensure_ascii=False)

    print("[7/7] Saving reports...")
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    (output_directory / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    save_confusion_matrix(matrix, output_directory / "confusion_matrix.png")
    save_training_history(
        history,
        output_directory / "training_history.csv",
        output_directory / "training_history.png",
    )

    prediction_frame = test_frame.copy()
    prediction_frame["phishing_probability"] = probabilities
    prediction_frame["prediction"] = predictions
    prediction_frame["correct"] = (
        prediction_frame["prediction"] == prediction_frame["label"]
    )
    prediction_frame.to_csv(output_directory / "test_predictions.csv", index=False)

    print(f"      Model:   {model_path.resolve()}")
    print(f"      Reports: {output_directory.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw_urls.csv"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
