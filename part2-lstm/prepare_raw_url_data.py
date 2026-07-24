"""Download and prepare a balanced raw-URL dataset for the LSTM.

Source: UCI PhiUSIIL Phishing URL (Website), dataset ID 967.
UCI's original labels are 1=legitimate and 0=phishing. This project deliberately
converts them to 0=legitimate and 1=phishing so a larger score always means more risk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42


def prepare_dataset(output_path: Path, per_class: int) -> None:
    if per_class < 100:
        raise ValueError("--per-class must be at least 100.")

    print("[1/5] Downloading UCI PhiUSIIL dataset (ID 967)...")
    repository_dataset = fetch_ucirepo(id=967)
    features = repository_dataset.data.features.reset_index(drop=True)
    targets = repository_dataset.data.targets.reset_index(drop=True)

    if "URL" not in features.columns:
        raise KeyError(f"Expected a URL column, but found: {list(features.columns)}")
    if targets.shape[1] < 1:
        raise ValueError("The UCI dataset did not return a target column.")

    original_label_column = targets.columns[0]
    data = pd.DataFrame(
        {
            "url": features["URL"].astype(str),
            "uci_label": pd.to_numeric(targets[original_label_column], errors="coerce"),
        }
    )

    print("[2/5] Cleaning blank, invalid, and duplicate URL rows...")
    data["url"] = data["url"].str.strip()
    data = data.dropna(subset=["uci_label"])
    data = data[data["uci_label"].isin([0, 1])]
    data = data[data["url"].str.len().between(4, 2048)]
    data = data.drop_duplicates(subset="url").reset_index(drop=True)

    # UCI: 1=legitimate, 0=phishing. Project: 0=legitimate, 1=phishing.
    data["label"] = 1 - data["uci_label"].astype(int)

    counts = data["label"].value_counts().sort_index()
    print("[3/5] Clean class counts (project convention):")
    print(f"      Legitimate (0): {int(counts.get(0, 0)):,}")
    print(f"      Phishing   (1): {int(counts.get(1, 0)):,}")

    available_per_class = int(min(counts.get(0, 0), counts.get(1, 0)))
    selected_per_class = min(per_class, available_per_class)
    if selected_per_class < per_class:
        print(
            f"      Requested {per_class:,} per class, but only "
            f"{available_per_class:,} balanced rows are available."
        )

    print(f"[4/5] Sampling {selected_per_class:,} rows from each class...")
    selected = []
    for label in (0, 1):
        class_rows = data[data["label"] == label].sample(
            n=selected_per_class,
            random_state=RANDOM_STATE + label,
        )
        selected.append(class_rows[["url", "label"]])

    prepared = pd.concat(selected, ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    )
    prepared = prepared.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    print("[5/5] Dataset preparation complete.")
    print(f"      Saved to: {output_path.resolve()}")
    print(f"      Total rows: {len(prepared):,}")
    print("      Columns: url, label")
    print("      Label meaning: 0=Legitimate, 1=Phishing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw_urls.csv"),
        help="Output CSV path (default: data/raw_urls.csv)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=10_000,
        help="Number of legitimate and phishing URLs to retain (default: 10000 each)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare_dataset(arguments.output, arguments.per_class)
