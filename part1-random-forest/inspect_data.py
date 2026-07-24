"""Inspect the supplied phishing dataset without modifying it."""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "phishing.csv"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    data = pd.read_csv(CSV_PATH)

    print(f"File: {CSV_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {data.shape[1]}")
    print(f"Missing cells: {int(data.isna().sum().sum()):,}")
    print(f"Duplicate rows: {int(data.duplicated().sum()):,}")
    print(f"First five columns: {data.columns[:5].tolist()}")
    print(f"Last column: {data.columns[-1]}")

    if "CLASS_LABEL" in data.columns:
        print("\nCLASS_LABEL counts:")
        print(data["CLASS_LABEL"].value_counts().sort_index())
    else:
        print("\nWARNING: CLASS_LABEL was not found.")
        print("Available columns:")
        print(data.columns.tolist())


if __name__ == "__main__":
    main()
