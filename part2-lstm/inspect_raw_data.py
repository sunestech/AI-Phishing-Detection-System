"""Inspect the prepared raw-URL dataset before training the LSTM."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw_urls.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Run prepare_raw_url_data.py first."
    )

data = pd.read_csv(DATA_PATH)
print(f"File: {DATA_PATH.resolve()}")
print(f"Rows: {len(data):,}")
print(f"Columns: {list(data.columns)}")
print(f"Missing cells: {int(data.isna().sum().sum()):,}")
print(f"Duplicate URLs: {int(data.duplicated(subset='url').sum()):,}")
print("\nClass counts (0=Legitimate, 1=Phishing):")
print(data["label"].value_counts().sort_index().to_string())
print("\nURL length summary:")
print(data["url"].astype(str).str.len().describe().round(2).to_string())
print("\nExample rows:")
print(data.sample(min(5, len(data)), random_state=42).to_string(index=False))
