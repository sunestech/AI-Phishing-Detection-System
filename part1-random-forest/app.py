"""Local Flask verifier for the trained Part 1 Random Forest model.

This application deliberately tests rows from the same held-out 20% split used by
train_advanced.py. The supplied dataset contains 48 pre-engineered numeric features,
so this Part 1 app does not accept a raw URL string.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "phishing.csv"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
TARGET_COLUMN = "CLASS_LABEL"
ID_COLUMN = "id"
RANDOM_STATE = 42

app = Flask(__name__)


def load_artifact() -> tuple[Any, list[str], dict[int, str]]:
    """Load either the corrected artifact dictionary or a bare sklearn model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Run: python train_advanced.py"
        )

    with MODEL_PATH.open("rb") as file:
        saved_object = pickle.load(file)

    if isinstance(saved_object, dict) and "model" in saved_object:
        model = saved_object["model"]
        feature_names = list(saved_object.get("feature_names", []))
        raw_mapping = saved_object.get(
            "label_mapping", {0: "Legitimate", 1: "Phishing"}
        )
        label_mapping = {int(key): str(value) for key, value in raw_mapping.items()}
    else:
        model = saved_object
        feature_names = []
        label_mapping = {0: "Legitimate", 1: "Phishing"}

    return model, feature_names, label_mapping


def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


model, saved_feature_names, label_mapping = load_artifact()

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing {DATA_PATH.name} in {BASE_DIR}")

data = pd.read_csv(DATA_PATH)

if TARGET_COLUMN not in data.columns:
    raise ValueError(
        f"{DATA_PATH.name} must contain a {TARGET_COLUMN!r} column. "
        f"Found: {data.columns.tolist()}"
    )

feature_names = saved_feature_names or [
    column
    for column in data.columns
    if column not in {TARGET_COLUMN, ID_COLUMN}
]

missing_features = [name for name in feature_names if name not in data.columns]
if missing_features:
    raise ValueError(
        "The CSV is missing features expected by model.pkl: "
        + ", ".join(missing_features)
    )

# Re-create the exact deterministic test split used by train_advanced.py.
X = data[feature_names]
y = data[TARGET_COLUMN].astype(int)
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

holdout = data.loc[X_test.index].copy()
holdout["_actual_label"] = y_test

if ID_COLUMN not in holdout.columns:
    holdout[ID_COLUMN] = holdout.index

# Provide a balanced, manageable menu: 20 legitimate + 20 phishing holdout rows.
legitimate_examples = holdout[holdout["_actual_label"] == 0].head(20)
phishing_examples = holdout[holdout["_actual_label"] == 1].head(20)
sample_rows = (
    pd.concat([legitimate_examples, phishing_examples])
    .sort_values(ID_COLUMN)
    .reset_index(drop=True)
)

feature_importances = getattr(model, "feature_importances_", None)
if feature_importances is not None and len(feature_importances) == len(feature_names):
    top_feature_names = [
        name
        for name, _ in sorted(
            zip(feature_names, feature_importances),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]
else:
    top_feature_names = feature_names[:10]

metrics = load_metrics()


def find_holdout_row(sample_id_text: str) -> pd.Series:
    """Find one held-out record using its displayed ID."""
    matches = holdout[holdout[ID_COLUMN].astype(str) == sample_id_text]
    if matches.empty:
        raise ValueError("Choose one of the held-out record IDs shown in the list.")
    return matches.iloc[0]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    sample_options = [str(value) for value in sample_rows[ID_COLUMN].tolist()]
    selected_id = sample_options[0] if sample_options else ""

    if request.method == "POST":
        selected_id = request.form.get("sample_id", "").strip()
        try:
            row = find_holdout_row(selected_id)
            feature_frame = pd.DataFrame(
                [[row[name] for name in feature_names]],
                columns=feature_names,
            )

            predicted_number = int(model.predict(feature_frame)[0])
            actual_number = int(row["_actual_label"])

            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(feature_frame)[0, 1])

            result = {
                "record_id": selected_id,
                "actual_number": actual_number,
                "actual_text": label_mapping.get(actual_number, str(actual_number)),
                "predicted_number": predicted_number,
                "predicted_text": label_mapping.get(
                    predicted_number, str(predicted_number)
                ),
                "probability": probability,
                "correct": actual_number == predicted_number,
                "features": [
                    {"name": name, "value": row[name]}
                    for name in top_feature_names
                ],
            }
        except (TypeError, ValueError, KeyError) as exc:
            error = str(exc)

    accuracy = metrics.get("test_accuracy")

    return render_template(
        "index.html",
        sample_options=sample_options,
        selected_id=selected_id,
        result=result,
        error=error,
        accuracy=accuracy,
        feature_count=len(feature_names),
        holdout_count=len(holdout),
    )


if __name__ == "__main__":
    print("Part 1 verifier is ready at http://127.0.0.1:5000")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )
