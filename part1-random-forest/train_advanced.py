
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "phishing.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
REPORTS_DIR = BASE_DIR / "reports"
TARGET_COLUMN = "CLASS_LABEL"
ID_COLUMN = "id"
RANDOM_STATE = 42


def main() -> None:
    print("[1/5] Loading phishing.csv...")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    data = pd.read_csv(CSV_PATH)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Expected a target column named {TARGET_COLUMN!r}. "
            f"Found: {data.columns.tolist()}"
        )

    X = data.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = data[TARGET_COLUMN].astype(int)

    if X.isna().any().any():
        raise ValueError("The feature columns contain missing values.")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("CLASS_LABEL must contain only 0 and 1.")

    print(f"      Rows: {len(data):,}")
    print(f"      Features used: {X.shape[1]}")
    print(f"      Labels: {y.value_counts().sort_index().to_dict()}")

    print("[2/5] Splitting data: 80% training, 20% testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("[3/5] Tuning the Random Forest with 5-fold cross-validation...")
    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    parameter_grid = {
        "n_estimators": [150, 300],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1],
    }

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cross_validation,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("[4/5] Testing the best model on data it did not train on...")
    predictions = best_model.predict(X_test)
    phishing_probabilities = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=1)
    recall = recall_score(y_test, predictions, pos_label=1)
    f1 = f1_score(y_test, predictions, pos_label=1)
    roc_auc = roc_auc_score(y_test, phishing_probabilities)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Legitimate (0)", "Phishing (1)"],
        digits=4,
    )

    print(f"\nBest parameters: {search.best_params_}")
    print(f"Cross-validation accuracy: {search.best_score_:.4%}")
    print(f"Test accuracy: {accuracy:.4%}")
    print(f"Phishing precision: {precision:.4%}")
    print(f"Phishing recall: {recall:.4%}")
    print(f"Phishing F1-score: {f1:.4%}")
    print(f"ROC-AUC: {roc_auc:.4f}\n")
    print(report)

    print("[5/5] Saving the model and reports...")
    REPORTS_DIR.mkdir(exist_ok=True)

    artifact = {
        "model": best_model,
        "feature_names": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "label_mapping": {0: "Legitimate", 1: "Phishing"},
    }
    with MODEL_PATH.open("wb") as file:
        pickle.dump(artifact, file)

    metrics = {
        "best_parameters": search.best_params_,
        "cross_validation_accuracy": float(search.best_score_),
        "test_accuracy": float(accuracy),
        "phishing_precision": float(precision),
        "phishing_recall": float(recall),
        "phishing_f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (REPORTS_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions, labels=[0, 1]),
        display_labels=["Legitimate", "Phishing"],
    ).plot(values_format="d")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
