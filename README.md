# AI-Based Phishing Detection System

This repository presents two complementary approaches to phishing detection:

1. **Part 1 — Random Forest:** classifies websites from 48 pre-engineered numerical URL and webpage features.
2. **Part 2 — Character-Level Bi-LSTM:** classifies raw URL strings by learning character-sequence patterns with PyTorch.

Both implementations include data validation, model training, evaluation reports, model serialization, command-line testing, and local Flask web interfaces.

## Results at a glance

| Metric | Part 1: Random Forest | Part 2: Bi-LSTM |
|---|---:|---:|
| Dataset size | 10,000 records | 20,000 URLs |
| Test samples | 2,000 | 3,000 |
| Accuracy | **98.65%** | **99.70%** |
| Phishing precision | **98.80%** | **99.87%** |
| Phishing recall | **98.50%** | **99.53%** |
| Phishing F1-score | **98.65%** | **99.70%** |
| ROC-AUC | **0.9990** | **0.9990** |
| False positives | 12 | 2 |
| False negatives | 15 | 7 |

> These results came from different datasets and input representations, so they should not be interpreted as a perfectly controlled head-to-head comparison.

## Repository structure

```text
AI-Phishing-Detection-System/
├── README.md
├── .gitignore
├── part1-random-forest/
│   ├── README.md
│   ├── app.py
│   ├── inspect_data.py
│   ├── train_advanced.py
│   ├── requirements.txt
│   ├── templates/
│   └── reports/
└── part2-lstm/
    ├── README.md
    ├── app.py
    ├── prepare_raw_url_data.py
    ├── inspect_raw_data.py
    ├── data_utils.py
    ├── model.py
    ├── train_lstm.py
    ├── inference.py
    ├── predict_url.py
    ├── requirements.txt
    ├── templates/
    └── reports/
```

## Part 1 visual result

![Random Forest confusion matrix](part1-random-forest/reports/confusion_matrix.png)

## Part 2 visual results

![LSTM confusion matrix](part2-lstm/reports/confusion_matrix.png)

![LSTM training history](part2-lstm/reports/training_history.png)

## Reproducibility

Each part has its own `requirements.txt`, setup instructions, training command, and deployment instructions. Generated virtual environments, datasets, and trained model binaries are intentionally excluded from version control.

## Responsible-use notice

This is an educational detection system, not a replacement for browser protections, threat-intelligence services, sandboxing, or professional security controls. A high held-out test score does not guarantee correct predictions for every future or external URL.
