from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loan_default_data import TARGET, load_terminal_loans, time_split_default
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_pipeline(df: pd.DataFrame, c_value: float, l1_ratio: float, max_iter: int) -> Pipeline:
    feature_cols = [c for c in df.columns if c != TARGET]
    categorical_cols = [
        c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(name: str, y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    eps = np.finfo(float).eps
    proba = np.clip(proba, eps, 1 - eps)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "split": name,
        "rows": int(len(y_true)),
        "bad_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def append_results(
    output_path: Path,
    input_path: Path,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    results: list[dict[str, object]],
    c_value: float,
    l1_ratio: float,
    max_iter: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not output_path.exists():
        output_path.write_text("# Loan Default Modeling Results\n\n", encoding="utf-8")

    metric_cols = [
        "split",
        "rows",
        "bad_rate",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    confusion_cols = ["split", "threshold", "tn", "fp", "fn", "tp"]

    section = f"""
## Elastic Net Logistic Regression Full Run

Run time: {now}

Input file: `{input_path}`

Model configuration:

- `penalty`: elasticnet
- `solver`: saga
- `C`: {c_value}
- `l1_ratio`: {l1_ratio}
- `class_weight`: balanced
- `max_iter`: {max_iter}

Feature policy:

- Uses the shared cleaning, target, feature list, and time split from `loan_default_data.py`.
- Uses origination-time borrower, loan, credit-profile, application, and issue-year fields.
- Excludes post-origination repayment, recovery, hardship, settlement, last-payment, and last-FICO fields.
- Excludes `grade`, `sub_grade`, and `int_rate` to avoid using LendingClub's own risk/pricing output.

Time split:

| Split | Issue years | Rows | Bad rate |
| --- | --- | ---: | ---: |
| Train | <= 2015 | {len(train):,} | {train[TARGET].mean():.4f} |
| Validation | 2016 | {len(valid):,} | {valid[TARGET].mean():.4f} |
| Test | 2017 | {len(test):,} | {test[TARGET].mean():.4f} |

Metrics:

{markdown_table(results, metric_cols)}

Confusion matrix at threshold 0.5:

{markdown_table(results, confusion_cols)}

Raw metrics JSON:

```json
{json.dumps(results, indent=2)}
```

"""
    with output_path.open("a", encoding="utf-8") as f:
        f.write(section)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-data elastic net logistic regression.")
    parser.add_argument("--input", default="accepted_2007_to_2018Q4.csv")
    parser.add_argument("--results-md", default="loan_default_model_results.md")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    results_path = Path(args.results_md)

    df = load_terminal_loans(input_path, args.chunksize)
    train, valid, test = time_split_default(df)

    pipeline = build_pipeline(train, c_value=args.c, l1_ratio=args.l1_ratio, max_iter=args.max_iter)

    x_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].astype(int)
    pipeline.fit(x_train, y_train)

    results = []
    for name, split in [("validation_2016", valid), ("test_2017", test)]:
        x_split = split.drop(columns=[TARGET])
        y_split = split[TARGET].astype(int)
        proba = pipeline.predict_proba(x_split)[:, 1]
        results.append(evaluate(name, y_split, proba))

    append_results(
        results_path,
        input_path,
        train,
        valid,
        test,
        results,
        c_value=args.c,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
    )
    print(f"Appended elastic net logistic regression results to {results_path}")


if __name__ == "__main__":
    main()

