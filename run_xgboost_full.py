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
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c != TARGET]
    categorical_cols = [
        c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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
    return preprocessor, numeric_cols, categorical_cols


def build_model(args: argparse.Namespace, y_train: pd.Series) -> XGBClassifier:
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )


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
    args: argparse.Namespace,
    scale_pos_weight: float,
    feature_counts: dict[str, int],
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
## XGBoost Full Run

Run time: {now}

Input file: `{input_path}`

Model configuration:

- `n_estimators`: {args.n_estimators}
- `max_depth`: {args.max_depth}
- `learning_rate`: {args.learning_rate}
- `subsample`: {args.subsample}
- `colsample_bytree`: {args.colsample_bytree}
- `min_child_weight`: {args.min_child_weight}
- `reg_lambda`: {args.reg_lambda}
- `tree_method`: hist
- `scale_pos_weight`: {scale_pos_weight:.4f}

Feature policy:

- Uses the shared cleaning, target, feature list, and time split from `loan_default_data.py`.
- Uses origination-time borrower, loan, credit-profile, application, and issue-year fields.
- Excludes post-origination repayment, recovery, hardship, settlement, last-payment, and last-FICO fields.
- Excludes `grade`, `sub_grade`, and `int_rate` to avoid using LendingClub's own risk/pricing output.

Feature preprocessing:

- Numeric features: median imputation.
- Categorical features: most-frequent imputation plus one-hot encoding with rare-category grouping.
- Raw numeric columns: {feature_counts["numeric_raw"]}
- Raw categorical columns: {feature_counts["categorical_raw"]}
- Transformed model columns: {feature_counts["transformed"]}

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
    parser = argparse.ArgumentParser(description="Run full-data XGBoost for LendingClub default risk.")
    parser.add_argument("--input", default="accepted_2007_to_2018Q4.csv")
    parser.add_argument("--results-md", default="loan_default_model_results.md")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=10.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    results_path = Path(args.results_md)

    df = load_terminal_loans(input_path, args.chunksize)
    train, valid, test = time_split_default(df)

    x_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].astype(int)
    x_valid = valid.drop(columns=[TARGET])
    y_valid = valid[TARGET].astype(int)
    x_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].astype(int)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(train)
    x_train_t = preprocessor.fit_transform(x_train)
    x_valid_t = preprocessor.transform(x_valid)
    x_test_t = preprocessor.transform(x_test)

    model = build_model(args, y_train)
    model.fit(x_train_t, y_train)

    results = []
    for name, x_split, y_split in [
        ("validation_2016", x_valid_t, y_valid),
        ("test_2017", x_test_t, y_test),
    ]:
        proba = model.predict_proba(x_split)[:, 1]
        results.append(evaluate(name, y_split, proba))

    feature_counts = {
        "numeric_raw": len(numeric_cols),
        "categorical_raw": len(categorical_cols),
        "transformed": int(x_train_t.shape[1]),
    }
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    append_results(results_path, input_path, train, valid, test, results, args, scale_pos_weight, feature_counts)
    print(f"Appended XGBoost results to {results_path}")


if __name__ == "__main__":
    main()

