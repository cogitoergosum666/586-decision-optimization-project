from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loan_default_data import BAD_STATUSES, GOOD_STATUSES, TARGET, USE_COLUMNS, parse_issue_year
from run_xgboost_full import build_model, build_preprocessor


EXPORT_ID_COLUMNS = [
    "id",
    "loan_amnt",
    "term",
    "purpose",
    "addr_state",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "application_type",
    "issue_d",
    "loan_status",
]


def load_terminal_loans_with_id(input_path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    terminal_statuses = GOOD_STATUSES | BAD_STATUSES
    usecols = list(dict.fromkeys(["id", *USE_COLUMNS]))

    reader = pd.read_csv(
        input_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    for i, chunk in enumerate(reader, start=1):
        chunk = chunk[chunk["loan_status"].isin(terminal_statuses)].copy()
        if chunk.empty:
            continue

        chunk[TARGET] = chunk["loan_status"].isin(BAD_STATUSES).astype(int)
        chunk["issue_year"] = chunk["issue_d"].map(parse_issue_year)
        chunks.append(chunk)
        print(f"Loaded chunk {i}: kept {len(chunk):,} terminal rows")

    if not chunks:
        raise ValueError("No terminal loan rows found.")

    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["issue_year"]).copy()
    df["issue_year"] = df["issue_year"].astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export XGBoost default probabilities for the 2017 test set.")
    parser.add_argument("--input", default="accepted_2007_to_2018Q4.csv")
    parser.add_argument("--output", default="model_outputs/xgboost_test_2017_scores.csv")
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_terminal_loans_with_id(input_path, args.chunksize)

    train = df[df["issue_year"] <= 2015].copy()
    test = df[df["issue_year"] == 2017].copy()
    if train.empty or test.empty:
        raise ValueError("Expected non-empty training and 2017 test splits.")

    x_train = train.drop(columns=[TARGET, "id", "loan_status", "issue_d"])
    y_train = train[TARGET].astype(int)
    x_test = test.drop(columns=[TARGET, "id", "loan_status", "issue_d"])

    preprocessor, _, _ = build_preprocessor(train.drop(columns=["id", "loan_status", "issue_d"]))
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)

    model = build_model(args, y_train)
    model.fit(x_train_t, y_train)

    test = test.copy()
    test["pred_default_prob"] = model.predict_proba(x_test_t)[:, 1]

    export_columns = [
        "id",
        "issue_year",
        "loan_status",
        TARGET,
        "pred_default_prob",
        "loan_amnt",
        "term",
        "purpose",
        "addr_state",
        "annual_inc",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "application_type",
    ]
    test[export_columns].to_csv(output_path, index=False)
    print(f"Wrote {len(test):,} rows to {output_path}")


if __name__ == "__main__":
    main()

