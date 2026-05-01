from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}

TARGET = "target_default"

USE_COLUMNS = [
    "loan_amnt",
    "term",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "application_type",
    "issue_d",
    "loan_status",
]


def parse_issue_year(value: object) -> float:
    if not isinstance(value, str) or "-" not in value:
        return np.nan
    try:
        return float(value.split("-")[-1])
    except ValueError:
        return np.nan


def load_terminal_loans(input_path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    terminal_statuses = GOOD_STATUSES | BAD_STATUSES

    reader = pd.read_csv(
        input_path,
        usecols=USE_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )

    for i, chunk in enumerate(reader, start=1):
        chunk = chunk[chunk["loan_status"].isin(terminal_statuses)].copy()
        if chunk.empty:
            continue

        chunk[TARGET] = chunk["loan_status"].isin(BAD_STATUSES).astype(int)
        chunk["issue_year"] = chunk["issue_d"].map(parse_issue_year)
        chunk = chunk.drop(columns=["loan_status", "issue_d"])
        chunks.append(chunk)
        print(f"Loaded chunk {i}: kept {len(chunk):,} terminal rows")

    if not chunks:
        raise ValueError("No terminal loan rows found.")

    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["issue_year"]).copy()
    df["issue_year"] = df["issue_year"].astype(int)
    return df


def time_split_default(
    df: pd.DataFrame,
    train_end_year: int = 2015,
    validation_year: int = 2016,
    test_year: int = 2017,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["issue_year"] <= train_end_year].copy()
    valid = df[df["issue_year"] == validation_year].copy()
    test = df[df["issue_year"] == test_year].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError("Expected non-empty train, validation, and test splits.")

    return train, valid, test

