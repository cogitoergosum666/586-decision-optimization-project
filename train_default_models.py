import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loan_default_data import TARGET
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLUMNS = ["issue_d"]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in {TARGET, *DROP_COLUMNS}]
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
    return preprocessor, numeric_cols, categorical_cols


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "issue_year" not in df.columns:
        raise ValueError("Expected issue_year column. Run prepare_default_model_data.py first.")

    df = df.dropna(subset=["issue_year"]).copy()
    train = df[df["issue_year"] <= 2015]
    test = df[df["issue_year"].between(2016, 2017)]

    if len(train) == 0 or len(test) == 0:
        shuffled = df.sample(frac=1, random_state=42)
        split_at = int(len(shuffled) * 0.8)
        train = shuffled.iloc[:split_at]
        test = shuffled.iloc[split_at:]

    return train, test


def evaluate(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    proba = model.predict_proba(x_test)[:, 1]
    eps = np.finfo(float).eps
    proba = np.clip(proba, eps, 1 - eps)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
    }


def maybe_xgboost_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train default probability candidate models.")
    parser.add_argument("--data", default="model_data/default_model_data.csv")
    parser.add_argument("--sample", type=int, default=250_000)
    args = parser.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42)

    train, test = time_split(df)
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    x_train = train.drop(columns=[TARGET])
    x_test = test.drop(columns=[TARGET])

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(train)
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    xgb = maybe_xgboost_classifier()
    if xgb is not None:
        candidates["xgboost"] = xgb

    results = []
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Numeric features: {len(numeric_cols)}; categorical features: {len(categorical_cols)}")

    for name, estimator in candidates.items():
        model = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        model.fit(x_train, y_train)
        results.append(evaluate(name, model, x_test, y_test))

    result_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if "xgboost" not in candidates:
        print("xgboost is not installed, so the XGBoost candidate was skipped.")


if __name__ == "__main__":
    main()
