import argparse
from pathlib import Path

import pandas as pd
from loan_default_data import BAD_STATUSES, GOOD_STATUSES, TARGET, USE_COLUMNS, parse_issue_year


RAW_DEFAULT_INPUT = "accepted_2007_to_2018Q4.csv"


def prepare(input_path: Path, output_path: Path, chunksize: int, max_rows: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    written = 0

    reader = pd.read_csv(
        input_path,
        usecols=USE_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in reader:
        chunk = chunk[chunk["loan_status"].isin(GOOD_STATUSES | BAD_STATUSES)].copy()
        if chunk.empty:
            continue

        chunk[TARGET] = chunk["loan_status"].isin(BAD_STATUSES).astype(int)
        chunk["issue_year"] = chunk["issue_d"].map(parse_issue_year)
        chunk = chunk.drop(columns=["loan_status"])

        if max_rows is not None:
            remaining = max_rows - written
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        chunk.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        written += len(chunk)

        if max_rows is not None and written >= max_rows:
            break

    print(f"Wrote {written:,} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe LendingClub default modeling data.")
    parser.add_argument("--input", default=RAW_DEFAULT_INPUT)
    parser.add_argument("--output", default="model_data/default_model_data.csv")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    prepare(Path(args.input), Path(args.output), args.chunksize, args.max_rows)


if __name__ == "__main__":
    main()
