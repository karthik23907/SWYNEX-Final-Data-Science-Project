"""Reproducible data-preparation pipeline for the IBM Telco Customer Churn dataset.

Usage (from the repository root):
    python -m src.data_prep              # raw -> validated clean CSV
    python -m src.data_prep --download   # fetch the raw file first if it is missing

Every cleaning decision is implemented as a small, documented step and recorded in
an audit log so the numbers in the report always match what the code actually did.
"""
from __future__ import annotations

import argparse
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("data_prep")

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "telco_churn_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# ---- schema knowledge (post snake_case renaming) -------------------------------------
YES_NO_COLUMNS = ["partner", "dependents", "phone_service", "paperless_billing", "churn"]
ADDON_COLUMNS = [
    "multiple_lines", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]
# These values only restate information held in `phone_service` / `internet_service`.
REDUNDANT_LEVELS = {"No phone service": "No", "No internet service": "No"}
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
CATEGORICAL_COLUMNS = ["gender", "internet_service", "payment_method"]


@dataclass
class AuditLog:
    """Ordered record of every transformation applied (feeds the quality report)."""

    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, detail: str, rows_affected: int = 0) -> None:
        self.steps.append({"step": step, "detail": detail, "rows_affected": int(rows_affected)})
        log.info("%s | %s | rows affected: %d", step, detail, rows_affected)


# ---- I/O -----------------------------------------------------------------------------
def download_raw(path: Path = RAW_PATH, url: str = SOURCE_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading raw data from %s", url)
    urllib.request.urlretrieve(url, path)  # noqa: S310 (trusted, fixed URL)
    return path


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Re-run with --download.")
    return pd.read_csv(path)


# ---- cleaning ------------------------------------------------------------------------
def to_snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _yes_no_to_int(series: pd.Series) -> pd.Series:
    out = series.map({"Yes": 1, "No": 0})
    if out.isna().any():
        bad = sorted(series[out.isna()].unique())
        raise ValueError(f"Unexpected values in '{series.name}': {bad}")
    return out.astype("int8")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, AuditLog]:
    """Return (clean_dataframe, audit_log). The input frame is never modified."""
    audit = AuditLog()
    df = raw.copy()

    # 1. Consistent column names -------------------------------------------------------
    df.columns = [to_snake(c) for c in df.columns]
    audit.add("Rename columns", "CamelCase -> snake_case (e.g. customerID -> customer_id)")

    # 2. Whitespace --------------------------------------------------------------------
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    changed = 0
    for c in text_cols:
        stripped = df[c].str.strip()
        changed += int((stripped != df[c]).sum())
        df[c] = stripped
    audit.add("Strip whitespace", f"Applied to {len(text_cols)} text columns", changed)

    # 3. total_charges: blank strings hide as text, forcing the column to object --------
    blank = df["total_charges"].eq("")
    df["total_charges"] = pd.to_numeric(df["total_charges"].where(~blank), errors="raise")
    audit.add("Fix type: total_charges", "Blank strings -> NaN, column cast to float64", blank.sum())

    # 4. Missing values: only legitimate when the customer has no billing history ------
    missing = df["total_charges"].isna()
    if (df.loc[missing, "tenure"] != 0).any():
        raise ValueError("total_charges is missing for customers with tenure > 0; needs review.")
    df.loc[missing, "total_charges"] = 0.0
    audit.add(
        "Impute: total_charges",
        "All blanks belong to tenure = 0 customers (no bill issued yet) -> set to 0.0",
        missing.sum(),
    )

    # 5. Redundant category levels -----------------------------------------------------
    replaced = 0
    for c in ADDON_COLUMNS:
        replaced += int(df[c].isin(REDUNDANT_LEVELS).sum())
        df[c] = df[c].replace(REDUNDANT_LEVELS)
    audit.add(
        "Collapse redundant levels",
        "'No internet service' / 'No phone service' -> 'No' in 7 add-on columns "
        "(already encoded by internet_service / phone_service)",
        replaced,
    )

    # 6. Types -------------------------------------------------------------------------
    for c in YES_NO_COLUMNS + ADDON_COLUMNS:
        df[c] = _yes_no_to_int(df[c])
    audit.add("Encode Yes/No", f"{len(YES_NO_COLUMNS) + len(ADDON_COLUMNS)} columns -> int8 (1 = Yes)")

    df["senior_citizen"] = df["senior_citizen"].astype("int8")
    df["tenure"] = df["tenure"].astype("int16")
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].astype("category")
    df["contract"] = pd.Categorical(df["contract"], categories=CONTRACT_ORDER, ordered=True)
    audit.add(
        "Set categorical dtypes",
        f"{', '.join(CATEGORICAL_COLUMNS)} -> category; contract -> ordered category",
    )

    return df, audit


# ---- validation ----------------------------------------------------------------------
def run_checks(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Return a list of (check name, passed, detail)."""
    binary = YES_NO_COLUMNS + ADDON_COLUMNS + ["senior_citizen"]
    checks = [
        ("customer_id is unique", df["customer_id"].is_unique, f"{df['customer_id'].nunique()} unique ids"),
        ("No missing values", int(df.isna().sum().sum()) == 0, f"{int(df.isna().sum().sum())} nulls"),
        ("No duplicate rows", int(df.duplicated().sum()) == 0, f"{int(df.duplicated().sum())} duplicates"),
        ("tenure >= 0", bool((df["tenure"] >= 0).all()), f"range {df['tenure'].min()}-{df['tenure'].max()} months"),
        ("monthly_charges > 0", bool((df["monthly_charges"] > 0).all()),
         f"range {df['monthly_charges'].min():.2f}-{df['monthly_charges'].max():.2f}"),
        ("total_charges >= 0", bool((df["total_charges"] >= 0).all()), f"min {df['total_charges'].min():.2f}"),
        ("Binary columns contain only 0/1", all(set(df[c].unique()) <= {0, 1} for c in binary), f"{len(binary)} columns"),
        ("Contract levels as expected", set(df["contract"].unique()) == set(CONTRACT_ORDER), ", ".join(CONTRACT_ORDER)),
        ("tenure = 0 <=> total_charges = 0", bool(((df["tenure"] == 0) == (df["total_charges"] == 0)).all()),
         "billing history consistent with tenure"),
    ]
    return checks


def validate(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results = run_checks(df)
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        raise AssertionError(f"Validation failed: {failed}")
    log.info("All %d validation checks passed", len(results))
    return results


# ---- entry point ---------------------------------------------------------------------
def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--out", type=Path, default=CLEAN_PATH)
    parser.add_argument("--download", action="store_true", help="download the raw file if missing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if args.download and not args.raw.exists():
        download_raw(args.raw)

    raw = load_raw(args.raw)
    log.info("Loaded raw data: %d rows x %d columns", *raw.shape)
    df, _ = clean(raw)
    validate(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("Saved clean data -> %s  (%d rows x %d columns)", args.out.relative_to(ROOT), *df.shape)
    return df


if __name__ == "__main__":
    main()
