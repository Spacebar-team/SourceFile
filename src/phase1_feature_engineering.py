import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage."""
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float", "float64", "float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Days past due (DPD): positive values mean late payments.
    df["days_past_due"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)

    # Use entry payment date as a proxy for time ordering (more recent is closer to 0).
    df = df.sort_values(["SK_ID_CURR", "DAYS_ENTRY_PAYMENT"])  # ascending: older -> newer

    # Aggregate base stats.
    grouped = df.groupby("SK_ID_CURR", as_index=False)
    summary = grouped["days_past_due"].agg(
        avg_days_past_due="mean",
        dpd_std="std",
        dpd_mean="mean",
        dpd_count="count",
    )

    # Coefficient of variation with guard for low/zero mean.
    summary["payment_consistency_score"] = (
        summary["dpd_std"].fillna(0) / summary["dpd_mean"].replace(0, np.nan)
    ).fillna(0)

    # Late payment trend over last 6 months (approx 6 * 30 days).
    recent = df[df["DAYS_ENTRY_PAYMENT"] >= -180].copy()
    if recent.empty:
        trend = pd.DataFrame({"SK_ID_CURR": summary["SK_ID_CURR"], "late_payment_trend": 0})
    else:
        def slope_for_group(group: pd.DataFrame) -> float:
            if len(group) < 2:
                return 0.0
            x = group["DAYS_ENTRY_PAYMENT"].to_numpy(dtype=float)
            y = group["days_past_due"].to_numpy(dtype=float)
            x = x - x.mean()
            denom = (x ** 2).sum()
            if denom == 0:
                return 0.0
            return float((x * (y - y.mean())).sum() / denom)

        trend = recent.groupby("SK_ID_CURR").apply(slope_for_group).reset_index()
        trend.columns = ["SK_ID_CURR", "late_payment_trend"]

    features = summary[["SK_ID_CURR", "avg_days_past_due", "payment_consistency_score"]].merge(
        trend, on="SK_ID_CURR", how="left"
    )
    features["late_payment_trend"] = features["late_payment_trend"].fillna(0)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 behavioral features from installments_payments.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("installments_payments.csv"),
        help="Path to installments_payments.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase1_behavioral_features.csv"),
        help="Output feature CSV path",
    )
    parser.add_argument(
        "--max-entry-day",
        type=int,
        default=None,
        help="Optional cutoff for DAYS_ENTRY_PAYMENT to prevent leakage (e.g., -30)",
    )
    parser.add_argument(
        "--leakage-report",
        type=Path,
        default=None,
        help="Optional output CSV for leakage checks (max entry day per customer)",
    )
    args = parser.parse_args()

    usecols = [
        "SK_ID_CURR",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
    ]

    dtypes = {
        "SK_ID_CURR": "int32",
        "DAYS_INSTALMENT": "float32",
        "DAYS_ENTRY_PAYMENT": "float32",
    }

    df = pd.read_csv(
        args.input,
        usecols=usecols,
        dtype=dtypes,
    )
    df = optimize_dtypes(df)

    if args.max_entry_day is not None:
        df = df[df["DAYS_ENTRY_PAYMENT"] <= args.max_entry_day]

    features = compute_features(df)
    features.to_csv(args.output, index=False)

    if args.leakage_report is not None:
        leakage = (
            df.groupby("SK_ID_CURR")["DAYS_ENTRY_PAYMENT"]
            .max()
            .reset_index()
            .rename(columns={"DAYS_ENTRY_PAYMENT": "max_entry_payment_day"})
        )
        leakage.to_csv(args.leakage_report, index=False)


if __name__ == "__main__":
    main()
