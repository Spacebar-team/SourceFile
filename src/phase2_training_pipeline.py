import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score

import xgboost as xgb
import shap
import mlflow


def load_features(
    app_path: Path,
    phase1_path: Path,
    time_column: Optional[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    app = pd.read_csv(app_path)
    phase1 = pd.read_csv(phase1_path)

    if "SK_ID_CURR" not in app.columns or "TARGET" not in app.columns:
        raise ValueError("application_train.csv must include SK_ID_CURR and TARGET")

    data = app.merge(phase1, on="SK_ID_CURR", how="left")

    # Keep numeric columns only to simplify modeling.
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    drop_cols = ["TARGET"]
    if time_column and time_column in numeric_cols:
        drop_cols.append(time_column)
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    X = data[numeric_cols].copy()
    y = data["TARGET"].astype(int)

    # Fill missing values with median per column.
    X = X.fillna(X.median(numeric_only=True))

    return X, y, data


def apply_temporal_split(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    time_column: Optional[str],
    time_cutoff: Optional[float],
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    if not time_column or time_column not in data.columns or time_cutoff is None:
        return X, y, None, None

    train_mask = data[time_column] <= time_cutoff
    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_holdout = X.loc[~train_mask].copy()
    y_holdout = y.loc[~train_mask].copy()
    return X_train, y_train, X_holdout, y_holdout


def compute_class_weight(y: pd.Series) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return neg / pos


def top3_shap_for_rows(
    model: xgb.Booster,
    X: pd.DataFrame,
    output_path: Path,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification with native Booster, shap_values might not be a list
    # but a single array of (rows, cols).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    rows = []
    feature_names = X.columns.to_list()

    for i in range(X.shape[0]):
        vals = shap_values[i]
        top_idx = np.argsort(np.abs(vals))[-3:][::-1]
        row = {
            "row_index": int(i),
            "feature_1": feature_names[top_idx[0]],
            "score_1": float(vals[top_idx[0]]),
            "feature_2": feature_names[top_idx[1]],
            "score_2": float(vals[top_idx[1]]),
            "feature_3": feature_names[top_idx[2]],
            "score_3": float(vals[top_idx[2]]),
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    holdout: Optional[Tuple[pd.DataFrame, pd.Series]],
    n_splits: int,
    random_state: int,
    shap_output: Path,
    model_output: Path,
    feature_output: Path,
) -> None:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scale_pos_weight = compute_class_weight(y)

    mlflow.set_experiment("home_credit_delinquency_risk")

    fold_metrics: List[Tuple[float, float]] = []

    with mlflow.start_run(run_name="xgboost_cv"):
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Native XGBoost API to avoid sklearn wrapper issues
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            params = {
                "max_depth": 6,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "scale_pos_weight": scale_pos_weight,
                "nthread": 4,
                "seed": random_state,
            }
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=300,
                evals=[(dtrain, "train"), (dval, "val")],
                verbose_eval=False
            )

            # Predict
            preds = model.predict(dval)
            auc = roc_auc_score(y_val, preds)
            recall = recall_score(y_val, (preds >= 0.5).astype(int))

            fold_metrics.append((auc, recall))
            mlflow.log_metric(f"fold_{fold}_auc", auc)
            mlflow.log_metric(f"fold_{fold}_recall", recall)

            # Save SHAP top-3 for this fold
            # Explainer works with model object from xgb.train too
            fold_shap_path = shap_output.with_name(
                f"{shap_output.stem}_fold{fold}{shap_output.suffix}"
            )
            # Adapt shap call for native booster? TreeExplainer supports it.
            # We need dataframe for shap to match feature names
            top3_shap_for_rows(model, X_val, fold_shap_path)

        avg_auc = float(np.mean([m[0] for m in fold_metrics]))
        avg_recall = float(np.mean([m[1] for m in fold_metrics]))

        mlflow.log_metric("avg_auc", avg_auc)
        mlflow.log_metric("avg_recall", avg_recall)

        # Final model on all data
        dall = xgb.DMatrix(X, label=y)
        final_model = xgb.train(
            params,
            dall,
            num_boost_round=300,
            verbose_eval=False
        )
        
        # Save model (native format)
        final_model.save_model(model_output)
        pd.Series(X.columns.to_list(), name="feature").to_csv(feature_output, index=False)

        if holdout is not None:
            X_holdout, y_holdout = holdout
            if len(y_holdout) > 0:
                dholdout = xgb.DMatrix(X_holdout, label=y_holdout)
                holdout_preds = final_model.predict(dholdout)
                holdout_auc = roc_auc_score(y_holdout, holdout_preds)
                holdout_recall = recall_score(y_holdout, (holdout_preds >= 0.5).astype(int))
                mlflow.log_metric("holdout_auc", holdout_auc)
                mlflow.log_metric("holdout_recall", holdout_recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 XGBoost training pipeline with MLflow")
    parser.add_argument(
        "--application-train",
        type=Path,
        default=Path("application_train.csv"),
        help="Path to application_train.csv",
    )
    parser.add_argument(
        "--phase1-features",
        type=Path,
        default=Path("phase1_behavioral_features.csv"),
        help="Path to Phase 1 feature CSV",
    )
    parser.add_argument(
        "--shap-output",
        type=Path,
        default=Path("phase2_shap_top3.csv"),
        help="Output CSV prefix for SHAP top-3 per prediction",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Optional time column in application_train.csv for temporal split",
    )
    parser.add_argument(
        "--time-cutoff",
        type=float,
        default=None,
        help=(
            "Optional cutoff value for temporal split (train <= cutoff). "
            "Example: -500 keeps earlier history for training and reserves a recent window "
            "for testing pre-delinquency logic."
        ),
    )
    parser.add_argument(
        "--leakage-report",
        type=Path,
        default=None,
        help="Optional leakage report CSV from Phase 1 to validate cutoffs",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/xgb_model.json"),
        help="Output path for the trained model artifact",
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        default=Path("artifacts/feature_list.csv"),
        help="Output path for the feature list",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, y, data = load_features(args.application_train, args.phase1_features, args.time_column)

    if args.time_cutoff is not None:
        if not args.time_column or args.time_column not in data.columns:
            raise ValueError("time_column must be provided and exist when using time_cutoff")
        if args.leakage_report is None:
            raise ValueError("leakage_report is required when using time_cutoff")

    if args.leakage_report is not None and args.time_cutoff is not None:
        leakage = pd.read_csv(args.leakage_report)
        if "SK_ID_CURR" in leakage.columns and "max_entry_payment_day" in leakage.columns:
            merged = data[["SK_ID_CURR"]].merge(leakage, on="SK_ID_CURR", how="left")
            leaked = merged[merged["max_entry_payment_day"] > args.time_cutoff]
            if not leaked.empty:
                raise ValueError("Leakage check failed: entries beyond cutoff detected")

    X_train, y_train, X_holdout, y_holdout = apply_temporal_split(
        data, X, y, args.time_column, args.time_cutoff
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.feature_output.parent.mkdir(parents=True, exist_ok=True)

    holdout = (X_holdout, y_holdout) if X_holdout is not None else None
    train_and_evaluate(
        X_train,
        y_train,
        holdout,
        args.n_splits,
        args.random_state,
        args.shap_output,
        args.model_output,
        args.feature_output,
    )


if __name__ == "__main__":
    main()
