

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    models: Path
    reports: Path
    figures: Path


def get_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    data_raw = root / "data" / "raw"
    models = root / "models"
    reports = root / "reports"
    figures = reports / "figures"


    for p in [data_raw, models, reports, figures]:
        p.mkdir(parents=True, exist_ok=True)

    return Paths(root=root, data_raw=data_raw, models=models, reports=reports, figures=figures)



def safe_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
  
  
    df.columns = (
        df.columns.astype(str)
        .str.replace("\u00A0", " ", regex=False)  
        .str.replace("\ufeff", "", regex=False)  
        .str.strip()
    )

    rename_map = {
        "Temperature (°C)": "temperature_c",
        "Rainfall (mm)": "rainfall_mm",
        "Humidity (%)": "humidity_pct",
        "Crop Yield Impact Score": "crop_yield_impact_score",

        "vegetable_Commodity": "vegetable_commodity",
        "vegetable_Price per Unit (LKR/kg)": "price_lkr_per_kg",

  
        "vegitable_Commodity": "vegetable_commodity",
        "vegitable_Price per Unit (LKR/kg)": "price_lkr_per_kg",

        "Region": "region",
        "Date": "date",
    }


    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def validate_required_columns(df: pd.DataFrame) -> None:

    required = [
        "date",
        "region",
        "temperature_c",
        "rainfall_mm",
        "humidity_pct",
        "crop_yield_impact_score",
        "vegetable_commodity",
        "price_lkr_per_kg",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_date(df: pd.DataFrame) -> pd.DataFrame:

    if "date" not in df.columns:
        raise ValueError("Expected a 'Date' column (renamed to 'date').")


    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)


    if df["date"].isna().mean() > 0.2:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", unit="D", origin="1899-12-30")

    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise ValueError(f"Found {bad} rows with invalid dates after parsing. Fix them before training.")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter.astype(int)


    df["season"] = np.where(df["month"].isin([10, 11, 12, 1, 2, 3]), "maha", "yala")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.drop_duplicates().copy()


    df["region"] = df["region"].astype(str).str.strip()
    df["vegetable_commodity"] = df["vegetable_commodity"].astype(str).str.strip()
    df["season"] = df["season"].astype(str)

    numeric_cols = ["temperature_c", "rainfall_mm", "humidity_pct", "crop_yield_impact_score", "price_lkr_per_kg"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


    df = df.dropna(subset=["price_lkr_per_kg", "region", "vegetable_commodity", "date"])
    df = df.dropna(subset=["temperature_c", "rainfall_mm", "humidity_pct", "crop_yield_impact_score"])

    return df


def build_climate_baselines(df: pd.DataFrame) -> dict:
    
    cols = [
        "region",
        "month",
        "temperature_c",
        "rainfall_mm",
        "humidity_pct",
        "crop_yield_impact_score",
    ]

    tmp = df[cols].dropna().copy()
    tmp["region"] = tmp["region"].astype(str).str.strip()
    tmp["month"] = tmp["month"].astype(int)

    grouped = (
        tmp.groupby(["region", "month"], as_index=False)[
            ["temperature_c", "rainfall_mm", "humidity_pct", "crop_yield_impact_score"]
        ]
        .mean()
    )

    out = {}
    for _, r in grouped.iterrows():
        region = str(r["region"])
        month = int(r["month"])
        out.setdefault(region, {})
        out[region][str(month)] = {
            "temperature_c": float(r["temperature_c"]),
            "rainfall_mm": float(r["rainfall_mm"]),
            "humidity_pct": float(r["humidity_pct"]),
            "crop_yield_impact_score": float(r["crop_yield_impact_score"]),
        }

    return out


def time_split(df: pd.DataFrame):
    
    train = df[df["year"].between(2020, 2022)].copy()
    val = df[df["year"] == 2023].copy()
    test = df[df["year"] == 2024].copy()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Time split failed. Rows => train:{len(train)} val:{len(val)} test:{len(test)}. "
            "Check year coverage in your dataset."
        )

    return train, val, test


def build_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
    
    feature_cols = [
        "region",
        "vegetable_commodity",
        "season",
        "temperature_c",
        "rainfall_mm",
        "humidity_pct",
        "crop_yield_impact_score",
        "month",
        "weekofyear",
        "quarter",
        "year",
    ]

    X = df[feature_cols].copy()
    y = df["price_lkr_per_kg"].copy()

    
    cat_cols = ["region", "vegetable_commodity", "season"]
    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    return X, y, cat_indices


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.xlabel("Actual Price (LKR/kg)")
    plt.ylabel("Predicted Price (LKR/kg)")
    plt.title(title)

   
    minv = float(min(y_true.min(), y_pred.min()))
    maxv = float(max(y_true.max(), y_pred.max()))
    plt.plot([minv, maxv], [minv, maxv])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        required=True,
        help='Excel filename inside data/raw/ (e.g., "Vegetables_prices_with_climate_130000_2020_to_2025.xlsx")',
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = get_paths()
    xlsx_path = paths.data_raw / args.file

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {xlsx_path}\n"
            f"Put your .xlsx into: {paths.data_raw}\n"
            f"Example: data/raw/{args.file}"
        )

  
    df = pd.read_excel(xlsx_path)
    print("RAW COLUMNS FROM EXCEL:", list(df.columns))

    df = safe_rename_columns(df)
    print("COLUMNS AFTER RENAME:", list(df.columns))

    validate_required_columns(df)
    df = parse_date(df)
    df = add_time_features(df)
    df = basic_cleaning(df)

  
    climate_baselines = build_climate_baselines(df)
    baseline_out = paths.models / "climate_baselines.json"
    with open(baseline_out, "w", encoding="utf-8") as f:
        json.dump(climate_baselines, f, indent=2)
    print(" Saved climate baselines:", baseline_out)

   
    train_df, val_df, test_df = time_split(df)

    X_train, y_train, cat_idx = build_features_targets(train_df)
    X_val, y_val, _ = build_features_targets(val_df)
    X_test, y_test, _ = build_features_targets(test_df)

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=args.seed,
        depth=8,
        learning_rate=0.08,
        iterations=2000,
        l2_leaf_reg=3.0,
        subsample=0.8,
        rsm=0.9,
        eval_metric="RMSE",
        early_stopping_rounds=100,
        verbose=200,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

   
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics = {
        "dataset": os.path.basename(str(xlsx_path)),
        "rows_total_after_cleaning": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "metrics_val": eval_metrics(y_val.values, pred_val),
        "metrics_test": eval_metrics(y_test.values, pred_test),
        "feature_columns": list(X_train.columns),
        "categorical_features": ["region", "vegetable_commodity", "season"],
        "best_iteration": int(getattr(model, "best_iteration_", model.tree_count_)),
    }

   
    model_out = paths.models / "veg_price_model.joblib"
    joblib.dump(
        {"model": model, "cat_feature_indices": cat_idx, "feature_columns": list(X_train.columns)},
        model_out,
    )

   
    metrics_out = paths.reports / "metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    
    plot_path = paths.figures / "actual_vs_pred_test.png"
    plot_actual_vs_pred(y_test.values, pred_test, plot_path, title="Actual vs Predicted (Test Set)")

    print("\n✅ Training complete")
    print(f"Model saved:      {model_out}")
    print(f"Baselines saved:  {baseline_out}")
    print(f"Metrics saved:    {metrics_out}")
    print(f"Plot saved:       {plot_path}")
    print("\nTest metrics:")
    print(metrics["metrics_test"])


if __name__ == "__main__":
    main()