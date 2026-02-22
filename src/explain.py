from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def get_paths():
    root = Path(__file__).resolve().parents[1]
    data_raw = root / "data" / "raw"
    models = root / "models"
    reports = root / "reports"
    figures = reports / "figures"

    for p in [data_raw, models, reports, figures]:
        p.mkdir(parents=True, exist_ok=True)

    return root, data_raw, models, reports, figures


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
        "Region": "region",
        "Date": "date",

     
        "vegetable_Commodity": "vegetable_commodity",
        "vegetable_Price per Unit (LKR/kg)": "price_lkr_per_kg",

        "vegitable_Commodity": "vegetable_commodity",
        "vegitable_Price per Unit (LKR/kg)": "price_lkr_per_kg",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    if df["date"].isna().mean() > 0.2:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", unit="D", origin="1899-12-30")

    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise ValueError(f"Found {bad} rows with invalid dates after parsing.")

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return df[feature_cols].copy()


def save_matplotlib_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Excel filename inside data/raw/")
    parser.add_argument("--sample_index", type=int, default=0, help="Row index (after sorting) for local explanation")
    parser.add_argument("--max_rows", type=int, default=8000, help="Cap rows to speed up SHAP computations")
    parser.add_argument("--html_rows", type=int, default=300, help="Rows to use for global force HTML")
    args = parser.parse_args()

    root, data_raw, models_dir, reports_dir, figures_dir = get_paths()


    bundle_path = models_dir / "veg_price_model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model not found: {bundle_path}\nRun training first: python -m src.train ..."
        )

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    xlsx_path = data_raw / args.file
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    df = safe_rename_columns(df)
    df = parse_date(df)
    df = add_time_features(df)
    df = basic_cleaning(df)


    df = df.sort_values("date").reset_index(drop=True)


    if len(df) > args.max_rows:
        df = df.tail(args.max_rows).reset_index(drop=True)

    X = build_features(df)


    X = X[feature_columns]


    explainer = shap.Explainer(model)
    shap_values = explainer(X)


    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    save_matplotlib_png(figures_dir / "shap_summary.png")

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    save_matplotlib_png(figures_dir / "shap_bar.png")

    html_rows = min(args.html_rows, len(X))
    X_html = X.sample(n=html_rows, random_state=42)

    sv_html = explainer(X_html)

    force_plot = shap.plots.force(
        sv_html.base_values[0],
        sv_html.values,
        X_html,
    )

    shap.save_html(str(reports_dir / "shap_summary.html"), force_plot)


    idx = int(args.sample_index)
    if idx < 0 or idx >= len(X):
        raise ValueError(f"sample_index out of range. Must be 0..{len(X)-1}")

    single = X.iloc[[idx]]
    single_sv = explainer(single)


    plt.figure()
    shap.plots.waterfall(single_sv[0], show=False)
    save_matplotlib_png(figures_dir / "shap_waterfall_sample.png")


    force_local = shap.plots.force(
    single_sv.base_values[0],
    single_sv.values[0],
    single,
    )

    shap.save_html(str(reports_dir / "shap_waterfall_sample.html"), force_local)


    sample_df = pd.DataFrame({
        "feature": single.columns,
        "value": single.iloc[0].values,
        "shap_value": single_sv.values[0],
    })
    sample_df.to_csv(reports_dir / "shap_values_sample.csv", index=False)

    print("\n✅ SHAP explainability generated")
    print(f"PNG:  {figures_dir / 'shap_summary.png'}")
    print(f"PNG:  {figures_dir / 'shap_bar.png'}")
    print(f"PNG:  {figures_dir / 'shap_waterfall_sample.png'}")
    print(f"HTML: {reports_dir / 'shap_summary.html'}")
    print(f"HTML: {reports_dir / 'shap_waterfall_sample.html'}")
    print(f"CSV:  {reports_dir / 'shap_values_sample.csv'}")


if __name__ == "__main__":
    main()