
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from app.preprocess import load_raw, clean_and_engineer, feature_columns

def build_stacked_regressor(random_state: int = 42) -> StackingRegressor:
    # Base learners
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state
    )
    xgb = XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1
    )
    lgbm = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1
    )

    estimators = [
        ("rf", rf),
        ("xgb", xgb),
        ("lgbm", lgbm),
    ]
    final_estimator = Ridge(alpha=1.0, random_state=random_state)
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=False,
        n_jobs=-1
    )
    return stack

def evaluate_timeseries(model, X, y, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, r2s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
        maes.append(mean_absolute_error(yte, pred))
        r2s.append(r2_score(yte, pred))
    return float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(r2s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to raw CSV dataset")
    ap.add_argument("--outdir", default="models", help="Where to write model artifacts")
    ap.add_argument("--target", required=True, choices=["AQI", "PM2.5"], help="Prediction target")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.csv)
    df = clean_and_engineer(raw, add_lags=True, add_roll=True)

    feats = feature_columns(df)
    X = df[feats]
    y = df[args.target].astype(float)

    # Standardize features then stack model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", build_stacked_regressor())
    ])

    rmse, mae, r2 = evaluate_timeseries(pipe, X, y, n_splits=3)
    print(f"[{args.target}] CV RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

    # Fit final on full data and export
    pipe.fit(X, y)
    joblib.dump({"pipeline": pipe, "features": feats}, outdir / f"{args.target.lower().replace('.','')}_model.joblib")

    # Save a compact, recent slice of engineered data for the web app
    df_tail = df.tail(2000).copy()
    df_tail.to_parquet(outdir / "recent_engineered.parquet", index=False)

    print("Saved:", outdir / f"{args.target.lower().replace('.','')}_model.joblib")

if __name__ == "__main__":
    main()
