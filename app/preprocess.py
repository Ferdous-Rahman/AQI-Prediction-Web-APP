
import pandas as pd
import numpy as np

POLLUTANTS = ["SO2","NO","NO2","NOX","CO","O3","PM2.5","PM10"]
MET_VARS = ["Wind Speed","Wind Dir","Temperature","RH","Solar Rad","BP","Rain"]

def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    # Parse datetime - the Date column already contains full datetime
    # Try direct parsing first (handles M/D/YYYY H:MM format)
    df["datetime"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # If that fails, try with dayfirst=True
    if df["datetime"].isna().mean() > 0.05:
        df["datetime"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    
    # Drop Date and Time columns if they exist
    df = df.drop(columns=[c for c in ["Date","Time"] if c in df.columns])
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    
    # Ensure numeric
    for c in POLLUTANTS + MET_VARS + ["AQI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_and_engineer(df: pd.DataFrame,
                       add_lags: bool = True,
                       add_roll: bool = True) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    # Handle missing values: time-aware interpolation for short gaps
    df.set_index("datetime", inplace=True)
    df = df.asfreq("H")  # hourly grid
    numeric_cols = [c for c in df.columns if c != "datetime"]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df.reset_index(inplace=True)

    # Basic outlier handling using IQR per column (light-touch; avoids deleting too much)
    for c in POLLUTANTS + MET_VARS + ["AQI"]:
        if c not in df.columns:
            continue
        s = df[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            continue
        lo, hi = q1 - 3*iqr, q3 + 3*iqr
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Calendar features
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month

    # Wind direction to sin/cos
    if "Wind Dir" in df.columns:
        wd = np.deg2rad(df["Wind Dir"].astype(float))
        df["wind_sin"] = np.sin(wd)
        df["wind_cos"] = np.cos(wd)

    # Lag features (for key pollutants and key met variables)
    if add_lags:
        lag_cols = ["PM2.5","PM10","NO2","SO2","CO","O3","Temperature","RH","Wind Speed"]
        lag_cols = [c for c in lag_cols if c in df.columns]
        for c in lag_cols:
            for lag in [1,3,6,12,24]:
                df[f"{c}_lag{lag}"] = df[c].shift(lag)

    # Rolling means
    if add_roll:
        roll_cols = ["PM2.5","PM10","NO2","CO","O3"]
        roll_cols = [c for c in roll_cols if c in df.columns]
        for c in roll_cols:
            for w in [3,6,24]:
                df[f"{c}_ma{w}"] = df[c].rolling(window=w, min_periods=1).mean()

    # Drop early rows with NaNs from lags
    df = df.dropna().reset_index(drop=True)
    return df

def feature_columns(df: pd.DataFrame) -> list[str]:
    ignore = {"datetime","AQI"}  # AQI is target for AQI model
    cols = [c for c in df.columns if c not in ignore]
    return cols

def aqi_category(aqi: float) -> str:
    # US EPA standard AQI categories
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"
