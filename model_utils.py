# ==============================================================
# model_utils.py — Final Version (Compatible with Final app.py)
# ==============================================================

import numpy as np
import pandas as pd
import joblib
import shap
from lightgbm import LGBMRegressor

# ==============================================================
# MODEL LOADING
# ==============================================================

def load_model(path: str):
    """Safely load a model (plain joblib or dict containing schema)."""
    try:
        obj = joblib.load(path)
        if isinstance(obj, dict) and "model" in obj:
            return obj  # {model, features}
        return obj
    except Exception as e:
        print(f"[model_utils] Failed to load model {path}: {e}")
        return None


# ==============================================================
# FEATURE ENGINEERING
# ==============================================================

def add_hr_slopes(df: pd.DataFrame):
    """
    Compute HR slope (bpm per second) using gradient over time.
    Expects: columns 'time' and 'heart_rate'
    """
    df = df.copy()
    if "heart_rate" not in df.columns or "time" not in df.columns:
        df["hr_slope_time"] = 0.0
        return df

    try:
        df["hr_slope_time"] = np.gradient(df["heart_rate"], df["time"])
    except Exception:
        df["hr_slope_time"] = 0.0

    # Replace any infinities or NaN
    df["hr_slope_time"] = df["hr_slope_time"].replace([np.inf, -np.inf], 0).fillna(0)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 30):
    """
    Adds rolling averages for HR, power, cadence.
    """
    df = df.copy()
    roll_cols = [c for c in ["heart_rate", "power", "cadence"] if c in df.columns]

    for c in roll_cols:
        df[f"{c}_roll_mean"] = (
            df[c].rolling(window=window, min_periods=1).mean()
        )

    # Fill NaN
    for c in df.columns:
        if df[c].dtype in [float, int]:
            df[c] = df[c].fillna(0)

    return df


def prepare_features(df: pd.DataFrame):
    """
    Master feature-prep function used by both Training Notebook & app.py.
    Steps:
    1) rename hr → heart_rate
    2) add HR slope
    3) add rolling features
    """

    df = df.copy()

    # Standardize naming
    if "hr" in df.columns and "heart_rate" not in df.columns:
        df["heart_rate"] = df["hr"]

    df = add_hr_slopes(df)
    df = add_rolling_features(df, window=30)

    # Drop raw non-predictive fields if exist
    drop_cols = ["lactate", "recovery_score"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


# ==============================================================
# PREDICTION HELPERS
# ==============================================================

def _extract_model_and_features(model_obj):
    """Handle both plain model or dict container."""
    if isinstance(model_obj, dict) and "model" in model_obj:
        return model_obj["model"], model_obj.get("features")
    return model_obj, None


def predict_lactate(model_obj, X: pd.DataFrame):
    """Predict lactate using LightGBM model or wrapped dict."""
    model, feature_schema = _extract_model_and_features(model_obj)

    if feature_schema is not None:
        X = X.reindex(columns=feature_schema, fill_value=0.0)

    return model.predict(X)


def predict_recovery(model_obj, X: pd.DataFrame):
    """Predict recovery readiness."""
    model, feature_schema = _extract_model_and_features(model_obj)

    if feature_schema is not None:
        X = X.reindex(columns=feature_schema, fill_value=0.0)

    return model.predict(X)


# ==============================================================
# SHAP SUMMARIES
# ==============================================================

def get_shap_summary(model_obj, X: pd.DataFrame, top_n=12, show=False):
    """
    Compute SHAP values safely.
    Returns a shap.Explanation object.
    """
    model, feature_schema = _extract_model_and_features(model_obj)

    if feature_schema is not None:
        X = X.reindex(columns=feature_schema, fill_value=0)

    # TreeExplainer works for LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    if show:
        shap.summary_plot(shap_values.values, X, feature_names=X.columns)

    return shap_values


# ==============================================================
# OPTIONAL — smoothing (not used, but harmless)
# ==============================================================

def smooth_series(x, w=5):
    """Simple moving average."""
    x = pd.Series(x).rolling(window=w, min_periods=1).mean().values
    return x

