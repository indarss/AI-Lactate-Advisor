import os
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

# =========================================================
# 🧠 Utility functions for AI Lactate Advisor
# =========================================================

def load_model(path):
    """Safely load a LightGBM model from a .joblib file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def add_hr_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HR slope over time and vs power for trend tracking."""
    df = df.copy()
    if 'time' not in df.columns or 'heart_rate' not in df.columns:
        return df

    df['hr_slope_time'] = np.gradient(df['heart_rate'])
    if 'power' in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df['hr_slope_power'] = np.gradient(df['heart_rate']) / (np.gradient(df['power']) + 1e-6)
    else:
        df['hr_slope_power'] = 0.0
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Add rolling mean and std features for heart rate and power."""
    df = df.copy()
    if 'heart_rate' not in df.columns or 'power' not in df.columns:
        return df

    for col in ['heart_rate', 'power']:
        df[f'{col}_mean_{window}s'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_std_{window}s'] = df[col].rolling(window=window, min_periods=1).std()

    df['hr_power_ratio'] = (df['heart_rate'].rolling(window, min_periods=1).mean() + 1e-6) /                            (df['power'].rolling(window, min_periods=1).mean() + 1e-6)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare model-ready features (HR slope + rolling stats)."""
    df = add_hr_slopes(df)
    df = add_rolling_features(df, 30)
    df = df.dropna().reset_index(drop=True)
    return df


def make_features(df_window: pd.DataFrame) -> pd.DataFrame:
    """Quick window-based feature extraction for live or streaming prediction."""
    if df_window.empty:
        return pd.DataFrame([{
            "power_mean_30s": 0, "power_std_30s": 0, "power_slope_30s": 0,
            "hr_mean_30s": 0, "hr_std_30s": 0, "hr_slope_30s": 0,
            "power_hr_ratio": 0
        }])

    def slope(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]

    w = df_window
    power = w.get('power', np.zeros(len(w)))
    hr = w.get('hr', np.zeros(len(w)))

    feats = {
        'power_mean_30s': np.nanmean(power),
        'power_std_30s': np.nanstd(power),
        'power_slope_30s': slope(power),
        'hr_mean_30s': np.nanmean(hr),
        'hr_std_30s': np.nanstd(hr),
        'hr_slope_30s': slope(hr),
        'power_hr_ratio': (np.nanmean(power) + 1e-6) / (np.nanmean(hr) + 1e-6)
    }
    return pd.DataFrame([feats])


def predict_lactate(model, X: pd.DataFrame):
    """Predict lactate levels given feature matrix X."""
    preds = model.predict(X)
    return np.clip(preds, 0, 20)


def predict_recovery(model, X: pd.DataFrame):
    """Predict readiness/recovery score (0–100)."""
    preds = model.predict(X)
    return np.clip(preds, 0, 100)


@st.cache_resource(show_spinner=False)
def _get_explainer(model, sample_df):
    """Create or reuse SHAP explainer (cached per-model)."""
    return shap.Explainer(model, sample_df)


def get_shap_summary(model, X: pd.DataFrame, top_n: int = 10, show: bool = False):
    """Return SHAP values for model predictions."""
    explainer = _get_explainer(model, X)
    shap_values = explainer(X)
    if show:
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n)
    return shap_values


def smooth_series(series, window=5):
    """Simple moving average smoothing for chart display."""
    return series.rolling(window=window, min_periods=1).mean()
