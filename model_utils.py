"""
model_utils.py — AI Lactate Advisor utility functions
----------------------------------------------------
This module provides:
- Safe model loading
- Rolling-window and slope feature computation
- Prediction helpers for lactate and recovery
- SHAP explainability functions
- Data smoothing for real-time visualization

Author: Indars
Contact: sparnins@hotmail.com
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
from lightgbm import LGBMRegressor

# ------------------------------------------------------------
# 🧩  Model Loading
# ------------------------------------------------------------
def load_model(path: str):
    """Safely load a LightGBM model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# ------------------------------------------------------------
# 🧮  Feature Engineering
# ------------------------------------------------------------
def add_rolling_features(df: pd.DataFrame, window: int = 30):
    """Add rolling mean and std for HR and Power over given window."""
    df = df.copy()
    df[f"heart_rate_mean_{window}s"] = df["heart_rate"].rolling(window=window, min_periods=1).mean()
    df[f"power_mean_{window}s"] = df["power"].rolling(window=window, min_periods=1).mean()
    df[f"heart_rate_std_{window}s"] = df["heart_rate"].rolling(window=window, min_periods=1).std().fillna(0)
    df[f"power_std_{window}s"] = df["power"].rolling(window=window, min_periods=1).std().fillna(0)
    return df


def add_hr_slopes(df: pd.DataFrame):
    """Compute heart rate slope relative to time and power."""
    df = df.copy()
    df["hr_slope_time"] = df["heart_rate"].diff() / df["time"].diff()
    df["hr_slope_power"] = df["heart_rate"].diff() / df["power"].diff()
    return df.fillna(0)


def prepare_features(df: pd.DataFrame):
    """Apply all feature transformations."""
    df = add_hr_slopes(df)
    df = add_rolling_features(df, 30)
    return df


# ------------------------------------------------------------
# 🔮  Predictions
# ------------------------------------------------------------
def predict_lactate(model, df_features: pd.DataFrame):
    """Predict lactate concentration for a given set of features."""
    return np.clip(model.predict(df_features), 0, 12)


def predict_recovery(model, df_features: pd.DataFrame):
    """Predict recovery readiness score."""
    return np.clip(model.predict(df_features), 0, 100)


# ------------------------------------------------------------
# 💡  SHAP Explainability
# ------------------------------------------------------------
def get_shap_summary(model, X, top_n=10, show=True):
    """Generate global SHAP summary plot for the model."""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    if show:
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n)
    return shap_values


def get_shap_for_sample(model, X_sample):
    """Explain a single prediction using SHAP."""
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    return shap_values


# ------------------------------------------------------------
# 📈  Smoothing Utility
# ------------------------------------------------------------
def smooth_series(series, window=5):
    """Apply rolling smoothing for visualization."""
    return series.rolling(window=window, min_periods=1).mean()


# ------------------------------------------------------------
# 🧰  Model Training (optional)
# ------------------------------------------------------------
def train_lightgbm(X, y, params=None):
    """Train a LightGBM regressor with default or custom params."""
    default_params = dict(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )
    if params:
        default_params.update(params)
    model = LGBMRegressor(**default_params)
    model.fit(X, y)
    return model


# ------------------------------------------------------------
# ⚙️  Real-time Feature Builder and Visualization
# ------------------------------------------------------------
import matplotlib.pyplot as plt

def make_features(df_window):
    """Simple real-time feature engineering for small recent windows of HR and Power."""
    import numpy as np
    import pandas as pd
    if df_window.shape[0] == 0:
        return pd.DataFrame([{
            "power_mean_30s": 0,
            "power_std_30s": 0,
            "power_slope_30s": 0,
            "hr_mean_30s": 0,
            "hr_std_30s": 0,
            "hr_slope_30s": 0,
            "power_hr_ratio": 0
        }])
    w = df_window.copy()
    def slope(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr))
        coef = np.polyfit(x, arr, 1)
        return float(coef[0])
    power = w["power"].astype(float).values if "power" in w else np.zeros(len(w))
    hr = w["hr"].astype(float).values if "hr" in w else np.zeros(len(w))
    feats = {
        "power_mean_30s": float(np.nanmean(power)),
        "power_std_30s": float(np.nanstd(power)),
        "power_slope_30s": float(slope(power)),
        "hr_mean_30s": float(np.nanmean(hr)),
        "hr_std_30s": float(np.nanstd(hr)),
        "hr_slope_30s": float(slope(hr)),
        "power_hr_ratio": float((np.nanmean(power)+1e-6)/(np.nanmean(hr)+1e-6))
    }
    return pd.DataFrame([feats])


def plot_hr_slope(df, time_col='time', hr_col='heart_rate', power_col='power'):
    """Visualize heart rate slope trends over time and relative to power."""
    if 'hr_slope_time' not in df.columns:
        print("⚠️ No hr_slope_time column found. Make sure make_features() or add_hr_slopes() has been run.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[hr_col], label='Heart Rate (bpm)', color='red', alpha=0.7)
    plt.plot(df[time_col], df['hr_slope_time'] * 10 + df[hr_col].mean(),
             label='HR Slope (scaled)', color='blue', alpha=0.7)

    if power_col in df.columns:
        plt.plot(df[time_col], df[power_col] / df[power_col].max() * 50 + df[hr_col].min(),
                 label='Power (scaled)', color='green', alpha=0.5, linestyle='--')

    plt.title('Heart Rate Slope Trends')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
