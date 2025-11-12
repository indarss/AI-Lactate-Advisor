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

from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_lightgbm(X_train, y_train, X_val=None, y_val=None, params=None,
                   model_dir="models", model_name="lactate_lightgbm_model",
                   github_repo="AI-Lactate-Advisor", github_user="indarss"):
    """
    Train and version a LightGBM regression model.
    Automatically saves model with version timestamp, evaluates metrics,
    and returns the trained model.
    """

    os.makedirs(model_dir, exist_ok=True)

    # Default model hyperparameters
    if params is None:
        params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }

    print(f"🚀 Training {model_name} ...")
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate if validation data provided
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"✅ Validation R² = {r2:.3f}, MAE = {mae:.3f}")
    else:
        print("⚠️ No validation data provided. Skipping evaluation.")

 # --- Embed feature schema for downstream use ---
    model.feature_names_in_ = list(X_train.columns)
    model_metadata = {
        "model": model,
        "features": list(X_train.columns),
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name
    }
    
    # --- Save versioned and latest model ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(model_dir, f"{model_name}_{timestamp}.joblib")
    latest_path = os.path.join(model_dir, f"{model_name}.joblib")

    joblib.dump(model, versioned_path)
    joblib.dump(model, latest_path)

    print(f"💾 Model saved as:")
    print(f"   ┣━ {versioned_path}")
    print(f"   ┗━ {latest_path}")

   # --- Optional GitHub upload ---
    token = os.getenv("GITHUB_TOKEN")
    if token:
        try:
            from github import Github
            g = Github(token)
            repo = g.get_user().get_repo(github_repo)

            def upload_or_update(local_path, remote_path, message):
                with open(local_path, "rb") as f:
                    content = f.read()
                try:
                    existing = repo.get_contents(remote_path)
                    repo.update_file(existing.path, message, content, existing.sha, branch="main")
                    print(f"✅ Updated on GitHub: {remote_path}")
                except Exception:
                    repo.create_file(remote_path, message, content, branch="main")
                    print(f"✅ Uploaded to GitHub: {remote_path}")

            upload_or_update(latest_path, f"models/{model_name}.joblib", f"Auto-update: {model_name}")
            upload_or_update(versioned_path, f"models/{model_name}_{timestamp}.joblib", f"Auto-version: {model_name}")
            print("🌐 GitHub upload complete.")
        except Exception as e:
            print(f"⚠️ GitHub upload failed: {e}")
    else:
        print("⚠️ GITHUB_TOKEN not found — skipping GitHub upload.")

    return model
