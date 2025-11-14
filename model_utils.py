import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

# ---------------------------------------
# Feature Engineering Utilities
# ---------------------------------------

def add_hr_slopes(df):
    df = df.copy()
    if 'heart_rate' in df.columns:
        df['hr_slope_time'] = df['heart_rate'].diff().fillna(0)
    elif 'hr' in df.columns:
        df['hr_slope_time'] = df['hr'].diff().fillna(0)
    else:
        df['hr_slope_time'] = 0
    return df


def add_rolling_features(df, window=30):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[f"{c}_roll{window}"] = df[c].rolling(window, min_periods=1).mean()
    return df


def prepare_features(df):
    df = df.copy()
    if 'hr' in df.columns:
        df = df.rename(columns={'hr': 'heart_rate'})
    df = add_hr_slopes(df)
    df = add_rolling_features(df, 30)
    return df


# ---------------------------------------
# Model Saving / Loading with Schema
# ---------------------------------------

def save_model_with_schema(model, feature_list, path):
    """Wrap model + schema in a dict."""
    wrapper = {
        "model": model,
        "features": feature_list
    }
    joblib.dump(wrapper, path)


def load_model(path):
    """Load model, schema-aware."""
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "features": None}


# ---------------------------------------
# Predictions
# ---------------------------------------

def predict_lactate(model_obj, X):
    model = model_obj["model"] if isinstance(model_obj, dict) else model_obj
    return model.predict(X)


def predict_recovery(model_obj, X):
    model = model_obj["model"] if isinstance(model_obj, dict) else model_obj
    return model.predict(X)


# ---------------------------------------
# Training Helper
# ---------------------------------------

def train_lightgbm(X_train, y_train, X_val=None, y_val=None, model_name="model", out_dir="models"):
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None:
        pred = model.predict(X_val)
        r2 = r2_score(y_val, pred)
        mae = mean_absolute_error(y_val, pred)
        print(f"Validation R2={r2:.3f} | MAE={mae:.3f}")
    else:
        r2 = mae = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    latest_path = f"{out_dir}/{model_name}.joblib"
    version_path = f"{out_dir}/{model_name}_{timestamp}.joblib"

    save_model_with_schema(model, list(X_train.columns), latest_path)
    save_model_with_schema(model, list(X_train.columns), version_path)

    print(f"Model saved to:\n  {latest_path}\n  {version_path}")

    return model, r2, mae
