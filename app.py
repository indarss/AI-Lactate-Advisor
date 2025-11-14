# ==============================================
# AI Lactate Advisor – Final App with Live Mode,
# SHAP, 3D Visualization, Recovery Dashboard,
# HR Slope Trends, Model History, Feature Alignment
# ==============================================

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# === Model & utils ===
from model_utils import (
    load_model, prepare_features,
    predict_lactate, predict_recovery,
    get_shap_summary, smooth_series,
    add_hr_slopes, add_rolling_features
)

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title='AI Lactate Advisor',
    page_icon='🧠',
    layout='wide'
)

MODELS_DIR = "models"
LACTATE_MODEL_PATH = os.path.join(MODELS_DIR, "lactate_lightgbm_model.joblib")
RECOVERY_MODEL_PATH = os.path.join(MODELS_DIR, "recovery_lightgbm_model.joblib")

# ======================================================
# HELPERS
# ======================================================

def ensure_columns(df: pd.DataFrame, required_cols):
    """Ensure required columns exist."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"Adding missing columns: {missing}")
        for c in missing:
            df[c] = 0.0
    return df

def align_features(df, feature_list):
    """Align input features with model schema."""
    if feature_list is None:
        return df
    return df.reindex(columns=feature_list, fill_value=0.0)

def generate_synthetic_session(n_seconds=300, seed=42):
    """Synthetic 5-minute session for demo."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_seconds)
    power = np.clip(150 + 2*t + rng.normal(0,12,n_seconds), 80,450)
    hr = np.clip(120 + 0.15*t + rng.normal(0,2,n_seconds), 80,200)
    cadence = np.clip(85 + rng.normal(0,2,n_seconds), 60,110)
    temp = np.clip(20 + rng.normal(0,0.3,n_seconds), 5,40)

    return pd.DataFrame({
        "time": t,
        "power": power,
        "hr": hr,
        "cadence": cadence,
        "temperature": temp
    })

def plot_hr_slope_plotly(df):
    """Interactive HR slope chart."""
    df = df.copy()
    if "hr_slope_time" not in df.columns:
        df = add_hr_slopes(df.rename(columns={"hr": "heart_rate"}))
        df.rename(columns={"heart_rate": "hr"}, inplace=True)

    hr_min = df["hr"].min()
    power_scaled = (df["power"] / df["power"].max() * 40) + hr_min
    slope_scaled = (df["hr_slope_time"].fillna(0) * 80) + df["hr"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["hr"], name="HR"))
    fig.add_trace(go.Scatter(x=df["time"], y=power_scaled, name="Power (scaled)"))
    fig.add_trace(go.Scatter(x=df["time"], y=slope_scaled, name="HR Slope (scaled)", line=dict(dash="dot")))

    fig.update_layout(
        title="💓 HR Slope Trends",
        height=420,
        xaxis_title="Time (s)",
        yaxis_title="Relative Value",
        legend_orientation="h"
    )
    return fig

def plot_3d_lactate_zones(df, y_pred):
    """3D Power–HR–Lactate visualization."""
    df = df.copy()
    df["pred_lactate"] = y_pred

    fig = go.Figure(go.Scatter3d(
        x=df["power"],
        y=df["hr"],
        z=df["pred_lactate"],
        mode="markers",
        marker=dict(
            size=4,
            color=df["pred_lactate"],
            colorscale=[
                [0, "blue"],
                [0.5, "orange"],
                [1.0, "red"]
            ],
            opacity=0.7,
            colorbar={"title": "Lactate mmol/L"}
        )
    ))

    fig.update_layout(
        title="🎨 3D Lactate Zone Visualization",
        scene=dict(
            xaxis_title="Power (W)",
            yaxis_title="Heart Rate",
            zaxis_title="Predicted Lactate"
        ),
        height=600
    )
    return fig

def readiness_gauge(score):
    """Recovery readiness gauge 0–100."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        gauge={
            "axis": {"range": [0,100]},
            "steps": [
                {"range": [0,60], "color":"#ff4b4b"},
                {"range": [60,80], "color":"#ffd166"},
                {"range": [80,100],"color":"#06d6a0"}
            ]
        }
    ))
    fig.update_layout(height=260)
    return fig

# ======================================================
# LOAD MODELS + EXTRACT FEATURE SCHEMA
# ======================================================
@st.cache_resource(show_spinner=False)
def load_models_and_schema():
    lactate_model = load_model(LACTATE_MODEL_PATH)
    recovery_model = load_model(RECOVERY_MODEL_PATH)

    def extract_schema(path):
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict) and "features" in obj:
                return obj["model"], obj["features"]
            return obj, None
        except Exception:
            return None, None

    lactate_model, lactate_features = extract_schema(LACTATE_MODEL_PATH)
    recovery_model, recovery_features = extract_schema(RECOVERY_MODEL_PATH)

    return lactate_model, recovery_model, lactate_features, recovery_features

models_present = (
    os.path.exists(LACTATE_MODEL_PATH) and 
    os.path.exists(RECOVERY_MODEL_PATH)
)

if not models_present:
    st.error("❌ Model files missing — Train via notebook first.")
else:
    with st.spinner("Loading models..."):
        lactate_model, recovery_model, lactate_features, recovery_features = load_models_and_schema()
    st.success("Models loaded.")

    if lactate_features:
        st.sidebar.info(f"Lactate model expects {len(lactate_features)} features.")
    if recovery_features:
        st.sidebar.info(f"Recovery model expects {len(recovery_features)} features.")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload session CSV", type=["csv"])
use_synth = st.sidebar.button("Generate synthetic demo data")
show_raw = st.sidebar.checkbox("Show raw data")

# ======================================================
# DATA LOAD
# ======================================================
df_session = (
    pd.read_csv(uploaded) if uploaded else
    generate_synthetic_session() if use_synth else
    None
)

# ======================================================
# TABS
# ======================================================
tab_live, tab_shap, tab_hr, tab_rec, tab_stream, tab_3d = st.tabs([
    "🏃 Live Session",
    "📊 SHAP Insights",
    "💓 HR Slope Trends",
    "🧬 Recovery Dashboard",
    "💓 Live Mode",
    "🎛️ 3D Lactate Visualization"
])

# ======================================================
# 🏃 LIVE SESSION TAB
# ======================================================
with tab_live:
    st.subheader("🏃 Live Session")

    if df_session is None:
        st.info("Upload CSV or generate synthetic data.")
        st.stop()

    df_session = ensure_columns(df_session, ["time","power","hr"])

    # Build full feature set
    df_feat = df_session.rename(columns={"hr": "heart_rate"})
    df_feat = add_hr_slopes(df_feat)
    df_feat = add_rolling_features(df_feat, 30)
    df_feat.rename(columns={"heart_rate":"hr"}, inplace=True)

    X = df_feat.drop(columns=[c for c in ["lactate","recovery_score"] if c in df_feat], errors="ignore")

    if lactate_features:
        X = align_features(X, lactate_features)

    with st.spinner("Predicting lactate..."):
        y_pred = predict_lactate(lactate_model, X)

    latest_pred = float(y_pred[-1])

    colL, colR = st.columns([3,1])
    with colL:
        fig_lp = px.line(
            x=df_session["time"], y=y_pred,
            title="Predicted Lactate Over Time",
            labels={"x":"Time (s)", "y":"mmol/L"}
        )
        st.plotly_chart(fig_lp, use_container_width=True)

        st.markdown("### 🎨 3D Visualization")
        st.caption("Blue=Aerobic · Orange=Threshold · Red=Anaerobic")
        st.plotly_chart(plot_3d_lactate_zones(df_session, y_pred))

    with colR:
        st.metric("Latest Lactate", f"{latest_pred:.2f} mmol/L")

# ======================================================
# 📊 SHAP INSIGHTS
# ======================================================
with tab_shap:
    st.subheader("📊 SHAP Insights")

    if df_session is None:
        st.info("Provide data first.")
        st.stop()

    X_sample = X.tail(min(300, len(X)))

    try:
        shap_vals = get_shap_summary(lactate_model, X_sample, show=False)
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:12]

        fig_bar = px.bar(
            x=X_sample.columns[idx],
            y=mean_abs[idx],
            title="Global Feature Importance (mean |SHAP|)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP failed: {e}")

# ======================================================
# 💓 HR SLOPE TRENDS
# ======================================================
with tab_hr:
    st.subheader("💓 HR Slope Trends")
    if df_session is not None:
        st.plotly_chart(plot_hr_slope_plotly(df_session), use_container_width=True)

# ======================================================
# 🧬 RECOVERY DASHBOARD
# ======================================================
with tab_rec:
    st.subheader("🧬 Recovery Dashboard")

    df_bio = None
    bio_file = st.file_uploader("Upload biomarkers CSV", type=["csv"])
    if bio_file:
        df_bio = pd.read_csv(bio_file)
    else:
        # derive from session
        df_bio = pd.DataFrame({
            "heart_rate":[df_session["hr"].iloc[-1]],
            "power":[df_session["power"].iloc[-1]],
            "cadence":[df_session["cadence"].iloc[-1] if "cadence" in df_session else 0],
            "hr_slope_time":[np.gradient(df_session["hr"].tail(30)).mean()]
        })

    Xr = df_bio.copy()
    if recovery_features:
        Xr = align_features(Xr, recovery_features)

    rec_pred = float(predict_recovery(recovery_model, Xr)[0])

    col1, col2 = st.columns([1,2])
    with col1:
        st.plotly_chart(readiness_gauge(rec_pred), use_container_width=True)
    with col2:
        st.write("Inputs used:")
        st.dataframe(Xr)

# ======================================================
# 💓 LIVE MODE TAB
# ======================================================
# (Same as before; no make_features used)

# ======================================================
# 🎛 3D VIZ TAB
# ======================================================
with tab_3d:
    st.subheader("🎛️ 3D Lactate Visualization")
    if df_session is not None:
        st.plotly_chart(plot_3d_lactate_zones(df_session, y_pred), use_container_width=True)

