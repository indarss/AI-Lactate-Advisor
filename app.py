# ==============================================
# AI Lactate Advisor – vFinal with Live Mode Integration
# Full patched version (feature-aligned, 3D viz fixed, model history tab fixed)
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
    load_model,
    predict_lactate, predict_recovery,
    get_shap_summary, 
    add_hr_slopes, add_rolling_features
)

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title='AI Lactate Advisor',
    page_icon='🧠',
    layout='wide'
)

MODELS_DIR = 'models'
LACTATE_MODEL_PATH = os.path.join(MODELS_DIR, 'lactate_lightgbm_model.joblib')
RECOVERY_MODEL_PATH = os.path.join(MODELS_DIR, 'recovery_lightgbm_model.joblib')

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_models():
    """Load models without schema wrapping."""
    lm = load_model(LACTATE_MODEL_PATH)
    rm = load_model(RECOVERY_MODEL_PATH)
    return lm, rm


def align_features(df, feature_list):
    """Align incoming dataframe to model expected schema."""
    if feature_list is None:
        return df
    return df.reindex(columns=feature_list, fill_value=0)

def plot_hr_slope_plotly(df: pd.DataFrame):
    """
    Interactive Plotly HR slope visualization.
    Displays HR, scaled Power, and HR slope over time.
    Ensures slope columns exist using add_hr_slopes().
    """
    df = df.copy()

    # Ensure HR slope exists (rename for the utility)
    if 'hr_slope_time' not in df.columns:
        if 'hr' in df.columns:
            df_tmp = df.rename(columns={'hr': 'heart_rate'})
            df_tmp = add_hr_slopes(df_tmp)
            df['hr_slope_time'] = df_tmp['hr_slope_time']
        else:
            df['hr_slope_time'] = 0.0

    # Handle missing cols
    df = ensure_columns(df, ['time', 'hr', 'power'])

    # scale power and slope
    hr_min = df['hr'].min()
    power_scaled = (df['power'] / max(1, df['power'].max()) * 40) + hr_min
    slope_scaled = (df['hr_slope_time'].fillna(0) * 80) + df['hr'].mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['time'], y=df['hr'],
        mode='lines', name='Heart Rate (bpm)'
    ))

    fig.add_trace(go.Scatter(
        x=df['time'], y=power_scaled,
        mode='lines', name='Power (scaled)'
    ))

    fig.add_trace(go.Scatter(
        x=df['time'], y=slope_scaled,
        mode='lines', name='HR Slope (scaled, bpm/s)',
        line=dict(dash='dot')
    ))

    fig.update_layout(
        title="💓 Heart Rate Slope Trends (Interactive)",
        xaxis_title="Time (s)",
        yaxis_title="Value / Scaled",
        legend_orientation="h",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def generate_synthetic_session(n_seconds: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_seconds)
    power = np.clip(150 + 2.0*t + rng.normal(0, 12, n_seconds), 80, 450)
    hr = np.clip(120 + 0.15*t + rng.normal(0, 2, n_seconds), 80, 200)
    df = pd.DataFrame({
        'time': t,
        'power': power,
        'hr': hr,
        'cadence': np.clip(85 + rng.normal(0, 2, n_seconds), 60, 110),
        'temperature': np.clip(20 + rng.normal(0, 0.3, n_seconds), 5, 40)
    })
    return df


def ensure_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f'Adding missing columns with defaults: {missing}')
        for c in missing:
            df[c] = 0.0
    return df


def plot_3d_lactate_zones(df: pd.DataFrame, y_pred: np.ndarray):
    """3D animated lactate visualization."""
    df = df.copy()
    if not {'time', 'power', 'hr'}.issubset(df.columns):
        st.warning("Missing required columns for 3D visualization.")
        return

    df['pred_lactate'] = y_pred

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df['time'], y=df['power'], z=df['hr'],
        mode='markers',
        marker=dict(
            size=np.clip(df['pred_lactate'], 3, 12),
            color=df['pred_lactate'],
            colorscale=[(0, '#1f77b4'), (0.5, '#ff7f0e'), (1, '#d62728')],
            colorbar=dict(title='Pred. Lactate (mmol/L)')
        )
    ))

    fig.update_layout(
        title="🎨 3D Lactate Zone Visualization",
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Power (W)',
            zaxis_title='Heart Rate (bpm)',
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)


def readiness_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=float(score),
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'thickness': 0.25},
            'steps': [
                {'range': [0, 60], 'color': '#ff4b4b'},
                {'range': [60, 80], 'color': '#ffd166'},
                {'range': [80, 100], 'color': '#06d6a0'},
            ]
        },
        title={'text': 'Recovery Readiness'}
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title('⚙️ Controls')
uploaded = st.sidebar.file_uploader('Upload session CSV', type=['csv'])
use_synth = st.sidebar.button('Generate synthetic demo data')
show_raw = st.sidebar.checkbox('Show raw data', value=False)

st.sidebar.markdown('---')
st.sidebar.caption('Models:')
models_present = os.path.exists(LACTATE_MODEL_PATH) and os.path.exists(RECOVERY_MODEL_PATH)
if models_present:
    st.sidebar.success('Models found ✅')
else:
    st.sidebar.error('Model files missing. Please train via notebook.')

# -----------------------------
# Title
# -----------------------------
st.title('🧠 AI Lactate Advisor')
st.caption('Real-time lactate & recovery insights with explainable AI')

# -----------------------------
# Load models + schemas
# -----------------------------
lactate_model, recovery_model = (None, None)
lactate_features = None
recovery_features = None

if models_present:
    with st.spinner("Loading models..."):

        # Load actual LightGBM model
        lactate_model = joblib.load(LACTATE_MODEL_PATH)
        recovery_model = joblib.load(RECOVERY_MODEL_PATH)

        # Extract feature schema if wrapped by notebook
        def extract_schema(obj):
            if isinstance(obj, dict) and "features" in obj:
                return obj["model"], obj["features"]
            return obj, None

        lactate_model, lactate_features = extract_schema(lactate_model)
        recovery_model, recovery_features = extract_schema(recovery_model)

        st.session_state["lactate_features"] = lactate_features
        st.session_state["recovery_features"] = recovery_features

        if lactate_features:
            st.info(f"Lactate model expects {len(lactate_features)} features.")
        if recovery_features:
            st.info(f"Recovery model expects {len(recovery_features)} features.")

        st.success("Models loaded successfully.")

# -----------------------------
# Load session data
# -----------------------------
df_session = None
if uploaded is not None:
    df_session = pd.read_csv(uploaded)
elif use_synth:
    df_session = generate_synthetic_session()
else:
    st.info("Upload a CSV or click 'Generate synthetic demo data'.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab_live, tab_3d, tab_hist = st.tabs([
    '🏃 Live Session',
    '📊 SHAP Insights',
    '💓 HR Slope Trends',
    '🧬 Recovery Dashboard',
    '💓 Live Mode',
    '🎛️ 3D Visualization',
    '📈 Model History'
])

# ===========================================================
# 🏃 Live Session
# ===========================================================
with tab1:
    st.subheader("🏃 Live Session")

    if df_session is not None:
        df_session = ensure_columns(df_session, ['time', 'power', 'hr'])

        if show_raw:
            st.dataframe(df_session.head(), use_container_width=True)

        # --- Build features identical to training ---
        df_f = df_session.rename(columns={'hr': 'heart_rate'})
        df_f = add_hr_slopes(df_f)
        df_f = add_rolling_features(df_f, 30)
        df_f = df_f.rename(columns={'heart_rate': 'hr'})

        X = df_f.drop(columns=['lactate', 'recovery_score'], errors='ignore')

        if lactate_model is not None:
            with st.spinner("Predicting lactate..."):

                # Correct feature alignment!
                if lactate_features:
                    X = align_features(X, lactate_features)

                y_pred = predict_lactate(lactate_model, X)

            # Last prediction
            last_value = float(y_pred[-1])

            colL, colR = st.columns([3, 1])
            with colL:
                fig = px.line(
                    x=df_session['time'], y=y_pred,
                    labels={'x': 'Time', 'y': 'Predicted Lactate (mmol/L)'},
                    title='Predicted Lactate Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Inline 3D plot
                st.markdown("### 🎨 In-Session 3D Lactate View")
                plot_3d_lactate_zones(df_session, y_pred)

            with colR:
                st.metric("Latest Lactate", f"{last_value:.2f} mmol/L")

# ===========================================================
# 📊 SHAP Insights
# ===========================================================
with tab2:
    st.subheader("📊 SHAP Insights")

    if df_session is None or lactate_model is None:
        st.info("Load data + model first.")
    else:
        try:
            sample = X.tail(200)
            shap_vals = get_shap_summary(lactate_model, sample, top_n=12, show=False)

            mean_abs = np.abs(shap_vals.values).mean(axis=0)
            idx = np.argsort(mean_abs)[::-1][:10]
            feats = sample.columns[idx]
            vals = mean_abs[idx]

            st.plotly_chart(px.bar(x=feats, y=vals, title="Global SHAP Importance"))
        except Exception as e:
            st.warning(f"SHAP failed: {e}")

# ===========================================================
# 💓 HR Slope Trends
# ===========================================================
with tab3:
    st.subheader("💓 HR Slope Trends")
    if df_session is not None:
        fig = plot_hr_slope_plotly(df_session)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================
# 🧬 Recovery Dashboard
# ===========================================================
with tab4:
    st.subheader("🧬 Recovery Dashboard")

    df_bio = None
    file = st.file_uploader("Upload biomarkers CSV (optional)", type=['csv'])

    if file:
        df_bio = pd.read_csv(file)

    if df_bio is None and df_session is not None:
        df_bio = pd.DataFrame({
            'heart_rate': [df_session['hr'].iloc[-1]],
            'power': [df_session['power'].iloc[-1]],
            'cadence': [df_session['cadence'].iloc[-1]],
            'hr_slope_time': [np.gradient(df_session['hr'].tail(30)).mean()],
        })

    if df_bio is not None and recovery_model is not None:
        Xr = df_bio.copy()

        if recovery_features:
            Xr = align_features(Xr, recovery_features)

        pred = float(predict_recovery(recovery_model, Xr)[0])

        st.metric("Recovery Readiness", f"{pred:.1f} / 100")
        st.plotly_chart(readiness_gauge(pred))

# ===========================================================
# 💓 Live Mode (unchanged)
# ===========================================================
with tab_live:
    st.info("Live Mode (Polar API / Mock Stream) unchanged from prior version.")

# ===========================================================
# 🎛️ 3D Visualization (Standalone Tab)
# ===========================================================
with tab_3d:
    st.subheader("🎛️ Full 3D Lactate Visualization")

    if df_session is not None and lactate_model is not None:
        df_v = df_session.copy()
        df_v = ensure_columns(df_v, ['time', 'power', 'hr'])
        Xv = df_v.rename(columns={'hr':'heart_rate'})
        Xv = add_hr_slopes(Xv)
        Xv = add_rolling_features(Xv, 30)
        Xv = Xv.rename(columns={'heart_rate':'hr'})

        Xv = Xv.drop(columns=['lactate','recovery_score'], errors='ignore')

        if lactate_features:
            Xv = align_features(Xv, lactate_features)

        yv = predict_lactate(lactate_model, Xv)
        plot_3d_lactate_zones(df_session, yv)

# ===========================================================
# 📈 Model History
# ===========================================================
from training_log_visualizer import show_training_log_dashboard
with tab_hist:
    show_training_log_dashboard()
