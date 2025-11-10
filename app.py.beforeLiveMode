import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# === Model & utils ===
from model_utils import (
    make_features,
    load_model, prepare_features,
    predict_lactate, predict_recovery,
    get_shap_summary, smooth_series,
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
    # Safely load models once
    lm = load_model(LACTATE_MODEL_PATH)
    rm = load_model(RECOVERY_MODEL_PATH)
    return lm, rm

def generate_synthetic_session(n_seconds: int = 300, seed: int = 42) -> pd.DataFrame:
    # Create a 5-minute synthetic workout with HR and Power to demo the app.
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

def plot_hr_slope_plotly(df: pd.DataFrame):
    # Interactive Plotly HR slope trends (HR, Power, and HR slope).
    df = df.copy()
    if 'hr_slope_time' not in df.columns:
        df = add_hr_slopes(df.rename(columns={'hr': 'heart_rate'})).rename(columns={'heart_rate':'hr'})
    # scale power and slope for visualization
    hr_min = df['hr'].min() if 'hr' in df else 0
    power_scaled = (df['power'] / df['power'].max() * 40) + hr_min if 'power' in df else None
    slope_scaled = (df['hr_slope_time'].fillna(0) * 80) + df['hr'].mean()

    fig = go.Figure()
    if 'hr' in df:
        fig.add_trace(go.Scatter(x=df['time'], y=df['hr'],
                                 mode='lines', name='Heart Rate (bpm)'))
    if power_scaled is not None:
        fig.add_trace(go.Scatter(x=df['time'], y=power_scaled,
                                 mode='lines', name='Power (scaled)'))
    fig.add_trace(go.Scatter(x=df['time'], y=slope_scaled,
                             mode='lines', name='HR Slope (scaled, bpm/s)',
                             line=dict(dash='dot')))
    fig.update_layout(
        title='💓 Heart Rate Slope Trends (interactive)',
        xaxis_title='Time (s)',
        yaxis_title='Value / scaled',
        legend_orientation='h',
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def readiness_gauge(score: float):
    # Plotly gauge for recovery readiness (0-100).
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

# Load models (if available)
lactate_model, recovery_model = (None, None)
if models_present:
    try:
        with st.spinner('Loading models...'):
            lactate_model, recovery_model = _load_models()
            time.sleep(0.2)
        st.success('Models loaded.')
    except Exception as e:
        st.error(f'Failed to load models: {e}')

# -----------------------------
# Data source
# -----------------------------
df_session = None
if uploaded is not None:
    df_session = pd.read_csv(uploaded)
elif use_synth:
    df_session = generate_synthetic_session()
else:
    st.info("Upload a CSV or click 'Generate synthetic demo data' in the sidebar to begin.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    '🏃 Live Session',
    '📊 SHAP Insights',
    '💓 HR Slope Trends',
    '🧬 Recovery Dashboard'
])

# ===========================================================
# 🏃 Live Session
# ===========================================================
with tab1:
    st.subheader('🏃 Live Session')
    if df_session is not None:
        # Ensure necessary columns
        df_session = ensure_columns(df_session, ['time', 'power', 'hr'])
        st.write('Session preview:')
        if show_raw:
            st.dataframe(df_session.head(100), use_container_width=True)

        # Feature prep for lactate predictions
        with st.spinner('Computing features...'):
            df_for_model = df_session.rename(columns={'hr': 'heart_rate'})  # align with utility naming
            df_for_model = add_hr_slopes(df_for_model)
            df_for_model = add_rolling_features(df_for_model, 30)
            X = df_for_model.drop(columns=[c for c in ['lactate', 'recovery_score'] if c in df_for_model.columns], errors='ignore')

        # Predict lactate
        if lactate_model is not None:
            with st.spinner('Predicting lactate levels...'):
                y_pred = predict_lactate(lactate_model, X)
                time.sleep(0.1)
            st.success('Lactate prediction complete ✅')

            # Show last prediction
            latest_pred = float(y_pred[-1]) if len(y_pred) else None
            if latest_pred is not None:
                colL, colR = st.columns([3,1])
                with colL:
                    fig_lp = px.line(x=df_session['time'], y=y_pred, labels={'x': 'Time (s)', 'y': 'Predicted Lactate (mmol/L)'},
                                     title='Predicted Lactate Over Time')
                    st.plotly_chart(fig_lp, use_container_width=True)
                with colR:
                    st.metric('Latest lactate', f'{latest_pred:.2f} mmol/L')
        else:
            st.warning('Lactate model is not loaded.')

# ===========================================================
# 📊 SHAP Insights
# ===========================================================
with tab2:
    st.subheader('📊 Model Explainability (SHAP)')
    if uploaded is None and not use_synth:
        st.info('Upload a CSV or generate demo data in the sidebar to view SHAP insights.')
    elif lactate_model is None:
        st.warning('Lactate model not loaded.')
    else:
        # Reuse X from Live Session section if available
        try:
            sample_n = min(300, len(X))
        except Exception:
            st.info('Compute features in Live Session first.')
            sample_n = 0
        if sample_n >= 10:
            X_sample = X.tail(sample_n)
            with st.spinner('Computing SHAP global importance...'):
                try:
                    shap_values = get_shap_summary(lactate_model, X_sample, top_n=12, show=False)
                    # Build a simple Plotly bar from mean |SHAP|
                    mean_abs = np.abs(shap_values.values).mean(axis=0)
                    sort_idx = np.argsort(mean_abs)[::-1][:12]
                    top_feats = [X_sample.columns[i] for i in sort_idx]
                    top_vals = [mean_abs[i] for i in sort_idx]
                    fig_bar = px.bar(x=top_feats, y=top_vals, title='Global Feature Importance (mean |SHAP|)')
                    fig_bar.update_layout(xaxis_title='Feature', yaxis_title='Importance', xaxis_tickangle=45, margin=dict(l=10,r=10,t=40,b=120))
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.warning(f'SHAP summary failed: {e}')

            # Per-sample 'waterfall-style' using bar deltas on the last row
            st.markdown('**Per-sample explanation (latest window)**')
            try:
                last_row = X_sample.tail(1)
                shap_last = get_shap_summary(lactate_model, last_row, top_n=12, show=False)
                contrib = shap_last.values[0]
                order = np.argsort(np.abs(contrib))[::-1][:10]
                feats = last_row.columns[order]
                vals = contrib[order]
                fig_local = px.bar(x=feats, y=vals, title='Per-Sample SHAP Impact (last window)')
                fig_local.update_layout(xaxis_title='Feature', yaxis_title='Impact on prediction', xaxis_tickangle=45, margin=dict(l=10,r=10,t=40,b=120))
                st.plotly_chart(fig_local, use_container_width=True)
            except Exception as e:
                st.warning(f'Per-sample SHAP failed: {e}')

# ===========================================================
# 💓 HR Slope Trends
# ===========================================================
with tab3:
    st.subheader('💓 Heart Rate Slope Trends')
    if df_session is None:
        st.info('Upload a CSV or generate demo data to view HR slope trends.')
    else:
        # Ensure minimal columns
        df_plot = ensure_columns(df_session.copy(), ['time', 'hr', 'power'])
        fig = plot_hr_slope_plotly(df_plot)
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================
# 🧬 Recovery Dashboard
# ===========================================================
with tab4:
    st.subheader('🧬 Recovery Dashboard')
    st.caption('Uses biomarker + wearable features to estimate readiness (0–100).')

    # Option 1: User-provided biomarkers CSV
    biomarkers_file = st.file_uploader('Upload biomarkers CSV (optional)', type=['csv'], key='biomarkers')
    df_bio = None
    if biomarkers_file is not None:
        try:
            df_bio = pd.read_csv(biomarkers_file)
        except Exception as e:
            st.error(f'Failed to read biomarkers CSV: {e}')

    # Option 2: derive features from the session (if present)
    if df_bio is None and df_session is not None:
        # Merge session last values as a simple readiness proxy input
        bio_input = pd.DataFrame({
            'heart_rate': [df_session['hr'].tail(1).values[0] if 'hr' in df_session else 0],
            'power': [df_session['power'].tail(1).values[0] if 'power' in df_session else 0],
            'cadence': [df_session['cadence'].tail(1).values[0] if 'cadence' in df_session else 0],
            'hr_slope_time': [np.gradient(df_session['hr'].tail(30)).mean() if 'hr' in df_session else 0],
        })
        df_bio = bio_input

    if df_bio is not None and recovery_model is not None:
        # Prepare features & predict
        with st.spinner('Estimating recovery readiness...'):
            Xr = df_bio.copy()
            # Make sure numeric
            for c in Xr.columns:
                Xr[c] = pd.to_numeric(Xr[c], errors='coerce').fillna(0.0)
            rec_pred = predict_recovery(recovery_model, Xr)[0]
            time.sleep(0.2)
        st.success('Recovery score computed ✅')

        gc1, gc2 = st.columns([1,2])
        with gc1:
            st.plotly_chart(readiness_gauge(rec_pred), use_container_width=True)
        with gc2:
            st.write('**Inputs used:**')
            st.dataframe(Xr, use_container_width=True)
            if rec_pred >= 80:
                st.success('🟢 High readiness — suitable for intense training.')
            elif rec_pred >= 60:
                st.warning('🟡 Moderate readiness — consider tempo/threshold.')
            else:
                st.error('🔴 Low readiness — prioritize recovery or easy aerobic work.')
    else:
        if recovery_model is None:
            st.warning('Recovery model not loaded.')
        else:
            st.info('Upload a biomarkers CSV or provide a session to derive basic inputs.')