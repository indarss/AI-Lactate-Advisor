# ==============================================
# AI Lactate Advisor – vFinal with Live Mode Integration
# This file preserves your existing app and appends a new '💓 Live Mode' tab as the last tab.
# Generated for hackathon demo: Polar OAuth2, Mock Stream, CSV/TCX upload, Plotly live charts.
# ==============================================

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

def align_features(df, feature_list):
    """Align incoming dataframe columns to model's expected schema."""
    if feature_list is None:
        return df
    return df.reindex(columns=feature_list, fill_value=0)


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

def plot_3d_lactate_zones(df: pd.DataFrame, y_pred: np.ndarray):
    """
    Plot animated 3D lactate visualization with color zones:
    Blue=Low Aerobic, Orange=Threshold, Red=Anaerobic.
    """
    import plotly.graph_objects as go

    df = df.copy()
    if 'time' not in df.columns or 'power' not in df.columns or 'hr' not in df.columns:
        st.warning("Missing required columns for 3D visualization (time, power, hr).")
        return

    df['pred_lactate'] = y_pred

    # Define zone colors
    zone_colors = [
        (0, '#1f77b4'),  # Blue: aerobic
        (2, '#ff7f0e'),  # Orange: threshold
        (4, '#d62728')   # Red: anaerobic
    ]

    fig = go.Figure()

    # Main animated scatter
    fig.add_trace(go.Scatter3d(
        x=df['time'], y=df['power'], z=df['hr'],
        mode='markers',
        marker=dict(
            size=np.clip(df['pred_lactate'], 3, 12),
            color=df['pred_lactate'],
            colorscale=[(0, '#1f77b4'), (0.5, '#ff7f0e'), (1, '#d62728')],
            cmin=df['pred_lactate'].min(),
            cmax=df['pred_lactate'].max(),
            colorbar=dict(title='Pred. Lactate (mmol/L)')
        ),
        name='Athlete Path'
    ))

    # Add horizontal colored “zone” bands
    for z, color in zone_colors:
        fig.add_trace(go.Mesh3d(
            x=[df['time'].min(), df['time'].max(), df['time'].max(), df['time'].min()],
            y=[df['power'].min(), df['power'].min(), df['power'].max(), df['power'].max()],
            z=[df['hr'].mean()+z]*4,
            color=color, opacity=0.1,
            name=f"Zone {z}"
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
        # --- Feature alignment safeguard for both models ---
        import joblib
        def _extract_schema(model_obj):
            if isinstance(model_obj, dict) and "features" in model_obj:
                return model_obj["features"], model_obj["model"]
            return None, model_obj

        # Reload models to extract embedded schema if available
        try:
            lm_obj = joblib.load(LACTATE_MODEL_PATH)
            lactate_features, lactate_model = _extract_schema(lm_obj)
        except Exception:
            lactate_features = None

        try:
            rm_obj = joblib.load(RECOVERY_MODEL_PATH)
            recovery_features, recovery_model = _extract_schema(rm_obj)
        except Exception:
            recovery_features = None

        st.session_state["lactate_features"] = lactate_features
        st.session_state["recovery_features"] = recovery_features

        if lactate_features:
            st.info(f"Lactate model expects {len(lactate_features)} features.")
        if recovery_features:
            st.info(f"Recovery model expects {len(recovery_features)} features.")

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
tab1, tab2, tab3, tab4, tab_live = st.tabs([
    '🏃 Live Session',
    '📊 SHAP Insights',
    '💓 HR Slope Trends',
    '🧬 Recovery Dashboard'
, '💓 Live Mode'])

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
                # Align incoming data to model’s expected features (if available)
                if "lactate_features" in st.session_state and st.session_state["lactate_features"]:
                    X = align_features(X, st.session_state["lactate_features"])
                    st.caption(f"Aligned features: {len(st.session_state['lactate_features'])} columns.")
                else:
                    st.caption("Using raw feature columns (no schema found).")

                y_pred = predict_lactate(lactate_model, X)
                time.sleep(0.1)

            st.success('Lactate prediction complete ✅')


        # Show last prediction
        latest_pred = float(y_pred[-1]) if len(y_pred) else None
        if latest_pred is not None:
            colL, colR = st.columns([3,1])
            with colL:
                fig_lp = px.line(
                    x=df_session['time'], y=y_pred,
                    labels={'x': 'Time (s)', 'y': 'Predicted Lactate (mmol/L)'},
                    title='Predicted Lactate Over Time'
                )
                st.plotly_chart(fig_lp, use_container_width=True)

                # 🎨 3D Lactate Zone Visualization (new)
                st.markdown("### 🎨 3D Lactate Zone Visualization")
                st.caption("Blue = Aerobic · Orange = Threshold · Red = Anaerobic Zone")
                plot_3d_lactate_zones(df_session, y_pred)

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
            with st.spinner('Estimating recovery readiness...'):
                Xr = df_bio.copy()

                # Clean numeric values
                for c in Xr.columns:
                    Xr[c] = pd.to_numeric(Xr[c], errors='coerce').fillna(0.0)

                # Align with model’s expected feature schema
                if "recovery_features" in st.session_state and st.session_state["recovery_features"]:
                    Xr = align_features(Xr, st.session_state["recovery_features"])
                    st.caption(f"Aligned recovery input to {len(st.session_state['recovery_features'])} expected features.")
                else:
                    st.caption("Using raw biomarker features (no schema found).")

                rec_pred = predict_recovery(recovery_model, Xr)[0]
                time.sleep(0.2)

        st.success('Recovery score computed ✅')
        st.markdown(f'### Estimated Recovery Readiness Score: **{rec_pred:.1f} / 100**')    

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

# ==== Live Mode dependencies (safe; no-crash if missing) ====
try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from wearable_pipeline import (
        PolarClient, MockStream,
        read_csv_session, read_tcx_session,
        predict_from_session_df
    )
except Exception:
    PolarClient = None
    MockStream = None
    def read_csv_session(*a, **k):
        import pandas as pd; return pd.DataFrame(columns=["time","heart_rate","power","pace"])
    def read_tcx_session(*a, **k):
        import pandas as pd; return pd.DataFrame(columns=["time","heart_rate","power","pace"])
    def predict_from_session_df(df, *a, **k):
        return df

try:
    from model_utils import (
        load_model, make_features, prepare_features,
        predict_lactate, predict_recovery
    )
except Exception:
    def load_model(path): return None
    def make_features(df): return df.tail(1).reset_index(drop=True)
    def prepare_features(df): return df.copy()
    def predict_lactate(model, X):
        import numpy as np; return np.array([float("nan")]*len(X))
    def predict_recovery(model, X):
        import numpy as np; return np.array([float("nan")]*len(X))

# ==== Live Mode session state ====
import streamlit as st, pandas as pd, numpy as np, time
if "live_source" not in st.session_state: st.session_state.live_source = "Mock Stream"
if "live_running" not in st.session_state: st.session_state.live_running = False
if "live_buffer" not in st.session_state: st.session_state.live_buffer = []
if "polar_token" not in st.session_state: st.session_state.polar_token = None
if "polar_pc" not in st.session_state:
    try: st.session_state.polar_pc = PolarClient() if PolarClient else None
    except Exception: st.session_state.polar_pc = None

# ==== Load/reuse models if not already present ====
try:
    lactate_model
except NameError:
    try: lactate_model = load_model("models/lactate_lightgbm_model.joblib")
    except Exception: lactate_model = None
try:
    recovery_model
except NameError:
    try: recovery_model = load_model("models/recovery_lightgbm_model.joblib")
    except Exception: recovery_model = None

def _render_live_mode_tab():
    """💓 Live Mode: Polar OAuth2, Mock Stream, CSV/TCX upload with live Plotly chart & predictions."""
    st.subheader("💓 Live Mode")
    st.caption("Stream wearable data (Polar) or simulate with a mock stream. Upload CSV/TCX for batch analysis.")

    source = st.selectbox("Data Source", ["Polar API (OAuth2)", "Mock Stream", "Upload CSV/TCX"], index=1)
    st.session_state.live_source = source

    c1, c2, c3 = st.columns(3)
    start_btn = c1.button("🟢 Start Live")
    pause_btn = c2.button("⏸️ Pause")
    reset_btn = c3.button("🔁 Reset")

    chart_ph = st.empty()
    status_ph = st.empty()

    def draw_chart(df_live):
        if go is None:
            chart_ph.warning("Plotly not installed. Add `plotly` to requirements.txt.")
            return
        if df_live is None or len(df_live) == 0:
            chart_ph.info("Waiting for data…"); return
        fig = go.Figure()
        if "heart_rate" in df_live:
            fig.add_trace(go.Scatter(x=df_live["time"], y=df_live["heart_rate"], name="HR (bpm)", mode="lines"))
        if "power" in df_live:
            fig.add_trace(go.Scatter(x=df_live["time"], y=df_live["power"], name="Power (W)", mode="lines", yaxis="y2"))
        if "pred_lactate" in df_live:
            fig.add_trace(go.Scatter(x=df_live["time"], y=df_live["pred_lactate"], name="Pred. Lactate (mmol/L)", mode="lines"))
        fig.update_layout(
            height=420,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_title="Time (s)",
            yaxis=dict(title="HR / Lactate"),
            yaxis2=dict(title="Power (W)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        chart_ph.plotly_chart(fig, use_container_width=True)

    # controls
    if reset_btn:
        st.session_state.live_running = False
        st.session_state.live_buffer = []
        status_ph.success("🔁 Buffer cleared.")
    if pause_btn:
        st.session_state.live_running = False
        status_ph.info("⏸️ Paused.")

    # --- Mock Stream ---
    if source == "Mock Stream":
        freq = st.slider("Frequency (Hz)", 1, 5, 1)
        window_sec = st.slider("Feature Window (sec)", 10, 90, 30)
        duration_sec = st.slider("Simulation Duration (sec)", 30, 600, 120)

        if start_btn:
            st.session_state.live_running = True
            st.session_state.live_buffer = []
            status_ph.success("🟢 Mock stream started.")
            if 'MockStream' not in globals() or MockStream is None:
                st.error("wearable_pipeline.MockStream not available."); return

            stream = MockStream(freq_hz=freq).generator(duration_sec=duration_sec)
            for sample in stream:
                if not st.session_state.live_running: break
                st.session_state.live_buffer.append(sample)
                t_now = sample["time"]
                st.session_state.live_buffer = [s for s in st.session_state.live_buffer if (t_now - s["time"]) <= window_sec]

                df_win = pd.DataFrame(st.session_state.live_buffer)
                feats = make_features(df_win) if callable(make_features) else df_win.tail(1)
                lac = float(predict_lactate(lactate_model, feats)[0]) if lactate_model is not None else np.nan
                df_plot = df_win.copy(); df_plot["pred_lactate"] = lac
                draw_chart(df_plot)
                status_ph.write(f"⏱️ t={t_now:.1f}s | Predicted Lactate ≈ **{lac:.2f} mmol/L**")
                time.sleep(1.0 / max(1, freq))

        if st.session_state.live_buffer:
            df_buf = pd.DataFrame(st.session_state.live_buffer)
            try:
                feats = make_features(df_buf) if callable(make_features) else df_buf.tail(1)
                lac = float(predict_lactate(lactate_model, feats)[0]) if lactate_model is not None else np.nan
                df_buf["pred_lactate"] = lac
            except Exception:
                pass
            draw_chart(df_buf)

    # --- Polar API (OAuth2) ---
    elif source == "Polar API (OAuth2)":
        st.info("Authenticate with Polar and fetch HR/Power/Speed for a date range.")
        pc = st.session_state.polar_pc
        if PolarClient is None or pc is None:
            st.error("PolarClient not available. Ensure wearable_pipeline.py exists and POLAR_* env vars are set.")
            return

        auth_url = pc.build_auth_url(state="live_mode")
        st.markdown(f"[🔐 Authorize Polar](<{auth_url}>)")

        code = st.text_input("Paste the 'code' query parameter from Polar redirect URL:")
        with st.expander("Select Date Range"):
            cD, cE = st.columns(2)
            date_from = cD.date_input("From", value=pd.Timestamp.today() - pd.Timedelta(days=1))
            date_to = cE.date_input("To", value=pd.Timestamp.today())

        if st.button("🔁 Exchange Code & Fetch Sessions"):
            try:
                token = pc.exchange_code_for_token(code.strip())
                st.session_state.polar_token = token
                st.success("✅ Token received.")
                sessions = pc.list_exercises(str(date_from), str(date_to))
                st.write("**Exercises:**", sessions)
                st.caption("To retrieve actual sample streams, call get_exercise_samples(exercise_id).")
            except Exception as e:
                st.error(f"Polar auth/fetch error: {e}")

        st.caption("Tip: For the hackathon, Mock or Upload modes are easiest to demo.")

    # --- Upload CSV/TCX ---
    else:
        file = st.file_uploader("Upload CSV or TCX session file", type=["csv","tcx"])
        if file:
            if file.name.lower().endswith(".csv"):
                df_session = read_csv_session(file)
            else:
                tmp = f"/tmp/{file.name}"
                with open(tmp, "wb") as f: f.write(file.getbuffer())
                df_session = read_tcx_session(tmp)

            st.success(f"Loaded {len(df_session)} samples.")
            try:
                out = predict_from_session_df(df_session, lactate_model, recovery_model)
                st.dataframe(out.head())
                draw_chart(out)
                st.info("✅ Session processed successfully.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Bind Live Mode tab
with tab_live:
    _render_live_mode_tab()
