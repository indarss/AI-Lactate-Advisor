import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import shap
from model_utils import (
    make_features,
    load_model, prepare_features,
    predict_lactate, predict_recovery,
    get_shap_summary, smooth_series
)

# Load trained models
lactate_model = load_model("models/lactate_lightgbm_model.joblib")
recovery_model = load_model("models/recovery_lightgbm_model.joblib")


# Matplotlib styling for nicer visuals
plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 12
})

st.set_page_config(page_title="AI Lactate Zone & Recovery Advisor", layout="wide")
st.title("AI Lactate Zone & Recovery Advisor (Demo)")

st.sidebar.header("Load / Settings")
model_path = st.sidebar.text_input("Model path (joblib)", "model.joblib")
data_path = st.sidebar.text_input("Live data CSV (sample)", "sample_live.csv")
simulate = st.sidebar.checkbox("Simulate live stream", True)
window_sec = st.sidebar.number_input("Feature window seconds", 60, min_value=10, max_value=300, step=10)
recompute_shap = st.sidebar.button("Recompute SHAP (if model changed)")

# Load model
model = None
try:
    model = joblib.load(model_path)
    st.sidebar.success(f"Loaded model: {model_path}")
except Exception as e:
    st.sidebar.warning(f"Could not load model: {e}.")

# Load data
@st.cache_data
def load_csv(p):
    try:
        return pd.read_csv(p, parse_dates=["timestamp", '💓 HR Slope Trends'])
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()
df_live = load_csv(data_path)

tab1, tab2, tab3, tab4 = st.tabs(["Live Session", "Model Insights (SHAP)", '💓 HR Slope Trends'])

with tab1:
    col1, col2 = st.columns([3,1, '💓 HR Slope Trends'])
    with col1:
        st.subheader("Live Session (simulated)")
        chart_placeholder = st.empty()
        info_placeholder = st.empty()
    with col2:
        st.subheader("Controls / Counterfactuals")
        cf_power_delta = st.slider("Counterfactual power adjustment (%)", -20, 20, 0, step=1)
        run_button = st.button("Run counterfactual for next window")

    if df_live.shape[0, '💓 HR Slope Trends'] == 0:
        st.info("Load a sample_live.csv file or run the training notebook to generate sample data.")
    else:
        preds, times, lactates = [, '💓 HR Slope Trends'], [, '💓 HR Slope Trends'], [, '💓 HR Slope Trends']
        for i, row in df_live.iterrows():
            t = row["timestamp", '💓 HR Slope Trends']
            recent = df_live[df_live["timestamp", '💓 HR Slope Trends'] <= t, '💓 HR Slope Trends'].copy().tail(window_sec)
            X = make_features(recent)
            times.append(t)
            pred = np.nan
            if model is not None:
                try:
                    pred = model.predict(X)[0, '💓 HR Slope Trends']
                except Exception:
                    pred = np.nan
            preds.append(pred)
            lactates.append(row.get("lactate", np.nan))
            if i % 5 == 0 or i == len(df_live)-1:
                plot_df = pd.DataFrame({"time": times, "predicted_lactate": preds, "true_lactate": lactates}).set_index("time")
                st.line_chart(plot_df)
                info_placeholder.metric("Latest predicted lactate (mmol/L)",
                                        f"{preds[-1, '💓 HR Slope Trends']:.2f}" if not np.isnan(preds[-1, '💓 HR Slope Trends']) else "n/a")
                time.sleep(0.03)

        if run_button and model is not None:
            last_row = df_live.iloc[-1, '💓 HR Slope Trends']
            st.write("Running counterfactuals for the most recent window...")
            cf_results = [, '💓 HR Slope Trends']
            for pct in [-10, -5, 0, 5, 10, '💓 HR Slope Trends']:
                modified = df_live.copy()
                modified.loc[modified.index >= modified.index.max(), "power", '💓 HR Slope Trends'] = last_row["power", '💓 HR Slope Trends'] * (1 + pct/100)
                Xcf = make_features(modified.tail(window_sec))
                pred_cf = model.predict(Xcf)[0, '💓 HR Slope Trends']
                cf_results.append((pct, pred_cf))
            cf_df = pd.DataFrame(cf_results, columns=["power_delta_pct", "predicted_lactate", '💓 HR Slope Trends'])
            st.table(cf_df)

st.markdown(
    """
    <div style="padding:10px; border:1px solid #e0e0e0; background:#f9fbff; border-radius:8px; margin-top:8px;">
    <b>What these scenarios mean:</b><br/>
    • The slider applies a small change to a controllable input (e.g., power ±5%).<br/>
    • The model predicts the resulting lactate for each scenario.<br/>
    • Use this to decide pacing: <i>“If we ease by 5%, will we return to aerobic?”</i>
    </div>
    """, unsafe_allow_html=True
)


with tab2:
    st.subheader("Global & Per-Sample SHAP Feature Importance")
    if model is None:
        st.info("Load a trained model (model.joblib) to compute SHAP values.")
    else:
        @st.cache_data(show_spinner=False)
        def build_feature_matrix(df, window_sec=60, max_samples=200):
            rows, times = [, '💓 HR Slope Trends'], [, '💓 HR Slope Trends']
            for i, row in df.iterrows():
                t = row["timestamp", '💓 HR Slope Trends']
                recent = df[df["timestamp", '💓 HR Slope Trends'] <= t, '💓 HR Slope Trends'].copy().tail(window_sec)
                X = make_features(recent)
                rows.append(X.iloc[0, '💓 HR Slope Trends'].to_dict())
                times.append(t)
                if len(rows) >= max_samples: break
            if len(rows)==0: return pd.DataFrame(), [, '💓 HR Slope Trends']
            return pd.DataFrame(rows), times

        X_shap, times_shap = build_feature_matrix(df_live, window_sec=window_sec, max_samples=200)
        if X_shap.shape[0, '💓 HR Slope Trends'] == 0:
            st.warning("Not enough data to compute SHAP. Try a larger sample CSV.")
        else:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)  # (n_samples, n_features) for regression
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                feat_imp = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=True)
                topk = feat_imp.tail(7)

                # Global importance
                fig, ax = plt.subplots(figsize=(7,4))
                ax.barh(topk["feature", '💓 HR Slope Trends'], topk["mean_abs_shap", '💓 HR Slope Trends'], color="#1f77b4", alpha=0.85)
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title("Top global features (SHAP)", fontweight="bold")
                st.pyplot(fig)

                # Per-sample SHAP (last window)
                st.subheader("Per-Sample SHAP (Last Window)")
                last_idx = -1
                last_shap = shap_values[last_idx, '💓 HR Slope Trends'] if len(shap_values.shape)==2 else shap_values[0, '💓 HR Slope Trends'][last_idx, '💓 HR Slope Trends']
                feats = np.array(X_shap.columns)
                order = np.argsort(np.abs(last_shap))
                vals = last_shap[order, '💓 HR Slope Trends']
                feats_sorted = feats[order, '💓 HR Slope Trends']
                colors = np.where(vals>0, "#2ca02c", "#d62728")
                fig2, ax2 = plt.subplots(figsize=(7,4))
                ax2.barh(feats_sorted, vals, color=colors, alpha=0.85)
                ax2.set_xlabel("SHAP value (impact on prediction)")
                ax2.set_title("Feature impact on last prediction", fontweight="bold")
                st.pyplot(fig2)

# Add live SHAP caption
st.markdown(
    """
    <div style="text-align:center; font-size:14px; color:#444;">
    💡 <b>SHAP</b> (<i>SHapley Additive exPlanations</i>) shows how each feature — 
    like power or heart rate — pushes your predicted lactate up or down, 
    making the AI’s decision fully transparent.
    </div>
    """, unsafe_allow_html=True)


                st.caption("Green bars increase the prediction; red bars decrease it. Length indicates strength of influence for the last window.")

            except Exception as e:
                st.error(f"SHAP computation failed: {e}")

st.markdown("---")
st.write("Notes: Demo with global and per-sample SHAP visualizations. Use training notebook to create `model.joblib` and `sample_live.csv`.")


import plotly.graph_objects as go

with tabs[3]:
    st.subheader("💓 HR Slope Trends")
    st.markdown("""
    **What it shows:**  
    The **heart rate slope** reveals how fast heart rate rises with effort.  
    A **steep slope** suggests fatigue or nearing lactate threshold, while a **flat slope** indicates good aerobic efficiency.
    """)

    if 'time' in df.columns and 'heart_rate' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['heart_rate'], mode='lines', name='Heart Rate (bpm)', line=dict(color='red')))
        if 'hr_slope_time' in df.columns:
            fig.add_trace(go.Scatter(x=df['time'], y=df['hr_slope_time'] * 10 + df['heart_rate'].mean(),
                                     mode='lines', name='HR Slope (scaled)', line=dict(color='blue')))
        if 'power' in df.columns:
            fig.add_trace(go.Scatter(x=df['time'], y=df['power'] / df['power'].max() * 50 + df['heart_rate'].min(),
                                     mode='lines', name='Power (scaled)', line=dict(color='green', dash='dot')))
        fig.update_layout(
            title='Heart Rate Slope Trends (Interactive)',
            xaxis_title='Time (s)',
            yaxis_title='Relative Value',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Heart rate data not found. Please load a valid session.")
