import pandas as pd
import plotly.express as px
import streamlit as st
import os

def show_training_log_dashboard():

    LOG_PATH = "models/training_log.csv"

    if not os.path.exists(LOG_PATH):
        st.info("No training history found yet. Train models to generate logs.")
        return

    df = pd.read_csv(LOG_PATH)

    st.subheader("📈 Training History (Model Performance Over Time)")
    st.caption("Tracks R² and MAE for both lactate and recovery models.")

    # --- R² Scores ---
    try:
        fig_r2 = px.line(
            df,
            x="timestamp",
            y=["r2_lactate", "r2_recovery"],
            labels={"value": "R² Score", "timestamp": "Training Timestamp"},
            title="R² History (Lactate vs Recovery Models)",
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot R² history: {e}")

    # --- MAE Scores ---
    try:
        fig_mae = px.line(
            df,
            x="timestamp",
            y=["mae_lactate", "mae_recovery"],
            labels={"value": "MAE", "timestamp": "Training Timestamp"},
            title="MAE History (Lactate vs Recovery Models)",
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot MAE history: {e}")

    # --- Dataset Size ---
    if "rows" in df.columns:
        st.subheader("📊 Dataset Size Over Time")
        fig_rows = px.line(
            df,
            x="timestamp",
            y="rows",
            labels={"rows": "Total Rows Used"},
            title="Dataset Growth",
        )
        st.plotly_chart(fig_rows, use_container_width=True)
