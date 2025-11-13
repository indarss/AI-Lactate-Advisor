# ==============================================
# 📈 Training Log Visualizer (Streamlit)
# ==============================================
import os
import pandas as pd
import plotly.express as px
import streamlit as st

def show_training_log_dashboard():
    """
    Display model performance trends from models/training_log.csv.
    """
    st.title("📈 Model Training History")
    st.caption("Tracks R² and MAE metrics across all retraining sessions.")

    log_path = os.path.join("models", "training_log.csv")

    if not os.path.exists(log_path):
        st.error("No training log found. Please retrain a model first.")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        st.warning("Training log is empty. No runs recorded yet.")
        return

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    # --- R² Trend ---
    fig_r2 = px.line(
        df, x="timestamp", y="r2_test", color="model",
        title="R² Score Trend", markers=True,
        labels={"r2_test": "R² (Test)", "timestamp": "Time"}
    )
    fig_r2.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_r2, use_container_width=True)

    # --- MAE Trend ---
    fig_mae = px.line(
        df, x="timestamp", y="mae_test", color="model",
        title="MAE Trend", markers=True,
        labels={"mae_test": "MAE (Test)", "timestamp": "Time"}
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    # --- Best Model Summary ---
    st.markdown("### 🏆 Best Performance Summary")
    best = df.loc[df.groupby("model")["r2_test"].idxmax()]
    st.dataframe(best[["model", "timestamp", "r2_test", "mae_test", "n_test"]])

if __name__ == "__main__":
    st.set_page_config(page_title="Training Log Visualizer", page_icon="📈", layout="wide")
    show_training_log_dashboard()
