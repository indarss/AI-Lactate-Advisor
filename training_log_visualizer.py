import pandas as pd
import streamlit as st
import plotly.express as px

def show_training_log_dashboard():
    try:
        df = pd.read_csv("models/training_log.csv")
    except:
        st.warning("No training_log.csv found in models/")
        return

    st.subheader("📈 Training History")
    st.dataframe(df)

    # --- R² Chart ---
    fig_r2 = px.line(
        df,
        x="timestamp",
        y=["r2_lactate", "r2_recovery"],
        title="Model R² Over Time",
        markers=True
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    # --- MAE Chart ---
    fig_mae = px.line(
        df,
        x="timestamp",
        y=["mae_lactate", "mae_recovery"],
        title="Model MAE Over Time",
        markers=True
    )
    st.plotly_chart(fig_mae, use_container_width=True)
