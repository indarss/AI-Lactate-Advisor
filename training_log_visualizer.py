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

    fig = px.line(
        df, x="timestamp", y=["lac_r2","rec_r2"],
        title="Model R² Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        df, x="timestamp", y=["lac_mae","rec_mae"],
        title="Model MAE Over Time"
    )
    st.plotly_chart(fig2, use_container_width=True)
