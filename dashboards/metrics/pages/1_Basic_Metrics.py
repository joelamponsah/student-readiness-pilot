# pages/1_Basic_Metrics.py
import streamlit as st
import pandas as pd
from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics

st.title("1 — Basic Metrics")

# try to get df from session or disk
df = load_data_from_disk_or_session()
if df is None:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
else:
    df = compute_basic_metrics(df)
    st.subheader("Preview (first 10 rows)")
    st.dataframe(df[['accuray_total', 'speed_raw', 'speed_acc_raw', 'time_consumed']].head(10))

    st.subheader("KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean accuracy", f"{df['accuracy_total'].mean():.3f}")
    col2.metric("Mean adj speed (correct/min)", f"{df['speed_acc_raw'].mean():.3f}")
    col3.metric("Mean time consumed", f"{df['time_consumed'].mean():.3f}")

    st.subheader("Distributions")
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='accuracy_total', nbins=30, title='Accuracy distribution')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x='speed_acc_raw', nbins=30, title='Adjusted speed (correct/min)')
        st.plotly_chart(fig2, use_container_width=True)
