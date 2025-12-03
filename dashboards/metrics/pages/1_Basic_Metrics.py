# pages/1_Basic_Metrics.py
import streamlit as st
import pandas as pd
from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics2

st.title("1 — Basic Metrics")

# try to get df from session or disk
df = load_data_from_disk_or_session()
if df is None:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
else:
    df = compute_basic_metrics2(df)
    st.subheader("Preview (first 10 rows)")
    st.dataframe(df.head(10))

    st.subheader("KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean accuracy", f"{df['accuracy_total'].mean():.3f}")
    col2.metric("Mean adj speed (correct/min)", f"{df['adj_speed'].mean():.3f}")
    col3.metric("Mean time consumed", f"{df['time_consumed'].mean():.3f}")
    
    st.subheader("Distributions")
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='accuracy_total', nbins=30, title='Accuracy distribution')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x='adj_speed', nbins=30, title='Adjusted speed (correct/min)')
        st.plotly_chart(fig2, use_container_width=True)

#import streamlit as st
#import pandas as pd
#from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics

st.title("Basic Metrics (User-Level Summary)")

df = load_data_from_disk_or_session()
if df is None:
    st.warning("No dataset loaded. Upload in sidebar or ensure data/verify_df_fixed.csv exists.")
    st.stop()

# compute row-level metrics
df = compute_basic_metrics2(df)

required_cols = ["user_id", "name"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Aggregate metrics by user
agg_df = df.groupby(["user_id", "name"]).agg({
    "accuracy_total": "mean",
    "adj_speed": "mean",
    "speed_raw": "mean",
    "speed_marks": "mean",
    "time_consumed": "mean",
    "efficiency_ratio": "mean",
    "attempted_questions": "sum",
    "correct_answers": "sum",
    "marks": "sum",
    "test_id": "nunique"
}).reset_index()

agg_df.rename(columns={
    "test_id": "tests_taken"
}, inplace=True)

st.subheader("User-Level Basic Metrics")
st.dataframe(agg_df)

# Optional: KPIs summary
st.subheader("Global KPIs")

col1, col2, col3 = st.columns(3)
col1.metric("Avg. Accuracy", f"{agg_df['accuracy_total'].mean():.3f}")
col2.metric("Avg. Adjusted Speed", f"{agg_df['speed_acc_raw'].mean():.3f}")
col3.metric("Avg. Efficiency", f"{agg_df['efficiency_ratio'].mean():.3f}")

st.info("This table aggregates all basic behavioral metrics at the user level.")

   
