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
    
    st.subheader("Accuracy")
    " From a user's test scores (marks) and number of questions, we can measure a users;"
    "1. Personal Accuracy"
    "2. Test Accuracy"
    "We can measure a users personal accuracy by:"
    "Attempted Accuracy = correct answers / attempted questions"
    "We can also measure test accuracy (total accuracy)"
    "Total Accuracy = correct answers / total questions"

    st.dataframe(df[["user_id", "test_id", "accuracy_attempt", "accuracy_total"]].head())

    st.subheader("Speed")
    "In physics: 	Speed = Distance / Time​"
    
    "In our context:"
    
    "“Distance” ≈ number of questions attempted (or correctly answered, depending on the variant)"
    "“Time” ≈ time taken"
    "Thus, we can define several relevant speed metrics."

    "1. Raw Speed = attempted_questions / time taken"
    "2. Accurate Speed (adjusted speed) = correct_answers / time taken"
    "3. speed_marks = marks / time taken"
    "4. Relative Speed = time remaining / test duration"
    "5. Time Consumed = time taken / test duration"
    
    st.dataframe(df[["user_id", "test_id", "time_consumed", "speed_raw", "adj_speed", "speed_norm", "speed_rel_time" ]].head())

    st.subheader("Accuracy to Speed Ratio")

    "From deriving and calculating a user's accuracy ans speed we can that define the a relationship between both in a ratio"
    " our Accurate speed definition is already a form of accuracy-speed ratio since we are taking correct answers over the time taken"
    " Accurate Speed = correct answers / time taken"

    "We can also look at the effeciency ratio which measure accuracy over time consumed"
    "Effeciency Ratio = test accuracy / time consumed"

    df['accurate_speed'] = df['adj_speed']
    st.dataframe(df[["user_id", "test_id", "accurate_speed", "efficiency_ratio"]].head())

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

required_cols = ["user_id", "l_name"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Aggregate metrics by user
agg_df = df.groupby(["user_id", "l_name"]).agg({
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
col2.metric("Avg. Adjusted Speed", f"{agg_df['adj_speed'].mean():.3f}")
col3.metric("Avg. Efficiency", f"{agg_df['efficiency_ratio'].mean():.3f}")

st.info("This table aggregates all basic behavioral metrics at the user level.")

   
