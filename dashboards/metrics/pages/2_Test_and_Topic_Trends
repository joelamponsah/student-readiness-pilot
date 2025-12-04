import streamlit as st
import pandas as pd
from utils.plotting import line_plot

st.title("📈 Test & Topic Trends")

df = pd.read_csv("data/verify_df_fixed.csv")

trend = df.groupby("test_id").agg(
    accuracy=("accuracy_total", "mean"),
    speed=("speed_acc_raw", "mean"),
    avg_time=("time_taken", "mean"),
    attempts=("test_id", "count")
).reset_index()


st.plotly_chart(line_plot(trend, "test_id", "accuracy", title="Accuracy Trend"))
st.plotly_chart(line_plot(trend, "test_id", "speed", title="Speed Trend"))
