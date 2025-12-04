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

    st.subheader("Distributions")
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='accuracy_total', nbins=30, title='Accuracy distribution')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x='adj_speed', nbins=30, title='Adjusted speed (correct/min)')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Accuracy to Speed Ratio")

    "From deriving and calculating a user's accuracy ans speed we can that define the a relationship between both in a ratio"
    " our Accurate speed definition is already a form of accuracy-speed ratio since we are taking correct answers over the time taken"
    " Accurate Speed = correct answers / time taken"

    "We can also look at the effeciency ratio which measure accuracy over time consumed"
    "Effeciency Ratio = test accuracy / time consumed"

    df['accurate_speed'] = df['adj_speed']
    st.dataframe(df[["user_id", "test_id", "accurate_speed", "efficiency_ratio"]].head())

    st.subheader("Mean / Averages ")
    "We can now take the averages (mean) of our prime metrics and further derive more advanced metrics"
    "This shows us how the population is performing as a whole"
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean accuracy", f"{df['accuracy_total'].mean():.3f}")
    col2.metric("Mean accurate speed (correct/min)", f"{df['adj_speed'].mean():.3f}")
    col3.metric("Mean time consumed", f"{df['time_consumed'].mean():.3f}")
    col4.metric("Mean time taken", f"{df['time_taken'].mean():.3f}")
    col5.metric("Mean efficiency", f"{df['efficiency_ratio'].mean():.3f}")

    st.subheader("Relative Average")
    "Now that we now the averages we can calculate ther relative avergae of a user"
    "This shows us how a user is performing individuallu amongst the group/population"
    "Rel Avg = test accuracy - test average"
    
    st.subheader("Standard Deviations / Variablilty")
    "We find out how different users/tests are by calculating the standard deviations"
    " This guves us the variability in the results and can be related to consistency."
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Std accuracy", f"{df['accuracy_total'].std():.3f}")
    col2.metric("Std accurate speed (correct/min)", f"{df['adj_speed'].std():.3f}")
    col3.metric("Std time consumed", f"{df['time_consumed'].std():.3f}")
    col4.metric("Std time taken", f"{df['time_taken'].std():.3f}")
    col5.metric("Std efficiency", f"{df['efficiency_ratio'].std():.3f}")
   





   
