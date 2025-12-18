# pages/1_Basic_Metrics.py
import streamlit as st
import pandas as pd
from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics2

st.title("Basic Metrics")

# try to get df from session or disk
df = load_data_from_disk_or_session()
if df is None:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
else:
    df = compute_basic_metrics2(df)
    
    st.subheader("Accuracy")
    " From a user's test scores (marks) and number of questions and number of attempts, we can measure a user's;"
    "1. Personal Accuracy"
    "2. Test Accuracy"
    "We can measure a users personal accuracy by:"
    st.info("Attempted Accuracy = correct answers / attempted questions")
    "We can also measure test accuracy (total accuracy):"
    st.info("Total Accuracy = correct answers / total questions")

    st.dataframe(df[["user_id", "test_id", "accuracy_attempt", "accuracy_total"]].head())

    st.subheader("Time")
    st.write("We can also treat time as a unit in itself and determine metrics with regards to time only and also a combination with accuracy to deduce speed.")
    st.info("1. Raw time = time taken")
    st.info("2. Time Consumed = time taken / test duration")
    st.info("3. Relative Time (speed_rel_time) = time remaining / test duration")

    st.subheader("Speed")
    st.info("In physics: 	Speed = Distance / Time​")
    
    "In our context:"
    
    "“Distance” ≈ number of questions attempted (or correctly answered, depending on the variant)"
    "“Time” ≈ time taken"
    "Thus, we can define several relevant speed metrics."

    st.info("1. Raw Speed = attempted questions / time taken")
    st.info("2. Accurate Speed (adjusted speed) = correct answers / time taken")
    st.info("3. speed_marks = marks / time taken")


    st.subheader("Latency (optional)")
    st.write("We can check minutes per question or answers as opposed to questions per minute as an alternative to speed.")
    st.write("This will help calculate the latency.")
    st.write("However, for this research we will focus on speed but note the latency definitions for future use cases")
    st.info("Raw Latency = time taken / atttempted questions")
    st.info("Accuracte Latency = time taken / correct answers")
    
    st.dataframe(df[["user_id", "test_id", "time_consumed", "speed_raw", "adj_speed", "speed_norm", "speed_rel_time" ]].head())

    st.subheader("Distributions of Accuracy, Time consumption & Raw Speed")
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='accuracy_total', nbins=30, title='Accuracy distribution')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x='time_consumed', nbins=30, title='Time Consumption (in minutes)')
        st.plotly_chart(fig2, use_container_width=True)
   # with c3:
    fig3 = px.histogram(df, x ='speed_raw', nbins=30, title='Speed based on Attempted Questions')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Accuracy to Speed Ratio")

    "From deriving and calculating a user's accuracy and speed we can define a relationship between both in a ratio"
    "Our Accurate speed definition is already a form of accuracy-speed ratio since we are taking correct answers over the time taken"
    st.info("Accurate Speed = correct answers / time taken")

    "We can also look at the effeciency ratio which measure accuracy over time consumed"
    st.info("Effeciency Ratio = test accuracy / time consumed")

    df['accurate_speed'] = df['adj_speed']
    st.dataframe(df[["user_id", "test_id", "accurate_speed", "efficiency_ratio"]].head())

    st.subheader("Distributions of Speed-Accuracy Ratios")

    c4, c5 = st.columns(2)
    with c4:
        fig4 = px.histogram(df, x='accurate_speed', nbins=30, title='Correct Answers by Time Taken Distribution')
        st.plotly_chart(fig, use_container_width=True)
    with c5:
        fig5 = px.histogram(df, x='effeciency_ratio', nbins=30, title='Learner Effeciency Scores - Accuracy by time Consumption')
        st.plotly_chart(fig2, use_container_width=True)

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
    "Now that we know the averages we can calculate the relative avergae of a user"
    "This shows us how a user is performing individually amongst the group/population"
    st.info("Rel Avg = test accuracy - test average")
    df["accuracy_avg"] = df["accuracy_total"].mean()
    df["rel_acc"] = df["accuracy_total"] - df["accuracy_avg"]

    st.subheader("Distributions of Relative Accuracy & Relative Speed")
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='rel_acc', nbins=30, title='Relative Accuracy')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x='speed_rel_time', nbins=30, title='Relative Speed (in minutes)')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Standard Deviations / Variablilty")
    "We find out how different users/tests are by calculating the standard deviations"
    " This gives us the variability in the results and can be related to consistency."
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Std accuracy", f"{df['accuracy_total'].std():.3f}")
    col2.metric("Std accurate speed (correct/min)", f"{df['adj_speed'].std():.3f}")
    col3.metric("Std time consumed", f"{df['time_consumed'].std():.3f}")
    col4.metric("Std time taken", f"{df['time_taken'].std():.3f}")
    col5.metric("Std efficiency", f"{df['efficiency_ratio'].std():.3f}")
   





   
