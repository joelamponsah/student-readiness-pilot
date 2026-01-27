# pages/1_Basic_Metrics.py
import streamlit as st
import pandas as pd
from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics

st.title("Basic Metrics")

# try to get df from session or disk
df = load_data_from_disk_or_session()
if df is None:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
else:
    df = compute_basic_metrics(df)
    
    st.subheader("Accuracy")
    " From a user's test scores (marks) and number of questions and number of attempts, we can measure a user's;"
    "1. Personal Accuracy"
    "2. Test Accuracy"
    "We can measure a users personal accuracy by:"
    st.info("Attempted Accuracy = correct answers / attempted questions")
    "We can also measure test accuracy (total accuracy):"
    st.info("Total Accuracy = correct answers / total questions")

    "We can also check a standard readinsess score by taking the normalized accuracy which is mark of the pass mark"
    st.info("Readiness Score = mark / pass mark")

    st.dataframe(df[["user_id", "test_id", "accuracy_attempt", "accuracy", "readiness_score"]].head())

    st.subheader("Time")
    st.write("We can also treat time as a unit in itself and determine metrics with regards to time only and also a combination with accuracy to deduce speed.")
    st.info("1. Raw time = time taken")
    st.info("2. Time Used = time taken / test duration")
    st.info("3. Time Left = time remaining / test duration")

    st.subheader("Speed")
    st.info("In physics: 	Speed = Distance / Time​")
    
    "In our context:"
    
    "“Distance” ≈ number of questions attempted (or correctly answered, depending on the variant)"
    "“Time” ≈ time taken"
    "Thus, we can define several relevant speed metrics."

    st.info("1. Raw Speed = attempted questions / time taken")
    st.info("2. Accurate Speed (adjusted speed) = correct answers / time taken")
    st.info("3. speed_marks = marks / time taken")

    "for learner behaviour we will use speed marks as our main speed metric"


    st.subheader("Latency (optional)")
    st.write("We can check minutes per question or answers as opposed to questions per minute as an alternative to speed.")
    st.write("This will help calculate the latency.")
    st.write("However, for this research we will focus on speed but note the latency definitions for future use cases")
    st.info("Raw Latency = time taken / atttempted questions")
    st.info("Accuracte Latency = time taken / correct answers")
    
    st.dataframe(df[["user_id", "test_id", "time_used", "time_left", "speed_attempt", "speed" ]].head())

    st.subheader("Distributions of Accuracy, Time consumption & Raw Speed")
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='accuracy', nbins=30, title='Accuracy distribution')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x='time_used', nbins=30, title='Time Consumption (in minutes)')
        st.plotly_chart(fig2, use_container_width=True)
   # with c3:
        fig3 = px.histogram(df, x ='speed', nbins=30, title='Speed based on Marks')
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Accuracy to Speed Ratio")

    "From deriving and calculating a user's accuracy and speed we can define a relationship between both in a ratio"
    "Our Accurate speed definition is already a form of accuracy-speed ratio since we are taking correct answers over the time taken"
    st.info("Accurate Speed = correct answers / time taken")

    "We can also look at the effeciency ratio which measure accuracy over time consumed"
    st.info("Effeciency = test accuracy / time consumed")

    df['accurate_speed'] = df['speed_answer']
    st.dataframe(df[["user_id", "test_id", "accurate_speed", "efficiency"]].head())

    st.subheader("Distributions of Speed-Accuracy Ratios")
    import plotly.express as px

    c4, c5 = st.columns(2)
    with c4:
        fig4 = px.histogram(df, x='accurate_speed', nbins=50, title='Correct Answers by Time Taken Distribution')
        st.plotly_chart(fig4, use_container_width=True)
    with c5:
        fig5 = px.histogram(df, x='efficiency', nbins=50, title='Efficieny - Accuracy by time Consumption')
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Mean / Averages ")
    "We can now take the averages (mean) of our prime metrics and further derive more advanced metrics"
    "This shows us how the population is performing as a whole"
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean accuracy", f"{df['accuracy'].mean():.3f}")
    col2.metric("Mean speed (marks/min)", f"{df['speed'].mean():.3f}")
    col3.metric("Mean time consumed", f"{df['time_used'].mean():.3f}")
    col4.metric("Mean time taken", f"{df['time_taken'].mean():.3f}")
    col5.metric("Mean efficiency", f"{df['efficiency'].mean():.3f}")

    st.subheader("Relative Average")
    "Now that we know the averages we can calculate the relative average of a user"
    "This shows us how a user is performing individually amongst the group/population"
    st.info("Rel Avg = test accuracy - test average")
    df["accuracy_avg"] = df["accuracy"].mean()
    df["rel_acc"] = df["accuracy"] - df["accuracy_avg"]

    st.subheader("Distributions of Relative Accuracy & Relative Speed")
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='rel_acc', nbins=30, title='Relative Accuracy')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(df, x='time_left', nbins=30, title='Time Remaining (in minutes)')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Standard Deviations / Variablilty")
    "We find out how different users/tests are by calculating the standard deviations"
    " This gives us the variability in the results and can be related to consistency."
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Std accuracy", f"{df['accuracy'].std():.3f}")
    col2.metric("Std speed (marks/min)", f"{df['speed'].std():.3f}")
    col3.metric("Std time used", f"{df['time_used'].std():.3f}")
    col4.metric("Std time taken", f"{df['time_taken'].std():.3f}")
    col5.metric("Std efficiency", f"{df['efficiency'].std():.3f}")

#user_metrics['avg_efficiency_percent'] = user_metrics['avg_efficiency'].apply(lambda x: f"{x:.2%}")








   
