import streamlit as st
from utils.metrics import load_data_from_disk_or_session, compute_difficulty_df

st.title("Test Difficulty & Stability & The Difficulty Consistency Index (DCI)")

df = load_data_from_disk_or_session()

if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

# ------------------------------------------------
# 🔍 Filters
# ------------------------------------------------
test_filter = st.sidebar.multiselect(
    "Filter by Test",
    options=sorted(df["name"].unique())
)

if test_filter:
    df = df[df["name"].isin(test_filter)]

st.subheader('Test Difficulty')
"We factor in how difficult a test is to see whether it affects accuracy and speed"
"Therefore, every test is rated based on its diffiuclty."
difficulty_df = compute_difficulty_df(df)

st.dataframe(difficulty_df.groupby('difficulty_label')['difficulty'].agg(['min', 'max', 'mean', 'count']))

st.subheader("Test Consistency")
"We looked at the consistency of a test by checking how far test takers were from the average marks." 
"That is the standard deviation of accuracy of the tests." 
"If a test has high standard deviation (test consistency) it means the test has a low consistency rating and vice versa."

st.subheader("Difficulty Consistency Index (DCI)")
"A good test isn’t just about being hard or easy — it’s about consistency. "
"If two tests have the same pass ratio (say both “hard”)," 
"but one produces a wide spread of scores while the other produces tightly clustered ones, the second is more reliable."
"So we want a single metric that reflects: How hard and consistent a test is in evaluating skill."
"Using a combination of the average accuracy per test and the test consistency we label the each test"

st.subheader("DCI = average accuracy X test Consistency")

st.subheader('Stability')
"Based on the DCI we can categorize tests based on stability"


st.subheader("Per-Test Difficulty & DCI Metrics")
st.dataframe(difficulty_df, use_container_width=True)

csv = difficulty_df.to_csv(index=False)
st.download_button("Download Difficulty & DCI CSV", csv, "difficulty_dci.csv")
