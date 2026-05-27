import pandas as pd
import streamlit as st

from utils.metrics import load_data_from_disk_or_session, save_uploaded_df


st.set_page_config(page_title="Test / Exercise Readiness Dashboard", layout="wide")
st.sidebar.title("Test / Exercise Readiness")

st.sidebar.markdown("## Data")
uploaded = st.sidebar.file_uploader("Upload processed verify_df_fixed.csv (optional)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    save_uploaded_df(df, path="data/verify_df_fixed.csv")
    st.session_state["df"] = df
    st.sidebar.success("Uploaded and saved to data/verify_df_fixed.csv")
else:
    df = load_data_from_disk_or_session(default_path="data/verify_df_fixed.csv")
    if df is not None:
        st.session_state["df"] = df

st.title("v1.3 Test / Exercise Readiness")
st.caption(
    "Bridge release: Learn Smarter-aligned test/exercise readiness on top of the trusted v1.2 DQ baseline. "
    "This is not the full Learn Smarter model build."
)
st.write(
    "Use this hub to move between the maintained DQ, metrics, ranking, learner, and institute views. "
    "The page set is intentionally small and aligned to the v1.3 release surface."
)

st.subheader("Open Pages")
page_cols = st.columns(5)
page_links = [
    ("DQ Monitors", "pages/0_DQ_Monitors.py"),
    ("Basic Metrics", "pages/1_Basic_Metrics.py"),
    ("Ranking & Leaderboard", "pages/4_Ranking_and_Leaderboard.py"),
    ("User Summary", "pages/7_User_Summary.py"),
    ("Institute Summary", "pages/Institute_Summary.py"),
]
for col, (label, page) in zip(page_cols, page_links):
    with col:
        st.page_link(page, label=label)

if "df" in st.session_state and st.session_state["df"] is not None:
    current = st.session_state["df"]
    st.subheader("Current Dataset")
    stats = st.columns(4)
    stats[0].metric("Rows", f"{len(current):,}")
    stats[1].metric("Columns", f"{len(current.columns):,}")
    stats[2].metric("Users", f"{current['user_id'].nunique():,}" if "user_id" in current.columns else "N/A")
    stats[3].metric("Tests", f"{current['test_id'].nunique():,}" if "test_id" in current.columns else "N/A")
    st.dataframe(current.head(5), use_container_width=True)
else:
    st.info("No dataset loaded. Upload a processed verify_df_fixed.csv in the sidebar or place it at data/verify_df_fixed.csv in the repo.")
