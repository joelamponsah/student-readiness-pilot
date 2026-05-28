import pandas as pd
import streamlit as st

from utils.metrics import load_data_from_disk_or_session, save_uploaded_df


st.set_page_config(page_title="Home - Test / Exercise Readiness", layout="wide")

st.sidebar.title("Home")
st.sidebar.markdown("## Data")
uploaded = st.sidebar.file_uploader(
    "Load verify_df_fixed.csv",
    type=["csv"],
    help="Upload the processed dataset used by the dashboard pages.",
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    save_uploaded_df(df, path="data/verify_df_fixed.csv")
    st.session_state["df"] = df
    st.sidebar.success("Loaded data and saved it to data/verify_df_fixed.csv")
else:
    df = load_data_from_disk_or_session(default_path="data/verify_df_fixed.csv")
    if df is not None:
        st.session_state["df"] = df

st.title("Home")
st.caption(
    "v1.3 is a Learn Smarter-aligned Test / Exercise Readiness release. "
    "It keeps the dashboard focused on DQ-gated evidence, proxy readiness metrics, and maintained summary pages."
)

left, right = st.columns([1.3, 1])

with left:
    st.subheader("Project scope")
    st.markdown(
        "- DQ-gated learner and institute views\n"
        "- Core metrics for accuracy, speed, pass outcomes, and readiness\n"
        "- v1.3 proxy metrics: Inferred BLS Proxy, Current ALS Proxy, Potential ALS Proxy, and CAS Proxy\n"
        "- Standardized mappings for schools and B2B banks/rural banks\n"
        "- Maintained pages for DQ Monitors, Basic Metrics, Ranking & Leaderboard, User Summary, and Institute Summary"
    )

with right:
    st.subheader("Open pages")
    page_links = [
        ("DQ Monitors", "pages/0_DQ_Monitors.py"),
        ("Basic Metrics", "pages/1_Metrics.py"),
        ("Ranking & Leaderboard", "pages/4_Ranking_and_Leaderboard.py"),
        ("User Summary", "pages/7_User_Summary.py"),
        ("Institute Summary", "pages/Institute_Summary.py"),
    ]
    for label, page in page_links:
        st.page_link(page, label=label)

st.divider()

if "df" in st.session_state and st.session_state["df"] is not None:
    current = st.session_state["df"]
    st.subheader("Current dataset")
    stats = st.columns(4)
    stats[0].metric("Rows", f"{len(current):,}")
    stats[1].metric("Columns", f"{len(current.columns):,}")
    stats[2].metric("Users", f"{current['user_id'].nunique():,}" if "user_id" in current.columns else "N/A")
    stats[3].metric("Tests", f"{current['test_id'].nunique():,}" if "test_id" in current.columns else "N/A")
    st.dataframe(current.head(5), use_container_width=True)
else:
    st.info("Upload `verify_df_fixed.csv` in the sidebar or place it at `data/verify_df_fixed.csv`.")
