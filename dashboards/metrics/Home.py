import pandas as pd
import streamlit as st

from utils.metrics import (
    detect_dataset_profile,
    is_attempt_level_dataset,
    load_data_from_disk_or_session,
    save_uploaded_df,
)


st.set_page_config(page_title="Home - Test / Exercise Readiness", layout="wide")

st.sidebar.title("Home")
st.sidebar.markdown("## Data")
uploaded = st.sidebar.file_uploader(
    "Load v1.3 CSV",
    type=["csv"],
    help=(
        "Upload an attempt-level file for dashboard pages. "
        "v13_user_test_readiness_summary.csv is a summary artifact and will be previewed here only."
    ),
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    profile = detect_dataset_profile(df)
    st.session_state["uploaded_profile"] = profile

    if is_attempt_level_dataset(df):
        save_uploaded_df(df, path="data/verify_df_fixed.csv")
        st.session_state["df"] = df
        st.sidebar.success("Loaded attempt-level data for dashboard pages.")
    elif profile == "v13_user_test_readiness_summary":
        save_uploaded_df(df, path="data/v13_user_test_readiness_summary.csv")
        st.session_state["v13_user_test_summary"] = df
        if "df" in st.session_state and not is_attempt_level_dataset(st.session_state["df"]):
            st.session_state["df"] = None
        st.sidebar.warning("Loaded summary artifact for Home preview only.")
    elif profile == "v13_group_readiness_summary":
        save_uploaded_df(df, path="data/v13_group_readiness_summary.csv")
        st.session_state["v13_group_summary"] = df
        st.sidebar.warning("Loaded group summary artifact for Home preview only.")
    elif profile == "smoke_report":
        save_uploaded_df(df, path="data/smoke_report.csv")
        st.session_state["v13_smoke_report"] = df
        st.sidebar.warning("Loaded smoke report for Home preview only.")
    else:
        st.sidebar.error(
            "Unrecognized CSV schema. Use proxy_sequence_attempts.csv or the raw attempt dataset for dashboard pages."
        )
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
    st.caption(f"Detected profile: `{detect_dataset_profile(current)}`")
    stats = st.columns(4)
    stats[0].metric("Rows", f"{len(current):,}")
    stats[1].metric("Columns", f"{len(current.columns):,}")
    stats[2].metric("Users", f"{current['user_id'].nunique():,}" if "user_id" in current.columns else "N/A")
    stats[3].metric("Tests", f"{current['test_id'].nunique():,}" if "test_id" in current.columns else "N/A")
    st.dataframe(current.head(5), use_container_width=True)
else:
    st.info(
        "Upload `proxy_sequence_attempts.csv` or an attempt-level raw extract for dashboard pages. "
        "`v13_user_test_readiness_summary.csv` is already aggregated and is previewed below when loaded."
    )

summary = st.session_state.get("v13_user_test_summary")
if summary is not None:
    st.subheader("Loaded v1.3 user-test summary artifact")
    st.caption(
        "This file is one row per learner/test after DQ/proxy processing. "
        "It is not an attempt-level input for DQ, Ranking, User Summary, or Institute Summary pages."
    )
    stats = st.columns(5)
    stats[0].metric("Rows", f"{len(summary):,}")
    stats[1].metric("Columns", f"{len(summary.columns):,}")
    stats[2].metric("Users", f"{summary['user_id'].nunique():,}" if "user_id" in summary.columns else "N/A")
    stats[3].metric("Tests", f"{summary['test_id'].nunique():,}" if "test_id" in summary.columns else "N/A")
    stats[4].metric(
        "Repeated groups",
        f"{int((pd.to_numeric(summary.get('attempt_count', 0), errors='coerce') >= 2).sum()):,}"
        if "attempt_count" in summary.columns else "N/A",
    )
    st.dataframe(summary.head(20), use_container_width=True)

group_summary = st.session_state.get("v13_group_summary")
if group_summary is not None:
    st.subheader("Loaded v1.3 group summary artifact")
    st.dataframe(group_summary.head(20), use_container_width=True)

smoke_report = st.session_state.get("v13_smoke_report")
if smoke_report is not None:
    st.subheader("Loaded v1.3 smoke report")
    st.dataframe(smoke_report, use_container_width=True)
