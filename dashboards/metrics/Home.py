# app.py
import streamlit as st
import pandas as pd
from utils.metrics import get_v13_artifacts, load_data_from_disk_or_session, save_uploaded_df

st.set_page_config(page_title="Learner Readiness Dashboard", layout="wide")
st.sidebar.title("Learner Readiness")

st.sidebar.markdown("## Data")
uploaded = st.sidebar.file_uploader("Upload raw_attempts.csv (required dashboard input)", type=["csv"])

if uploaded is not None:
    # Save upload to session_state and to disk so all pages can use it.
    df = pd.read_csv(uploaded)
    save_uploaded_df(df, path="data/raw_attempts.csv")
    st.session_state['df'] = df
    st.sidebar.success("Uploaded and saved to data/raw_attempts.csv")
else:
    # Try to load existing file or session.
    df = load_data_from_disk_or_session(default_path="data/raw_attempts.csv")
    if df is not None:
        st.session_state['df'] = df

st.title("Learner Readiness Dashboard")
st.write("The dashboard expects a raw attempt-level file named `data/raw_attempts.csv`. Each row should represent one learner attempt on one test/exercise. Derived artifacts such as `verify_df_fixed.csv` and `proxy_sequence_attempts.csv` are legacy or downstream outputs, not the primary app input.")
st.write("Navigate through pages with the menu on the top-left. Use the sidebar to upload raw attempt data if you want to replace the default file.")

if 'df' in st.session_state and st.session_state['df'] is not None:
    st.success(f"Dataset loaded with {len(st.session_state['df']):,} rows and {len(st.session_state['df'].columns):,} columns.")
    st.dataframe(st.session_state['df'].head(5))

    st.subheader("v1.3 Shared Pipeline Check")
    try:
        raw_df, artifacts = get_v13_artifacts()
        if raw_df is None:
            st.warning("raw_attempts.csv could not be loaded, so the shared v1.3 pipeline did not run.")
        else:
            smoke = artifacts.get("smoke_report")
            st.write(
                f"Raw rows: {len(raw_df):,} | "
                f"users: {raw_df['user_id'].nunique():,} | "
                f"tests: {raw_df['test_id'].nunique():,}"
            )
            st.write(f"Artifact keys: {list(artifacts.keys())}")

            counts = [
                ("published_kpi", "published rows/users"),
                ("proxy_sequence", "proxy sequence rows/users"),
                ("user_test_summary", "user_test_summary rows/users"),
                ("readiness_user", "readiness_user rows/users"),
                ("difficulty_df", "difficulty rows/users"),
            ]
            for key, label in counts:
                frame = artifacts.get(key)
                if frame is not None and hasattr(frame, "shape"):
                    users = frame["user_id"].nunique() if "user_id" in frame.columns else "N/A"
                    tests = frame["test_id"].nunique() if "test_id" in frame.columns else "N/A"
                    st.write(f"{label}: {len(frame):,} rows | users: {users} | tests: {tests}")

            if smoke is not None:
                st.caption("Smoke report")
                st.dataframe(smoke.T, use_container_width=True)
            else:
                st.warning("Shared v1.3 pipeline ran, but no smoke_report was returned.")
    except Exception as exc:
        st.error("Shared v1.3 pipeline build failed.")
        st.exception(exc)
else:
    st.info("No dataset loaded. Upload raw_attempts.csv in the sidebar or place it at data/raw_attempts.csv in the repo. `verify_df_fixed.csv` is legacy/reference only.")
