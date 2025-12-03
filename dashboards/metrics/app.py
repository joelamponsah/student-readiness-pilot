# app.py
import streamlit as st
import pandas as pd
from utils.metrics import load_data_from_disk_or_session, save_uploaded_df

st.set_page_config(page_title="Student Readiness Dashboard", layout="wide")
st.sidebar.title("Student Readiness")

st.sidebar.markdown("## Data")
uploaded = st.sidebar.file_uploader("Upload processed verify_df_fixed.csv (optional)", type=["csv"])

if uploaded is not None:
    # Save upload to session_state and to disk so all pages can use it
    df = pd.read_csv(uploaded, low_memory=False)
    save_uploaded_df(df, path="data/verify_df_fixed.csv")
    st.session_state['df'] = df
    st.sidebar.success("Uploaded and saved to data/verify_df_fixed.csv")
else:
    # try to load existing file or session
    df = load_data_from_disk_or_session(default_path="data/verify_df_fixed.csv")
    if df is not None:
        st.session_state['df'] = df

st.title("Student Readiness Dashboard")
st.write("Use the sidebar to upload data (optional) and the top-left pages menu to navigate.")

if 'df' in st.session_state and st.session_state['df'] is not None:
    st.success(f"Dataset loaded with {len(st.session_state['df']):,} rows and {len(st.session_state['df'].columns):,} columns.")
    # quick preview + link to pages
    st.dataframe(st.session_state['df'].head(5))
else:
    st.info("No dataset loaded. Upload a processed verify_df_fixed.csv in the sidebar or place it at data/verify_df_fixed.csv in the repo.")
