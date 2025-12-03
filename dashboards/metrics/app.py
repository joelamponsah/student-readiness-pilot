import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Readiness Dashboard",
                   layout="wide",
                   page_icon="📚")

st.title("📚 Student Readiness Analytics Dashboard")

st.write("""
Welcome to the Student Readiness Dashboard.

Use the sidebar to navigate through:
- Basic accuracy & speed metrics  
- Test & topic performance trends  
- Difficulty, DCI & test stability  
- SAB index & leaderboard  
- Exam readiness predictive model  
""")

# Data loader
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/verify_df_fixed.csv")
        return df
    except:
        st.warning("Upload your dataset in the sidebar to proceed.")
        return None

with st.sidebar:
    st.header("📥 Data Upload")
    uploaded = st.file_uploader("Upload processed dataset", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Data loaded successfully!")
        df.to_csv("data/verify_df_fixed.csv", index=False)
    else:
        df = load_data()

if df is not None:
    st.subheader("Preview Data")
    st.dataframe(df.head())
