import streamlit as st
import pandas as pd
import plotly.express as px
from utils.insights import apply_insight_engine
from utils.metrics import (
    load_data_from_disk_or_session,
    compute_basic_metrics2,
    compute_sab_behavioral,
    compute_test_analytics,
    compute_difficulty_df
)
#from utils.institute_standardization import standardize_institute

st.set_page_config(page_title="Institute Performance", layout="wide")
st.title("Institute Performance Summary")

# ---------------------------------------------------
# Load & Compute
# ---------------------------------------------------
df = load_data_from_disk_or_session()
if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

#df = standardize_institute(
#    df=df,
 #   column='institute',
#    mapping_path='data/mapping.csv'
#)

#assert df['institute_std'].isna().sum() == 0, "Null institutes after standardization!"
#assert len(df) > 0, "Fact table is empty!"

df = compute_basic_metrics2(df)
sab_df = compute_sab_behavioral(df)
sab_df = apply_insight_engine(sab_df)
test_df = compute_test_analytics(df)

if "institute_standardized" not in df.columns:
    st.error("Missing `institute_name` column.")
    st.stop()

# ---------------------------------------------------
# Institute Selector
# ---------------------------------------------------
institutes = sorted(sab_df["institute_standardized"].dropna().unique())
institute = st.selectbox("Select Institute", institutes)

inst_df = sab_df[sab_df["institute_standardized"] == institute]
inst_users = sab_df[sab_df["user_id"].isin(inst_df["user_id"])]

# ---------------------------------------------------
# KPI METRICS
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("👥 Learners", inst_users["user_id"].nunique())
col2.metric("🧪 Unique Tests", inst_df["test_id"].nunique())
col3.metric("📊 Total Attempts", len(inst_df))

#col4, col5, col6 = st.columns(3)

col1.metric("🎯 Avg Accuracy", f"{inst_df['accuracy_total'].mean():.2f}")
col2.metric("Avg Speed", f"{inst_df['speed_raw'].mean():.2f}")
col3.metric("🧠 Avg Readiness (Robust SAB)", f"{inst_users['robust_SAB_scaled'].mean():.1f}")

at_risk = inst_users[inst_users["exam_status"] == "Not Eligible"]
non_risk = inst_users[inst_users["exam_status"] == "Conditionally Eligible"]
ready = inst_users[inst_users["exam_status"] == "Eligible"]
#st.metric
col1.metric("⚠️ At-Risk Learners", len(at_risk))
col2.metric("Non-risk Learners", len(non_risk))
col3.metric("Ready Learners", len(ready))

st.divider()

st.subheader("📊 Institute Readiness Summary")

eligible_pct = (inst_users["exam_status"] == "Eligible").mean() * 100
near_ready_pct = (inst_users["insight_code"] == "NEAR_READY").mean() * 100
at_risk_pct = (inst_users["exam_status"] == "Not Eligible").mean() * 100

st.markdown(
    f"""
    **{eligible_pct:.1f}%** of learners meet exam eligibility criteria.  
    **{near_ready_pct:.1f}%** are approaching readiness with targeted support.  
    **{at_risk_pct:.1f}%** require foundational intervention before exam attempts.
    """
)
st.divider()

st.subheader("Readiness Distribution")

insight_dist = (
    inst_users["insight_code"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Insight", "insight_code": "Learners"})
)

#st.bar_chart(insight_dist.set_index("Insight"))
st.bar_chart(insight_dist)



# ---------------------------------------------------
# TOP PERFORMERS
# ---------------------------------------------------
st.subheader("Top Performers")



st.dataframe(
    ready[[
        "user_id", "mean_accuracy", "mean_speed",
        "test_count", "robust_SAB_scaled"
    ]],
    use_container_width=True
)

# ---------------------------------------------------
# AT-RISK USERS
# ---------------------------------------------------

st.write("Select based on exam_status")
selected_status = st.multiselect(
    "Filter by Exam Status",
    inst_users["exam_status"].unique(),
    default=inst_users["exam_status"].unique()
)

filtered = inst_users[inst_users["exam_status"].isin(selected_status)]

if filtered.empty:
    st.success('No learners detected')
else:
    st.dataframe(
        filtered[[
            "user_id", "test_count", "speed_consistency", 
            "accuracy_consistency", "robust_SAB_scaled", "exam_status"
        ]],
        use_container_width=True )
#at_risk = inst_users[
 #   (inst_users["robust_SAB_scaled"] < 40) &
  #  (inst_users["test_count"] >= 5)
#].sort_values("robust_SAB_scaled")

#if at_risk.empty:
 #   st.success("No at-risk learners detected 🎉")
#else:
 #   st.dataframe(
  #      at_risk[[
   #         "user_id", "mean_accuracy",
    #        "speed_consistency", "accuracy_consistency",
     #       "test_count", "robust_SAB_scaled"
      #  ]],
       # use_container_width=True
    #)

# ---------------------------------------------------
# TEST STABILITY INSIGHTS
# ---------------------------------------------------
st.subheader("Test Stability & Difficulty")

inst_tests = test_df[test_df["test_id"].isin(inst_df["test_id"])]

fig = px.scatter(
    inst_tests,
    x="mean_accuracy",
    y="speed_consistency",
    size="taker_count",
    color="time_consistency",
    hover_data=["test_id"],
    title="Test Stability Map"
)

st.plotly_chart(fig, use_container_width=True)
