import streamlit as st

from utils.artifact_loader import load_artifact
from utils.ui_helpers import optional_filter


st.set_page_config(page_title="Learner Summary", layout="wide")
st.title("Learner Summary")
st.caption("Learner-level readiness evidence. One-attempt learners are included and flagged as low/repeat evidence where available.")

summary = load_artifact("learner_readiness_summary")
readiness = load_artifact("readiness_signals")
gain = load_artifact("learning_gain_signals")
habits = load_artifact("work_habits_signals")

if summary is None and readiness is None:
    st.warning("Missing learner readiness artifacts.")
    st.stop()

df = summary if summary is not None else readiness
if readiness is not None and summary is not None and "user_id" in summary.columns and "user_id" in readiness.columns:
    extra = [c for c in readiness.columns if c not in summary.columns or c == "user_id"]
    df = summary.merge(readiness[extra], on="user_id", how="left")

with st.sidebar:
    for col in ["institute_std", "readiness_band", "evidence_level", "class_id", "class_name"]:
        df = optional_filter(df, col, col)

cols = st.columns(4)
cols[0].metric("Learners", df["user_id"].nunique() if "user_id" in df.columns else len(df))
cols[1].metric("Avg readiness", f"{df['readiness_score_pct'].mean():.1f}%" if "readiness_score_pct" in df.columns and not df.empty else "N/A")
cols[2].metric("Avg score", f"{df['avg_score_pct'].mean():.1f}%" if "avg_score_pct" in df.columns and not df.empty else "N/A")
cols[3].metric("Avg completion", f"{df['avg_completion_pct'].mean():.1f}%" if "avg_completion_pct" in df.columns and not df.empty else "N/A")

st.dataframe(df, use_container_width=True)

if gain is not None:
    with st.expander("Learning Gain Signals"):
        st.dataframe(gain, use_container_width=True)
if habits is not None:
    with st.expander("Work Habits Signals"):
        st.dataframe(habits, use_container_width=True)

