# pages/1_Basic_Metrics.py

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics2


# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def safe_mean(df, col):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.mean() if not s.empty else None


def safe_std(df, col):
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s.std() if len(s) > 1 else 0


def fmt_num(x, decimals=3, fallback="0.000"):
    if x is None or pd.isna(x):
        return fallback
    return f"{x:.{decimals}f}"


def fmt_pct(x, decimals=1, fallback="0.0%"):
    if x is None or pd.isna(x):
        return fallback
    return f"{x * 100:.{decimals}f}%"


# ------------------------------------------------
# Page setup
# ------------------------------------------------
st.title("Basic Metrics & Formula Guide")

st.markdown(
    """
This page explains the **core formulas and assumptions** used across the Learner Readiness Dashboard.

The purpose is to help testers understand what the dashboard is measuring before reviewing the learner, test, and institute-level pages.
"""
)

st.info(
    """
**Showcase note:** This is a proof-of-concept version.  
The metrics below are useful learning-behaviour signals, but they are not yet a fully validated exam prediction model.
"""
)


# ------------------------------------------------
# Load data
# ------------------------------------------------
df = load_data_from_disk_or_session()

if df is None or df.empty:
    st.warning("No dataset loaded. Upload data in the sidebar or add `data/verify_df_fixed.csv`.")
    st.stop()

df = compute_basic_metrics2(df)


# ------------------------------------------------
# Showcase-safe metric cleanup / fallbacks
# ------------------------------------------------

# Convert key numeric columns safely
numeric_cols = [
    "marks",
    "no_of_questions",
    "time_taken",
    "accuracy_attempt",
    "accuracy_total",
    "time_consumed",
    "speed_raw",
    "adj_speed",
    "speed_norm",
    "speed_rel_time",
    "efficiency_ratio",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinite values
df = df.replace([float("inf"), float("-inf")], pd.NA)

# Safe fallback for accuracy_total
if "accuracy_total" not in df.columns or df["accuracy_total"].notna().sum() == 0:
    if "marks" in df.columns and "no_of_questions" in df.columns:
        noq = pd.to_numeric(df["no_of_questions"], errors="coerce").replace(0, pd.NA)
        marks = pd.to_numeric(df["marks"], errors="coerce")
        df["accuracy_total"] = marks / noq

# Safe fallback for time_consumed
# Preferred formula is: time_taken / test duration.
# If duration is unavailable, use a normalized proxy: time_taken / average time_taken.
if "time_consumed" not in df.columns or df["time_consumed"].notna().sum() == 0:
    if "time_taken" in df.columns and df["time_taken"].notna().sum() > 0:
        clean_time = pd.to_numeric(df["time_taken"], errors="coerce").replace(0, pd.NA)
        avg_time_taken = clean_time.mean()

        if pd.notna(avg_time_taken) and avg_time_taken > 0:
            df["time_consumed"] = clean_time / avg_time_taken
        else:
            df["time_consumed"] = pd.NA

# Clean invalid time_consumed values
if "time_consumed" in df.columns:
    df["time_consumed"] = pd.to_numeric(df["time_consumed"], errors="coerce")
    df.loc[df["time_consumed"] <= 0, "time_consumed"] = pd.NA

# Safe fallback for efficiency_ratio
if "efficiency_ratio" not in df.columns or df["efficiency_ratio"].notna().sum() == 0:
    if "accuracy_total" in df.columns and "time_consumed" in df.columns:
        acc = pd.to_numeric(df["accuracy_total"], errors="coerce")
        tc = pd.to_numeric(df["time_consumed"], errors="coerce").replace(0, pd.NA)
        df["efficiency_ratio"] = acc / tc

# Final numeric cleanup
for col in ["accuracy_total", "time_consumed", "efficiency_ratio"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ------------------------------------------------
# Core assumptions
# ------------------------------------------------
st.divider()
st.subheader("Core assumptions")

st.markdown(
    """
The dashboard treats each row as one learner attempt on one test.

| Data field | Meaning |
|---|---|
| `user_id` | Learner identifier |
| `test_id` | Test or exercise identifier |
| `marks` | Score achieved by the learner |
| `no_of_questions` | Total number of questions in the test |
| `time_taken` | Time spent on the test |
| `accuracy_total` | Main accuracy measure used across the dashboard |
| `speed_raw` | Basic speed signal |
| `adj_speed` | Correct-answer speed signal |
| `efficiency_ratio` | Accuracy adjusted by time usage |
"""
)

st.info(
    """
The model assumes that higher accuracy, reasonable speed, repeated attempts, and stable performance can help indicate learner readiness.  
However, these signals must be interpreted together, not as isolated numbers.
"""
)

st.warning(
    """
Where full test-duration data is unavailable, `time_consumed` is shown as a normalized proxy:

**time_consumed = time_taken / average time_taken**

This keeps the showcase readable, but future versions should use the true test duration wherever available.
"""
)


# ------------------------------------------------
# Formula glossary
# ------------------------------------------------
st.divider()
st.subheader("Formula glossary")

st.info("**Total Accuracy** = marks / total questions")
st.caption("Meaning: How much of the full test the learner answered correctly. This is the main accuracy signal.")

st.info("**Attempted Accuracy** = correct answers / attempted questions")
st.caption("Meaning: Accuracy only on the questions the learner attempted. Useful when learners do not complete every question.")

st.info("**Raw Speed** = attempted questions / time taken")
st.caption("Meaning: How quickly the learner moved through the test.")

st.info("**Adjusted Speed** = correct answers / time taken")
st.caption("Meaning: How quickly the learner answered correctly. This is more useful than raw speed because fast wrong answers are not rewarded.")

st.info("**Speed Marks** = marks / time taken")
st.caption("Meaning: Score gained per minute. Useful when marks are the cleanest available performance signal.")

st.info("**Time Consumed** = time taken / test duration")
st.caption("Meaning: How much of the available test time the learner used. In this showcase, a fallback proxy is used if test duration is unavailable.")

st.info("**Efficiency Ratio** = total accuracy / time consumed")
st.caption("Meaning: Accuracy achieved relative to the time used. A higher value suggests stronger accuracy-time efficiency.")

st.info("**Relative Accuracy** = learner accuracy - dataset average accuracy")
st.caption("Meaning: Whether a learner performed above or below the group average.")

st.info("**Standard Deviation** = spread around the average")
st.caption("Meaning: Helps show variability. High variability means performance is less consistent.")


# ------------------------------------------------
# Dataset snapshot
# ------------------------------------------------
st.divider()
st.subheader("Dataset snapshot")

attempts = len(df)
learners = df["user_id"].nunique() if "user_id" in df.columns else None
tests = df["test_id"].nunique() if "test_id" in df.columns else None

avg_accuracy = safe_mean(df, "accuracy_total")
avg_speed = safe_mean(df, "speed_raw")
avg_adj_speed = safe_mean(df, "adj_speed")
avg_time_taken = safe_mean(df, "time_taken")
avg_efficiency = safe_mean(df, "efficiency_ratio")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Attempts", f"{attempts:,}")
c2.metric("Learners", f"{learners:,}" if learners is not None else "N/A")
c3.metric("Tests", f"{tests:,}" if tests is not None else "N/A")
c4.metric("Avg accuracy", fmt_pct(avg_accuracy, 1))

c5, c6, c7, c8 = st.columns(4)
c5.metric("Avg raw speed", f"{avg_speed:.2f} q/min" if avg_speed is not None and pd.notna(avg_speed) else "0.00 q/min")
c6.metric("Avg adjusted speed", f"{avg_adj_speed:.2f} correct/min" if avg_adj_speed is not None and pd.notna(avg_adj_speed) else "0.00 correct/min")
c7.metric("Avg time taken", f"{avg_time_taken:.1f} min" if avg_time_taken is not None and pd.notna(avg_time_taken) else "0.0 min")
c8.metric("Avg efficiency", fmt_num(avg_efficiency, 2, fallback="0.00"))


# ------------------------------------------------
# Accuracy
# ------------------------------------------------
st.divider()
st.subheader("Accuracy metrics")

st.markdown(
    """
Accuracy measures how well learners perform on tests.

There are two useful accuracy views:
"""
)

st.info("**Attempted Accuracy** = correct answers / attempted questions")
st.info("**Total Accuracy** = correct answers / total questions")

st.markdown(
    """
For the dashboard, **Total Accuracy** is usually the safer headline metric because it considers the full test.
"""
)

accuracy_cols = [c for c in ["user_id", "test_id", "accuracy_attempt", "accuracy_total"] if c in df.columns]

if accuracy_cols:
    st.dataframe(df[accuracy_cols].head(20), use_container_width=True)
else:
    st.info("Accuracy columns are not available in this dataset.")


# ------------------------------------------------
# Time and speed
# ------------------------------------------------
st.divider()
st.subheader("Time and speed metrics")

st.markdown(
    """
Speed is treated as a learning-behaviour signal.

A learner who is accurate and works at a reasonable pace may be more ready than a learner who is accurate but extremely slow, or fast but mostly wrong.
"""
)

st.info("**Raw Time** = time taken")
st.info("**Time Consumed** = time taken / test duration")
st.info("**Relative Time** = time remaining / test duration")
st.info("**Raw Speed** = attempted questions / time taken")
st.info("**Adjusted Speed** = correct answers / time taken")

speed_cols = [
    "user_id",
    "test_id",
    "time_taken",
    "time_consumed",
    "speed_raw",
    "adj_speed",
    "speed_norm",
    "speed_rel_time",
]

speed_cols = [c for c in speed_cols if c in df.columns]

if speed_cols:
    st.dataframe(df[speed_cols].head(20), use_container_width=True)
else:
    st.info("Speed/time columns are not available in this dataset.")


# ------------------------------------------------
# Accuracy, time and speed distributions
# ------------------------------------------------
st.divider()
st.subheader("Distributions: accuracy, time and speed")

st.markdown(
    """
These charts show the spread of performance across the dataset.

They help answer:
- Are most learners scoring low, medium, or high?
- Are learners using very little or too much time?
- Are there extreme speed patterns?
"""
)

c1, c2 = st.columns(2)

with c1:
    if "accuracy_total" in df.columns and df["accuracy_total"].notna().any():
        fig = px.histogram(
            df,
            x="accuracy_total",
            nbins=30,
            title="Accuracy distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `accuracy_total` column available.")

with c2:
    if "time_consumed" in df.columns and df["time_consumed"].notna().any():
        fig2 = px.histogram(
            df,
            x="time_consumed",
            nbins=30,
            title="Time consumed distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No `time_consumed` column available.")

if "speed_raw" in df.columns and df["speed_raw"].notna().any():
    fig3 = px.histogram(
        df,
        x="speed_raw",
        nbins=30,
        title="Raw speed distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No `speed_raw` column available.")


# ------------------------------------------------
# Efficiency
# ------------------------------------------------
st.divider()
st.subheader("Accuracy-to-speed relationship")

st.markdown(
    """
Accuracy and speed are more useful together than separately.

The dashboard uses efficiency-style metrics to understand whether learners are both accurate and reasonably paced.
"""
)

st.info("**Adjusted Speed** = correct answers / time taken")
st.caption("This rewards correct answers per minute, not just speed.")

st.info("**Efficiency Ratio** = total accuracy / time consumed")
st.caption("This compares accuracy against how much of the available time was used.")

if "adj_speed" in df.columns:
    df["accurate_speed"] = df["adj_speed"]

eff_cols = [c for c in ["user_id", "test_id", "accurate_speed", "efficiency_ratio"] if c in df.columns]

if eff_cols:
    st.dataframe(df[eff_cols].head(20), use_container_width=True)
else:
    st.info("Efficiency columns are not available in this dataset.")


st.subheader("Distributions: speed-accuracy signals")

c4, c5 = st.columns(2)

with c4:
    if "accurate_speed" in df.columns and df["accurate_speed"].notna().any():
        fig4 = px.histogram(
            df,
            x="accurate_speed",
            nbins=50,
            title="Adjusted speed distribution"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No adjusted speed data available.")

with c5:
    if "efficiency_ratio" in df.columns and df["efficiency_ratio"].notna().any():
        fig5 = px.histogram(
            df,
            x="efficiency_ratio",
            nbins=50,
            title="Efficiency ratio distribution"
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No efficiency ratio data available.")


# ------------------------------------------------
# Dataset averages
# ------------------------------------------------
st.divider()
st.subheader("Dataset averages")

st.markdown(
    """
Averages show how the overall population is performing.

These values help create a baseline for comparing individual learners, tests, and institutes.
"""
)

mean_accuracy = safe_mean(df, "accuracy_total")
mean_adj_speed = safe_mean(df, "adj_speed")
mean_time_consumed = safe_mean(df, "time_consumed")
mean_time_taken = safe_mean(df, "time_taken")
mean_efficiency = safe_mean(df, "efficiency_ratio")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Mean accuracy", fmt_num(mean_accuracy, 3))
col2.metric("Mean adjusted speed", fmt_num(mean_adj_speed, 3))
col3.metric("Mean time consumed", fmt_num(mean_time_consumed, 3))
col4.metric("Mean time taken", fmt_num(mean_time_taken, 3))
col5.metric("Mean efficiency", fmt_num(mean_efficiency, 3))

st.info(
    """
**How to interpret dataset averages**

These averages describe the overall baseline of the dataset.

- **Mean accuracy** shows the typical test performance across all attempts.
- **Mean adjusted speed** shows how quickly learners answer correctly.
- **Mean time consumed** shows how much test time learners typically use.
- **Mean time taken** shows the average minutes spent per attempt.
- **Mean efficiency** combines accuracy and time use into one signal.

A high average accuracy with reasonable speed is a stronger signal than speed alone.  
Very high speed with low accuracy may suggest guessing or shallow engagement.
"""
)

if mean_accuracy is not None:
    if mean_accuracy >= 0.70:
        st.success("Overall accuracy is relatively strong. Many learners are performing well on the available tests.")
    elif mean_accuracy >= 0.50:
        st.warning("Overall accuracy is moderate. This suggests mixed readiness and room for targeted improvement.")
    else:
        st.error("Overall accuracy is low. This suggests many learners may need foundational support.")

if mean_efficiency is not None:
    if mean_efficiency >= 1:
        st.success("Efficiency is relatively strong: learners are gaining reasonable accuracy for the time used.")
    else:
        st.warning("Efficiency is low or moderate: learners may need support improving accuracy, pacing, or both.")

# ------------------------------------------------
# Relative performance
# ------------------------------------------------
st.divider()
st.subheader("Relative performance")

st.markdown(
    """
Relative performance compares each learner attempt to the dataset average.

This helps show whether a learner is above or below the overall group baseline.
"""
)

if "accuracy_total" in df.columns and df["accuracy_total"].notna().any():
    df["accuracy_avg"] = df["accuracy_total"].mean()
    df["rel_acc"] = df["accuracy_total"] - df["accuracy_avg"]

    st.info("**Relative Accuracy** = learner accuracy - dataset average accuracy")

    c6, c7 = st.columns(2)

    with c6:
        fig6 = px.histogram(
            df,
            x="rel_acc",
            nbins=30,
            title="Relative accuracy distribution"
        )
        st.plotly_chart(fig6, use_container_width=True)

    with c7:
        if "speed_rel_time" in df.columns and df["speed_rel_time"].notna().any():
            fig7 = px.histogram(
                df,
                x="speed_rel_time",
                nbins=30,
                title="Relative time / speed distribution"
            )
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("No `speed_rel_time` column available.")
else:
    st.info("No `accuracy_total` column available for relative performance.")


# ------------------------------------------------
# Variability / consistency
# ------------------------------------------------
st.divider()
st.subheader("Variability / consistency")

st.markdown(
    """
Variability shows how spread out the results are.

Lower variability can suggest more consistent performance.  
Higher variability can suggest unstable performance, uneven test difficulty, or mixed learner behaviour.
"""
)

st.info("**Standard Deviation** = how far values typically spread around the average")

std_accuracy = safe_std(df, "accuracy_total")
std_adj_speed = safe_std(df, "adj_speed")
std_time_consumed = safe_std(df, "time_consumed")
std_time_taken = safe_std(df, "time_taken")
std_efficiency = safe_std(df, "efficiency_ratio")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Std accuracy", fmt_num(std_accuracy, 3))
col2.metric("Std adjusted speed", fmt_num(std_adj_speed, 3))
col3.metric("Std time consumed", fmt_num(std_time_consumed, 3))
col4.metric("Std time taken", fmt_num(std_time_taken, 3))
col5.metric("Std efficiency", fmt_num(std_efficiency, 3))

st.info(
    """
**How to interpret variability**

Standard deviation shows how spread out the results are.

- **Low standard deviation** means learners are performing more consistently.
- **High standard deviation** means performance is uneven across learners or attempts.
- High variability can point to differences in learner ability, test difficulty, effort, or data quality.
- In readiness work, consistency matters because one strong score alone is not enough proof of readiness.
"""
)

if std_accuracy is not None:
    if std_accuracy < 0.15:
        st.success("Accuracy is fairly consistent across attempts. This makes the average accuracy easier to trust.")
    elif std_accuracy < 0.30:
        st.warning("Accuracy has moderate variation. Some learners or tests may need closer review.")
    else:
        st.error("Accuracy varies widely. This suggests uneven performance and possible readiness gaps.")

if std_efficiency is not None:
    if std_efficiency < 0.50:
        st.success("Efficiency is relatively stable across the dataset.")
    elif std_efficiency < 1.00:
        st.warning("Efficiency has moderate variation. Learners may differ in pacing or accuracy-time balance.")
    else:
        st.error("Efficiency varies widely. Some learners may be rushing, struggling, or using much more time than others.")
# ------------------------------------------------
# How to interpret this page
# ------------------------------------------------
st.divider()
st.subheader("How to interpret this page")

st.markdown(
    """
Use this page as the **formula reference** for the rest of the dashboard.



Recommended next steps:
1. Go to **User Summary** to inspect one learner.
2. Go to **Tests Overview** to inspect one test.
3. Go to **Institute Summary** to inspect school-level readiness.
4. When done go to Advanced Metrics to understand SAB, RObust Sab and how readiness signals are being calculated
5. Use this page whenever you need to explain what a metric means.
"""
)
