import streamlit as st

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="Learner Readiness Dashboard",
    page_icon="📘",
    layout="wide"
)

# ------------------------------------------------
# Header
# ------------------------------------------------
st.title("📘 Learner Readiness Dashboard")
st.subheader("Grant Showcase Version")

st.markdown(
    """
This dashboard is an early proof-of-concept for **Learner Readiness Intelligence**.

It uses historical assessment activity from the eCampus platform to show how learner performance, test behaviour, 
speed, accuracy, pass outcomes, and school-level patterns can be transformed into useful readiness insights.

The purpose of this showcase is to demonstrate what is already possible with the existing data, and why funding is needed 
to develop a more robust learner readiness model.
"""
)

st.info(
    """
**Important note:** This is a first-version showcase dashboard.  
It is not yet the final production model. Some outputs are exploratory and should be interpreted as decision-support signals, not final exam predictions.
"""
)

# ------------------------------------------------
# Quick navigation
# ------------------------------------------------
st.divider()
st.header("How to use this dashboard")

st.markdown(
    """
Start with the pages below depending on what you want to understand:
"""
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
### 🏫 School / Institution View
Use this if you want to understand performance at school level.

**Best page to start:**
- **Institute Summary**

This page shows:
- readiness breakdown by school
- at-risk learners
- almost-ready learners
- exam-ready learners
- pass/fail overview
- hardest tests
- recommended actions
- learner intervention lists
"""
    )

with col2:
    st.markdown(
        """
### 👤 Individual Learner View
Use this if you want to inspect one learner.

**Best page to start:**
- **User Summary**

This page shows:
- learner profile
- attempts and activity window
- pass rate
- average score
- accuracy and speed
- readiness insight
- subject/test breakdown
- weekly trends
- difficulty summary
"""
    )

with col3:
    st.markdown(
        """
### 🧪 Test / Assessment View
Use this if you want to understand test performance.

**Best pages to start:**
- **Tests Overview**
- **Accuracy Speed by User and Tests**
- **Difficulty / DCI / Stability**

These pages show:
- test-level performance
- test difficulty
- score distribution
- speed distribution
- learner performance per test
- test stability and quality signals
"""
    )

# ------------------------------------------------
# Recommended demo paths
# ------------------------------------------------
st.divider()
st.header("Recommended demo paths")

st.markdown(
    """
For the smoothest review, try these paths:
"""
)

demo1, demo2 = st.columns(2)

with demo1:
    st.markdown(
        """
### Path 1: School leadership view

1. Open **Institute Summary**
2. Select **Opoku Ware** if available
3. Review:
   - Learners
   - Average Accuracy
   - Average Readiness Score
   - At-risk / Almost-ready / Exam-ready groups
4. Scroll to:
   - Recommended actions
   - Priority intervention list
   - Hardest tests

**Use this path to show how a school leader could identify who needs support.**
"""
    )

with demo2:
    st.markdown(
        """
### Path 2: Learner diagnosis view

1. Open **User Summary**
2. Search/select a learner by **User ID**
3. Review:
   - total attempts
   - pass rate
   - average score
   - readiness insight
   - performance by test
   - weekly trends
   - difficulty summary

**Use this path to show how the dashboard explains learner behaviour, not just raw scores.**
"""
    )

demo3, demo4 = st.columns(2)

with demo3:
    st.markdown(
        """
### Path 3: Test quality view

1. Open **Tests Overview**
2. Select a test by name
3. Review:
   - total attempts
   - unique learners
   - highest / lowest / average score
   - pass rate
   - activity window
   - accuracy and marks distributions

**Use this path to show how tests themselves can be evaluated.**
"""
    )

with demo4:
    st.markdown(
        """
### Path 4: Funding story

1. Start with **Institute Summary**
2. Move to **User Summary**
3. Then open **Tests Overview**
4. Explain the gap:
   - current version shows strong signals
   - funding is needed for better data quality
   - improved modelling
   - real-time interventions
   - teacher and student-facing recommendations

**Use this path for grant reviewers.**
"""
    )

# ------------------------------------------------
# Page guide
# ------------------------------------------------
st.divider()
st.header("Page guide")

st.markdown(
    """
| Page | What it shows | Best used for |
|---|---|---|
| **Basic Metrics** | General speed, accuracy, efficiency and attempts | Quick performance overview |
| **Accuracy Speed by User and Tests** | Aggregated user/test speed and accuracy metrics | Comparing learners and tests |
| **Test and Topic Trends** | Performance patterns across tests/topics over time | Spotting improvement or decline |
| **Difficulty / DCI / Stability** | Test difficulty and consistency indicators | Understanding assessment quality |
| **Ranking and Leaderboard** | Relative learner performance | Finding top performers and weak spots |
| **Advanced Metrics** | Deeper calculated indicators | Technical exploration |
| **Exam Readiness Model** | Readiness-style outputs and model signals | Showing the predictive direction |
| **User Summary** | Full learner-level profile | Diagnosing individual learners |
| **Tests Overview** | Test-level drilldown by test name | Reviewing assessment performance |
| **Institute Summary** | School-level dashboard | Best page for school/grant showcase |
"""
)

# ------------------------------------------------
# What the dashboard is trying to prove
# ------------------------------------------------
st.divider()
st.header("What this dashboard is trying to prove")

st.markdown(
    """
The dashboard is built around a simple idea:

> Learner readiness should not be judged by one score alone.

A stronger readiness signal should consider:
- how accurate the learner is
- how consistently they perform
- how many attempts they have made
- how fast they work
- whether they pass or fail tests
- whether the tests are easy, moderate, or hard
- how their performance changes over time
- how they compare with peers
- how their school or class is performing overall

This showcase demonstrates that eCampus already has enough historical learning data to begin building this type of intelligence layer.
"""
)

# ------------------------------------------------
# Current limitations
# ------------------------------------------------
st.divider()
st.header("Current limitations")

st.warning(
    """
This version is useful for demonstration, but it is not yet a fully validated production readiness model.
"""
)

st.markdown(
    """
Known areas for improvement include:
- stronger data quality controls
- cleaner learner and school identifiers
- better handling of incomplete attempts
- clearer pass mark validation
- more robust test difficulty calibration
- real-time data pipelines
- teacher-facing recommendations
- student-facing learning guidance
- validation against actual exam outcomes
"""
)

# ------------------------------------------------
# Why funding is needed
# ------------------------------------------------
st.divider()
st.header("Why funding is needed")

st.markdown(
    """
Funding would allow this prototype to become a stronger, production-ready Learner Readiness system.

The next stage should focus on:
- improving the data foundation
- validating the readiness model
- adding real-time monitoring
- building teacher dashboards
- creating student intervention recommendations
- supporting schools with early-warning insights
- linking readiness signals to actual exam performance

The long-term goal is to help schools identify struggling learners earlier and support them before final exams.
"""
)

# ------------------------------------------------
# Contact
# ------------------------------------------------
st.divider()
st.header("Questions or feedback")

st.markdown(
    """
For questions, feedback, or access support, contact:

**Your Name**  
**your.email@example.com**

Please replace this with the correct project contact email before sharing the dashboard.
"""
)

st.caption("Learner Readiness Dashboard · Grant Showcase Version")
