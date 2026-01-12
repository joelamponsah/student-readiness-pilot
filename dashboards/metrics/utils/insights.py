import numpy as np

INSIGHT_TEXT = {
    "READY": {
        "status": "Eligible",
        "message": "Consistent performance with strong accuracy and stable speed.",
        "action": "Proceed to full-length mock exams."
    },
    "NEAR_READY": {
        "status": "Conditionally Eligible",
        "message": "Close to exam-ready but minor gaps remain.",
        "action": "Target weak topics and attempt 2–3 more tests."
    },
    "ACCURACY_RISK": {
        "status": "Not Eligible",
        "message": "Accuracy consistency is low across attempts.",
        "action": "Focus on concept reinforcement before speed."
    },
    "SPEED_RISK": {
        "status": "Not Eligible",
        "message": "Speed consistency is unstable.",
        "action": "Practice timed sections with pacing goals."
    },
    "FAST_GUESSING": {
        "status": "Not Eligible",
        "message": "High speed combined with low accuracy detected.",
        "action": "Slow down and prioritize correctness."
    },
    "INCONSISTENT": {
        "status": "Not Eligible",
        "message": "High variability in performance.",
        "action": "Stabilize performance across multiple tests."
    },
    "LOW_EVIDENCE": {
        "status": "Not Eligible",
        "message": "Insufficient test data to assess readiness.",
        "action": "Attempt at least 3 full tests."
    }
}

# -------------------------
# Rule Engine
# -------------------------
def blocking_insight(row, cohort_median_speed):
    if row["test_count"] < 3:
        return "LOW_EVIDENCE"

    if row["mean_accuracy"] < 0.35 and row["mean_speed"] > cohort_median_speed:
        return "FAST_GUESSING"

    if row["std_acc"] > 0.35:
        return "INCONSISTENT"

    return None


def gap_insight(row):
    if row["accuracy_consistency"] < 0.5:
        return "ACCURACY_RISK"

    if row["speed_consistency"] < 0.35:
        return "SPEED_RISK"

    if row["robust_SAB_scaled"] < 50:
        return "NEAR_READY"

    return "READY"


def generate_insight(row, cohort_median_speed):
    block = blocking_insight(row, cohort_median_speed)
    if block:
        return block
    return gap_insight(row)


def apply_insight_engine(sab_df):
    sab_df = sab_df.copy()
    cohort_median_speed = sab_df["mean_speed"].median()

    sab_df["insight_code"] = sab_df.apply(
        lambda r: generate_insight(r, cohort_median_speed),
        axis=1
    )

    sab_df["exam_status"] = sab_df["insight_code"].map(
        lambda c: INSIGHT_TEXT[c]["status"]
    )
    sab_df["insight_message"] = sab_df["insight_code"].map(
        lambda c: INSIGHT_TEXT[c]["message"]
    )
    sab_df["recommended_action"] = sab_df["insight_code"].map(
        lambda c: INSIGHT_TEXT[c]["action"]
    )

    return sab_df
