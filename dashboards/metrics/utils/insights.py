import numpy as np
from utils.insight_text import stakeholder_summary
from utils.coach_feedback import coach_feedback


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

BLOCK_REASON_TEXT = {
    "LOW_EVIDENCE": "Insufficient evidence (needs more attempts)",
    "FAST_GUESSING": "Rushing/guessing risk (high speed, low accuracy)",
    "INCONSISTENT": "Unstable performance (high variability)",
    "ACCURACY_RISK": "Accuracy below threshold",
    "SPEED_RISK": "Speed/pacing instability",
    "NEAR_READY": "Almost ready (minor gaps remain)",
    "READY": "Ready"
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

def add_blocking_reason(sab_df):
    sab_df = sab_df.copy()
    sab_df["blocking_reason"] = sab_df["insight_code"].map(lambda c: BLOCK_REASON_TEXT.get(c, str(c)))
    sab_df["is_blocked"] = sab_df["exam_status"].map(lambda s: str(s).lower() != "eligible")
    return sab_df


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
    
    sab_df["stakeholder_insight"] = sab_df.apply(stakeholder_summary, axis=1)
    sab_df["coach_feedback"] = sab_df.apply(coach_feedback, axis=1)
    
    
    #NEW
    sab_df = add_risk_band(sab_df)
    sab_df = add_readiness_probability(sab_df)
    sab_df["redemption_plan"] = sab_df.apply(redemption_plan, axis=1)
        
    sab_df = add_blocking_reason(sab_df)          # optional but fine
    sab_df = add_readiness_probability(sab_df)    # must come after insight_code/exam_status


    return sab_df


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def add_readiness_probability(sab_df):
    """
    Readiness Probability (0..1) aligned with eligibility rules.
    - Base score uses Work Habits (robust_SAB_scaled), pass outcomes, and evidence.
    - Then apply rule-based caps/penalties so "Not Eligible" doesn't show high probability.
    """
    sab_df = sab_df.copy()

    # Ensure required columns exist
    for c in ["robust_SAB_scaled", "pass_rate", "avg_pass_ratio", "test_count", "std_acc", "insight_code", "exam_status"]:
        if c not in sab_df.columns:
            sab_df[c] = np.nan

    # Normalize components
    wh = (sab_df["robust_SAB_scaled"].fillna(0).clip(0, 100)) / 100.0       # 0..1
    pr = sab_df["pass_rate"].fillna(0).clip(0, 1)                          # 0..1
    # avg_pass_ratio is optional, keep weak influence; if missing it won't dominate
    ar = (sab_df["avg_pass_ratio"].fillna(0).clip(0, 2.0)) / 2.0           # 0..1
    evidence = (sab_df["test_count"].fillna(0) / (sab_df["test_count"].fillna(0) + 10)).clip(0, 1)

    # Inconsistency penalty
    inconsistent = (sab_df["std_acc"].fillna(0) > 0.35).astype(int)

    # Base score (conservative)
    score = (
        1.6 * wh +
        1.2 * pr +
        0.4 * ar +
        0.6 * evidence
        - 0.4 * inconsistent
        - 1.1
    )
    prob = _sigmoid(score)

    # --------- Rule-alignment layer (the important fix) ---------
    code = sab_df["insight_code"].astype(str)

    # Hard caps for “blocking” reasons
    # These ensure Not Eligible doesn’t show high probability.
    cap = np.full(len(sab_df), np.nan, dtype=float)

    cap[code == "LOW_EVIDENCE"] = 0.25          # not enough attempts
    cap[code == "FAST_GUESSING"] = 0.35         # risky behavior
    cap[code == "INCONSISTENT"] = 0.40          # unstable performance
    cap[code == "ACCURACY_RISK"] = 0.45
    cap[code == "SPEED_RISK"] = 0.50            # could be borderline
    cap[code == "NEAR_READY"] = 0.75            # allow higher but not “certain”
    cap[code == "READY"] = 0.95

    # Apply cap where defined
    prob = np.where(np.isnan(cap), prob, np.minimum(prob, cap))

    # Additional safety: if exam_status explicitly not eligible, cap at 0.50
    # (keeps the story consistent for stakeholders)
    status = sab_df["exam_status"].astype(str).str.lower()
    prob = np.where(status == "not eligible", np.minimum(prob, 0.50), prob)

    sab_df["readiness_probability"] = prob
    sab_df["readiness_probability_pct"] = (prob * 100).round(1)

    return sab_df

def add_risk_band(sab_df):
    """
    Simple cohort-relative banding for minister-friendly storytelling.
    Uses robust_SAB_scaled percentile (0..100).
    """
    sab_df = sab_df.copy()
    if "robust_SAB_scaled" not in sab_df.columns:
        sab_df["risk_band"] = "Unknown"
        return sab_df

    sab_df["risk_band"] = "On Track"
    sab_df.loc[sab_df["test_count"] < 3, "risk_band"] = "Low Evidence"
    sab_df.loc[sab_df["robust_SAB_scaled"] < 20, "risk_band"] = "At Risk"
    sab_df.loc[(sab_df["robust_SAB_scaled"] >= 20) & (sab_df["robust_SAB_scaled"] < 40), "risk_band"] = "Watchlist"
    sab_df.loc[sab_df["robust_SAB_scaled"] >= 75, "risk_band"] = "Ready"

    # Optional pass-based override (if available)
    if "user_pass_rate" in sab_df.columns and "avg_pass_ratio" in sab_df.columns:
        sab_df.loc[(sab_df["user_pass_rate"] < 0.5) & (sab_df["avg_pass_ratio"] < 0.9), "risk_band"] = "At Risk"

    return sab_df

def redemption_plan(row):
    """
    Milestone-based plan (demo-friendly).
    Uses insight_code + risk_band.
    """
    code = str(row.get("insight_code", "LOW_EVIDENCE"))
    band = str(row.get("risk_band", "On Track"))
    tc = float(row.get("test_count", 0) or 0)

    if tc < 3 or code == "LOW_EVIDENCE" or band == "Low Evidence":
        return [
            "Week 1: Complete 3 full tests to establish a reliable baseline.",
            "Week 2: Review mistakes; repeat 1 timed test under exam conditions.",
            "Checkpoint: Recompute readiness after 5 total attempts."
        ]

    if code == "FAST_GUESSING" or band == "At Risk":
        return [
            "Week 1: Pacing + accuracy drills (slow down, reduce careless errors).",
            "Week 2: 2 full tests focusing on accuracy-first strategy.",
            "Checkpoint: Aim for pass ratio ≥ 0.95 and rising pass rate."
        ]

    if code in ["ACCURACY_RISK", "INCONSISTENT"]:
        return [
            "Week 1: Concept reinforcement + error log on weak topics.",
            "Week 2: 2–3 tests; target stable accuracy across attempts.",
            "Checkpoint: Improve accuracy consistency and maintain steady pace."
        ]

    if code == "SPEED_RISK":
        return [
            "Week 1: Timed sections + pacing drills to stabilize speed.",
            "Week 2: 2 full tests under exam conditions.",
            "Checkpoint: Improve speed consistency without sacrificing accuracy."
        ]

    if code == "NEAR_READY" or band in ["Watchlist", "On Track"]:
        return [
            "Next 7 days: 2–3 tests targeting weak topics.",
            "Next 14 days: 1 full mock under exam conditions.",
            "Checkpoint: Maintain readiness probability > 75%."
        ]

    return [
        "Maintain performance with 1–2 full mocks per week.",
        "Practice under exam conditions and review mistakes.",
        "Checkpoint: Keep readiness stable above target threshold."
    ]

