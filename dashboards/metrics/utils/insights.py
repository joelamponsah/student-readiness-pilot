import numpy as np
import pandas as pd

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
    # ✅ NEW: coverage-consistent status
    sab_df = apply_coverage_override(sab_df)


    return sab_df


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def add_readiness_probability(sab_df):
    """
    Coverage-aware Readiness Probability (0..1), aligned with eligibility rules.
    Steps:
    1) Compute base probability from Work Habits + pass outcomes + evidence.
    2) Apply rule-based caps (LOW_EVIDENCE, FAST_GUESSING, etc.)
    3) Apply eligibility-alignment cap for "Not Eligible"
    4) Apply coverage guardrail via coverage_factor (and optional cap if coverage_risk is High)
    Produces:
      - readiness_probability_base(_pct)
      - readiness_probability(_pct)  [coverage-adjusted headline]
    """
    sab_df = sab_df.copy()

    # Ensure required columns exist
    needed = [
        "robust_SAB_scaled", "pass_rate", "avg_pass_ratio", "test_count", "std_acc",
        "insight_code", "exam_status",
        # coverage guardrail inputs (merged from metrics layer)
        "coverage_factor", "coverage_risk"
    ]
    for c in needed:
        if c not in sab_df.columns:
            sab_df[c] = np.nan

    # ---- Base components (0..1-ish) ----
    wh = (sab_df["robust_SAB_scaled"].fillna(0).clip(0, 100)) / 100.0
    pr = sab_df["pass_rate"].fillna(0).clip(0, 1)
    ar = (sab_df["avg_pass_ratio"].fillna(0).clip(0, 2.0)) / 2.0
    evidence = (sab_df["test_count"].fillna(0) / (sab_df["test_count"].fillna(0) + 10)).clip(0, 1)
    inconsistent = (sab_df["std_acc"].fillna(0) > 0.35).astype(int)

    # ---- Base score -> probability ----
    score = (
        1.6 * wh +
        1.2 * pr +
        0.4 * ar +
        0.6 * evidence
        - 0.4 * inconsistent
        - 1.1
    )
    prob = _sigmoid(score)

    # Store base probability BEFORE caps/guards (optional but useful for debugging)
    sab_df["readiness_probability_base"] = prob
    sab_df["readiness_probability_base_pct"] = (prob * 100).round(1)

    # ---- Rule-alignment caps by insight_code ----
    code = sab_df["insight_code"].astype(str)
    cap = np.full(len(sab_df), np.nan, dtype=float)

    cap[code == "LOW_EVIDENCE"] = 0.25
    cap[code == "FAST_GUESSING"] = 0.35
    cap[code == "INCONSISTENT"] = 0.40
    cap[code == "ACCURACY_RISK"] = 0.45
    cap[code == "SPEED_RISK"] = 0.50
    cap[code == "NEAR_READY"] = 0.75
    cap[code == "READY"] = 0.95

    prob = np.where(np.isnan(cap), prob, np.minimum(prob, cap))

    # ---- Eligibility alignment cap ----
    status = sab_df["exam_status"].astype(str).str.strip().str.lower()
    prob = np.where(status == "not eligible", np.minimum(prob, 0.50), prob)

    # ---- Coverage guardrail (headline probability) ----
    # coverage_factor is expected in [0.40, 1.00]. If missing, treat as 1.
    cf = pd.to_numeric(sab_df["coverage_factor"], errors="coerce").fillna(1.0).clip(0.40, 1.0)
    prob_adj = prob * cf

    # Optional: additional cap when coverage risk is High (prevents very high readiness with gaps)
    risk = sab_df["coverage_risk"].astype(str).str.strip().str.lower()
    prob_adj = np.where(risk == "high", np.minimum(prob_adj, 0.70), prob_adj)

    sab_df["readiness_probability"] = prob_adj
    sab_df["readiness_probability_pct"] = (prob_adj * 100).round(1)

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

def apply_coverage_override(sab_df):
    """
    If learner is 'Eligible' by habits but coverage-adjusted readiness is low,
    downgrade to 'Conditionally Eligible' with a clear explanation.
    """
    sab_df = sab_df.copy()

    # Guard columns
    for c in ["exam_status", "insight_code", "readiness_probability_pct", "coverage_risk", "blocking_reason"]:
        if c not in sab_df.columns:
            sab_df[c] = np.nan

    status = sab_df["exam_status"].astype(str).str.lower().str.strip()
    risk = sab_df["coverage_risk"].astype(str).str.lower().str.strip()
    p = pd.to_numeric(sab_df["readiness_probability_pct"], errors="coerce").fillna(0)

    # Override condition (tune thresholds as you like)
    override = (status == "eligible") & ((risk == "high") | (p < 50))

    sab_df.loc[override, "exam_status"] = "Conditionally Eligible"
    sab_df.loc[override, "insight_code"] = "NEAR_READY"
    sab_df.loc[override, "insight_message"] = (
        "Strong work habits, but coverage gaps exist across tests. Improve weak units before final exam."
    )
    sab_df.loc[override, "blocking_reason"] = (
        "Coverage gaps: learner is strong in some tests but has weak/low-evidence areas."
    )
    return sab_df
