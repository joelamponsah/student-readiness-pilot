def stakeholder_summary(row):
    """
    High-level, non-technical summary for leadership & instructors
    """
    if row["exam_status"] == "Eligible":
        return (
            f"This learner demonstrates consistent performance across assessments, "
            f"with stable accuracy ({row['mean_accuracy']:.0%}) and controlled pace. "
            f"Behavioral indicators suggest readiness for final examination conditions."
        )

    if row["insight_code"] == "NEAR_READY":
        return (
            f"This learner is approaching exam readiness but shows minor instability "
            f"in either speed or accuracy. With targeted practice and additional attempts, "
            f"they are likely to reach readiness soon."
        )

    if row["insight_code"] == "FAST_GUESSING":
        return (
            f"Performance data indicates unusually high speed paired with low accuracy, "
            f"suggesting rushed attempts or guessing behavior. Intervention should focus "
            f"on pacing and accuracy reinforcement."
        )

    if row["insight_code"] == "INCONSISTENT":
        return (
            f"Results show high variability across tests, indicating inconsistent mastery. "
            f"Stabilization through structured practice is recommended before exam attempts."
        )

    if row["insight_code"] in ["ACCURACY_RISK", "SPEED_RISK"]:
        return (
            f"Core performance indicators fall below readiness thresholds. "
            f"Focused remediation is required before exam eligibility."
        )

    return (
        "There is currently insufficient assessment data to determine exam readiness. "
        "Additional attempts are required for reliable evaluation."
    )
