def coach_feedback(row):
    """
    Personalized learner-facing feedback
    """
    if row["exam_status"] == "Eligible":
        return (
            "You're performing consistently and maintaining a healthy balance between "
            "accuracy and speed. Keep practicing under exam-like conditions to stay sharp."
        )

    if row["insight_code"] == "NEAR_READY":
        return (
            "You're very close to being exam-ready. Focus on strengthening weaker areas "
            "and aim for consistent performance over the next few tests."
        )

    if row["insight_code"] == "FAST_GUESSING":
        return (
            "You're moving quickly, which is good, but accuracy is suffering. Slow down, "
            "read questions carefully, and prioritize getting them right."
        )

    if row["insight_code"] == "ACCURACY_RISK":
        return (
            "Accuracy is holding you back. Take time to review concepts and understand "
            "mistakes before attempting more timed practice."
        )

    if row["insight_code"] == "SPEED_RISK":
        return (
            "Your accuracy is reasonable, but speed varies. Practice pacing to build "
            "confidence under time pressure."
        )

    if row["insight_code"] == "INCONSISTENT":
        return (
            "Your performance varies across attempts. Try shorter, focused practice "
            "sessions to build stability."
        )

    return (
        "Complete a few more tests so we can better understand your strengths and areas "
        "to improve."
    )
