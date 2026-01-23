from utils.metrics import compute_sab_behavioral, compute_test_analytics
from utils.insights import apply_insight_engine

def build_user_view(df_fact):
    users = compute_sab_behavioral(df_fact)
    # ensure institute_std survives:
    if "institute_std" not in users.columns:
        users = users.merge(df_fact[["user_id","institute_std"]].drop_duplicates("user_id"),
                            on="user_id", how="left")
    users = apply_insight_engine(users)
    return users

def build_test_view(df_fact):
    return compute_test_analytics(df_fact)
