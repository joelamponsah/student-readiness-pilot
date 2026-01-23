from utils.metrics import compute_sab_behavioral, compute_test_analytics
from utils.insights import apply_insight_engine

def build_user_view(df_fact):
    users = compute_sab_behavioral(df_fact)

    user_inst = df_fact[['user_id', 'institute_std']].drop_duplicates('user_id')
    users = users.merge(user_inst, on='user_id', how='left')
    users['institute_std'] = users['institute_std'].fillna('Unknown').astype(str)

    users = apply_insight_engine(users)
    return users


def build_test_view(df_fact):
    return compute_test_analytics(df_fact)
