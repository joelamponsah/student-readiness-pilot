# utils/metrics.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit may not be available in all contexts
    st = None

def save_uploaded_df(df: pd.DataFrame, path="data/raw_attempts.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_data_from_disk_or_session(default_path="data/raw_attempts.csv"):
    """
    Loads df from Streamlit session_state if present; otherwise from disk.
    FIXES:
      - previously referenced df before assignment
      - previously returned undeduped session df
    """
    # prefer session state if available
    try:
        import streamlit as st
        if "df" in st.session_state and st.session_state["df"] is not None:
            # Preserve the session copy exactly. Attempt-level dedupe belongs to
            # the DQ gate, not the loader.
            return st.session_state["df"].copy()
    except Exception:
        pass

    # else load from disk
    if os.path.exists(default_path):
        try:
            return pd.read_csv(default_path, low_memory=False)
        except Exception:
            return None
    return None


def _load_v13_pipeline_module():
    """Import the shared v1.3 pipeline lazily to avoid circular imports."""
    import importlib
    return importlib.import_module("utils.v13_pipeline")


if st is not None and hasattr(st, "cache_data"):
    @st.cache_data(show_spinner=False)
    def _cached_v13_artifacts_from_raw(raw_signature: tuple):
        raw_df = raw_signature[0]
        v13_pipeline = _load_v13_pipeline_module()
        return v13_pipeline.build_v13_artifacts(raw_df)
else:  # pragma: no cover
    def _cached_v13_artifacts_from_raw(raw_signature: tuple):
        raw_df = raw_signature[0]
        v13_pipeline = _load_v13_pipeline_module()
        return v13_pipeline.build_v13_artifacts(raw_df)


def get_v13_artifacts(default_path="data/raw_attempts.csv"):
    """
    Load the current raw attempt dataset and build/cache the shared v1.3 artifacts.

    Returns:
        (raw_df, artifacts)
    """
    raw_df = load_data_from_disk_or_session(default_path=default_path)
    if raw_df is None or raw_df.empty:
        return None, {}

    # Use a lightweight content signature for cache stability.
    signature = (
        raw_df.copy(),
        len(raw_df),
        tuple(raw_df.columns),
    )

    try:
        artifacts = _cached_v13_artifacts_from_raw(signature)
    except TypeError:
        # Fallback when caching cannot hash a dataframe cleanly.
        v13_pipeline = _load_v13_pipeline_module()
        artifacts = v13_pipeline.build_v13_artifacts(raw_df)

    return raw_df, artifacts
# basic metrics ------------------------------------------------


def _canonical_accuracy_denominator(df: pd.DataFrame) -> pd.Series:
    """
    Canonical denominator for score accuracy calculations.

    Preference order:
      1) accuracy_denominator, when already annotated by DQ/notebook logic
      2) max_marks_effective, when present and not derived from bank-size totals
      3) no_of_questions, the delivered attempt question count from test_takers
      4) question_limit, the configured delivered count fallback
      5) total_questions, bank-size fallback only

    tests.total_questions can contain the full randomized question bank, so it
    must not outrank attempt-level no_of_questions.
    """
    denom = pd.Series(np.nan, index=df.index)
    delivered = pd.Series(np.nan, index=df.index)
    for col in ["no_of_questions", "question_limit"]:
        if col in df.columns:
            delivered = delivered.fillna(pd.to_numeric(df[col], errors="coerce"))

    if "accuracy_denominator" in df.columns:
        denom = denom.fillna(pd.to_numeric(df["accuracy_denominator"], errors="coerce"))

    if "max_marks_effective" in df.columns:
        candidate = pd.to_numeric(df["max_marks_effective"], errors="coerce")
        bank_like = delivered.gt(0) & candidate.gt(delivered * 5)
        denom = denom.fillna(candidate.mask(bank_like))

    for col in ["no_of_questions", "question_limit", "total_questions"]:
        if col in df.columns:
            candidate = pd.to_numeric(df[col], errors="coerce")
            denom = denom.fillna(candidate)
    return denom.replace(0, np.nan)


def safe_accuracy_series(df: pd.DataFrame) -> pd.Series:
    """
    Canonical display-ready accuracy series.

    Preference order:
      1) accuracy_total_safe
      2) accuracy_total
      3) accuracy_attempt

    Zero-attempt rows are excluded from the display series using the preserved
    attempted_questions_raw field when available, because they are inactive rows
    rather than real attempts.
    """
    series = pd.Series(np.nan, index=df.index)
    for col in ["accuracy_total_safe", "accuracy_total", "accuracy_attempt"]:
        if col in df.columns:
            candidate = pd.to_numeric(df[col], errors="coerce")
            series = series.fillna(candidate)

    active_mask = None
    if "attempted_questions_raw" in df.columns:
        active_mask = pd.to_numeric(df["attempted_questions_raw"], errors="coerce").fillna(0) > 0
    elif "attempted_questions" in df.columns:
        active_mask = pd.to_numeric(df["attempted_questions"], errors="coerce").fillna(0) > 0

    if active_mask is not None:
        series = series.where(active_mask)

    return series.clip(lower=0, upper=1)



# --- Speed & accuracy base features (idempotent) ---

def compute_basic_metrics2(df):
    df = df.copy()

    # Ensure numeric columns exist
    for c in ['attempted_questions','correct_answers','marks','time_taken','duration','no_of_questions','pass_mark','total_questions','question_limit','max_marks_effective']:
        if c not in df.columns:
            df[c] = np.nan

    # Coerce numeric
    for c in ['attempted_questions','correct_answers','marks','time_taken','duration','no_of_questions','pass_mark','total_questions','question_limit','max_marks_effective']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Guard zeros
    df['time_taken'] = df['time_taken'].replace(0, np.nan)
    df['duration'] = df['duration'].replace(0, np.nan)
    # Preserve the raw value long enough to flag zero-attempt rows downstream.
    df['attempted_questions_raw'] = df['attempted_questions']
    df['attempted_questions'] = df['attempted_questions'].replace(0, np.nan)
    df['no_of_questions'] = df['no_of_questions'].replace(0, np.nan)
    df['total_questions'] = df['total_questions'].replace(0, np.nan)
    df['question_limit'] = df['question_limit'].replace(0, np.nan)
    # Attempt accuracy should use delivered questions first. total_questions can
    # be the full randomized question bank and is only a last-resort fallback.
    df['max_marks_effective'] = df['max_marks_effective'].fillna(df['no_of_questions']).fillna(df['question_limit']).fillna(df['total_questions'])
    df['max_marks_effective'] = df['max_marks_effective'].replace(0, np.nan)

    # Accuracy
    df['accuracy_attempt'] = (df['correct_answers'] / df['attempted_questions']).fillna(0)
    denom = _canonical_accuracy_denominator(df)
    df['accuracy_denominator'] = denom
    # Accuracy is a proportion, so keep it bounded even when the raw marks/denominator
    # combination is noisy or partially inferred.
    df['accuracy_total'] = (df['marks'] / denom).clip(lower=0, upper=1).fillna(0)
    
    df["accuracy_total_safe"] = np.where(
        df.get("no_of_questions_suspect", False) == False,
        (df["marks"] / denom).clip(lower=0, upper=1).replace([np.inf, -np.inf], np.nan),
        np.nan
    )


    # Speed per second
    df['speed_raw'] = df['attempted_questions'] / df['time_taken']      # attempted/min
    df['adj_speed'] = df['correct_answers'] / df['time_taken']          # correct/min

    # Requested: speed_acc_raw as questions per minute (attempted/min)
    df['speed_acc_raw'] = df['speed_raw']

    # Optional: correct per minute (kept)
    df['correct_per_min'] = df['adj_speed']

    # Time ratios (duration dependent)
    df['speed_rel_time'] = ((df['duration'] - df['time_taken']) / df['duration']).clip(lower=0)
    df['time_consumed'] = (df['time_taken'] / df['duration']).clip(0, 1)

    # Normalized speed
    df['speed_norm'] = 0.0
    if df['speed_raw'].notna().sum() >= 2:
        scaler = MinMaxScaler()
        df['speed_norm'] = scaler.fit_transform(df[['speed_raw']].fillna(0))

    # Efficiency (duration-based) and % version
    df['efficiency_ratio'] = np.where(
        df['time_consumed'].notna() & (df['time_consumed'] > 0),
        df['accuracy_total'] / df['time_consumed'],
        np.nan
    )
    df['efficiency_pct'] = (df['efficiency_ratio'] * 100).clip(lower=0, upper=300)

    # Fallback efficiency (no duration needed): score per minute
    df['efficiency_per_min'] = np.where(
        df['time_taken'].notna() & (df['time_taken'] > 0),
        df['accuracy_total'] / df['time_taken'],
        np.nan
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# --- Per-test analytics ---
def compute_test_analytics(df):
    df = compute_basic_metrics2(df)
    df = df[df['time_taken'] > 0]
    df['speed_marks'] = df['marks'] / df['time_taken']
    df['accuracy_total'] = (df['marks'] / _canonical_accuracy_denominator(df)).clip(lower=0, upper=1).fillna(0)
    df['efficiency_ratio'] = df['accuracy_total'] / df['time_consumed'].replace(0, np.nan)
    pass_col = 'pass_mark_effective' if 'pass_mark_effective' in df.columns else 'pass_mark'
    pass_source = df.loc[df.index, pass_col] if pass_col in df.columns else pd.Series(np.nan, index=df.index)
    if 'pass_mark_ambiguous' in df.columns:
        pass_source = pass_source.mask(df['pass_mark_ambiguous'])
    agg = df.groupby('test_id').agg(
        mean_time=('time_taken','mean'),
        std_time=('time_taken','std'),
        mean_speed=('speed_marks','mean'),
        std_speed=('speed_marks','std'),
        mean_efficiency=('efficiency_ratio','mean'),
        mean_accuracy=('accuracy_total','mean'),
        pass_count=('marks', lambda x: (x >= pass_source.loc[x.index]).sum() if pass_col in df.columns else np.nan),
        taker_count=('test_taker_id','count')
    ).reset_index()
    # consistency (damped)
    agg['std_speed'] = agg['std_speed'].fillna(0)
    agg['std_time'] = agg['std_time'].fillna(0)
    agg['mean_speed'] = agg['mean_speed'].replace(0, np.nan)
    agg['speed_consistency'] = 1 / (1 + (agg['std_speed'] / agg['mean_speed']).fillna(0))
    agg['time_consistency'] = 1 / (1 + (agg['std_time'] / agg['mean_time']).fillna(0))
    # pass_rate
    if 'pass_count' in agg.columns:
        agg['pass_rate'] = agg['pass_count'] / agg['taker_count']
    return agg

# --- Topic analytics ---
def compute_topic_analytics(df, topic_col='topic'):
    if topic_col not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df['speed_marks'] = df['marks'] / df['time_taken']
    agg = df.groupby(topic_col).agg(
        mean_accuracy=('accuracy_total','mean'),
        mean_speed=('speed_marks','mean'),
        tests=('test_id','nunique'),
        takers=('user_id','nunique')
    ).reset_index()
    return agg

# --- Difficulty & DCI ---
def compute_difficulty_df(df):
    df = compute_basic_metrics2(df)
    # pass_flag per row
    pass_col = 'pass_mark_effective' if 'pass_mark_effective' in df.columns else 'pass_mark'
    if pass_col in df.columns:
        df['passed'] = (df['marks'] >= df[pass_col]).astype(int)
    else:
        df['passed'] = (df['marks'] >= df['max_marks_effective']).astype(int)
        
    # --- Handle inactive (no attempts) users ---
    # `attempted_questions` is normalized above, so use the preserved raw value.
    attempted_raw = pd.to_numeric(df.get('attempted_questions_raw', df.get('attempted_questions', np.nan)), errors='coerce')
    df['inactive'] = (attempted_raw == 0).astype(int)
    df.loc[df['inactive'] == 1, ['marks', 'accuracy_total', 'accuracy_attempt', 'accuracy_norm']] = 0
    if 'pass_mark_ambiguous' in df.columns:
        df.loc[df['pass_mark_ambiguous'], 'passed'] = np.nan
    
        
    test_pass = df.groupby('test_id').agg(
        pass_rate=('passed','mean'),
        mean_accuracy=('accuracy_total','mean'),
        test_consistency=('accuracy_total','std'),
        takers=('user_id','nunique')
    ).reset_index()
    
    # DCI: closeness between mean_accuracy and (1 - difficulty), difficulty ~ 1-pass_rate
    test_pass['difficulty'] = 1 - test_pass['pass_rate']
    test_pass['DCI'] = 1 - (abs(test_pass['mean_accuracy'] - (1 - test_pass['difficulty'])) )

    # --- Fix difficulty scoring ---
    #df['difficulty_score'] = df['pass_mark'] / df['no_of_questions']
    test_pass['difficulty_label'] = pd.cut(
    test_pass['difficulty'],
    bins=[0, 0.59, 0.89, 1.0],
    labels=['easy', 'moderate', 'hard'],
    include_lowest=True
    )
    # test stability
    test_pass['stability'] = 1 / (1 + test_pass['test_consistency'].fillna(0))

    # Label the test stability
    test_pass['test_stability'] = pd.cut(
        test_pass['DCI'],
        bins=[0, 0.33, 0.66, 1.0],
        labels=['unstable', 'moderately stable', 'highly stable'],
        include_lowest=True
    )

    test_pass
    #test_pass = test_pass.merge(df[['passed', 'difficulty_score','difficulty_label']], on='test_id', how='left')

    #test_pass.fillna({'pass_rate': 0, 'test_consistency': 0, 'difficulty_score': 0}, inplace=True)

    # Normalize test consistency (variability)
    #scaler = MinMaxScaler()
    #df['consistency_norm'] = 1 - scaler.fit_transform(df[['test_consistency']])
    # Invert because high std = low consistency

    # Compute Difficulty–Consistency Index
    #df['DCI'] = df['difficulty_score'] * df['consistency_norm']
    
    return test_pass

# --- SAB behavioral computations (per-user) ---
import numpy as np
import pandas as pd

def compute_sab_behavioral(df):
    df = df.copy()
    df = df[df['time_taken'] > 0]

    # core attempt-level metrics
    df['speed'] = df['correct_answers'] / df['time_taken']
    denom = _canonical_accuracy_denominator(df)
    df['accuracy_total'] = (df['marks'] / denom).clip(lower=0, upper=1).fillna(0)

    sab = df.groupby('user_id').agg(
        mean_speed=('speed', 'mean'),
        std_speed=('speed', 'std'),
        mean_accuracy=('accuracy_total', 'mean'),
        std_acc=('accuracy_total', 'std'),
        test_count=('test_id', 'count')
    ).reset_index()

    sab[['std_speed', 'std_acc']] = sab[['std_speed', 'std_acc']].fillna(0)

    # dampened consistency (numerically stable)
    eps = 1e-6
    sab['speed_consistency'] = sab.apply(
        lambda r: r['test_count'] / (r['test_count'] + (r['std_speed'] / max(r['mean_speed'], eps)) + 5),
        axis=1
    )
    sab['accuracy_consistency'] = sab.apply(
        lambda r: r['test_count'] / (r['test_count'] + (r['std_acc'] / max(r['mean_accuracy'], eps)) + 5),
        axis=1
    )

    sab['SAB_index'] = sab['mean_accuracy'] * sab['speed_consistency']

    # robust normalization (winsorize mean_speed)
    Q1 = sab['mean_speed'].quantile(0.25)
    Q3 = sab['mean_speed'].quantile(0.75)
    IQR = max(Q3 - Q1, 1e-6)
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    sab['mean_speed_w'] = sab['mean_speed'].clip(lower, upper)

    mu_s = sab['mean_speed_w'].mean()
    sd_s = sab['mean_speed_w'].std(ddof=0)
    sab['normalized_speed'] = (sab['mean_speed_w'] - mu_s) / (sd_s if sd_s > 0 else 1)

    mu_a = sab['mean_accuracy'].mean()
    sd_a = sab['mean_accuracy'].std(ddof=0)
    sab['normalized_accuracy'] = (sab['mean_accuracy'] - mu_a) / (sd_a if sd_a > 0 else 1)

    # evidence weight
    sab['weight'] = sab['test_count'] / (sab['test_count'] + 20)

    sab['robust_SAB_index'] = (
        ((sab['normalized_speed'] + sab['normalized_accuracy']) / 2)
        * sab['speed_consistency']
        * sab['accuracy_consistency']
        * sab['weight']
    )

    # FIXED: percentile scaling 0–100 (never negative)
    sab['robust_SAB_scaled'] = sab['robust_SAB_index'].rank(pct=True) * 100

    # rank: 1 = best
    sab['rank'] = sab['robust_SAB_index'].rank(ascending=False, method='average')

    return sab

def compute_user_pass_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['user_id','marks','pass_mark','pass_mark_effective']:
        if c not in df.columns:
            df[c] = np.nan

    df['marks'] = pd.to_numeric(df['marks'], errors='coerce')
    pass_source = pd.to_numeric(df['pass_mark_effective'], errors='coerce').fillna(pd.to_numeric(df['pass_mark'], errors='coerce'))
    if 'pass_mark_ambiguous' in df.columns:
        pass_source = pass_source.mask(df['pass_mark_ambiguous'])

    # passed defined only where pass_mark is usable under current DQ policy
    df['passed'] = np.where(pass_source.notna(), (df['marks'] >= pass_source).astype(int), np.nan)

    # pass ratio requested: marks / effective pass_mark
    df['pass_ratio'] = (df['marks'] / pass_source).replace([np.inf, -np.inf], np.nan)
    df['pass_ratio'] = df['pass_ratio'].clip(lower=0, upper=2.0)

    agg = df.groupby('user_id').agg(
        tests_passed=('passed', lambda x: int(np.nansum(x)) if x.notna().any() else 0),
        graded_attempts=('passed', lambda x: int(x.notna().sum())),
        avg_pass_ratio=('pass_ratio', 'mean'),
    ).reset_index()

    agg['tests_failed'] = (agg['graded_attempts'] - agg['tests_passed']).clip(lower=0)
    agg['pass_rate'] = np.where(agg['graded_attempts'] > 0, agg['tests_passed'] / agg['graded_attempts'], np.nan)

    # percent versions
    agg['pass_rate_pct'] = (agg['pass_rate'] * 100).round(1)
    agg['avg_pass_ratio_pct'] = (agg['avg_pass_ratio'] * 100).round(1)

    return agg


def compute_user_coverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes coverage-aware features per user:
    - at_risk_tests_count
    - low_evidence_tests_count
    - total_tests_covered
    - coverage_risk (Low/Medium/High)
    - coverage_factor (0..1) to penalize readiness probability

    Requires: user_id, test_id, accuracy_total, marks, pass_mark (optional), and attempts per test.
    """
    d = df.copy()

    # Ensure required columns exist
    for c in ["user_id", "test_id", "accuracy_total", "marks", "pass_mark", "pass_mark_effective"]:
        if c not in d.columns:
            d[c] = np.nan

    # passed flag (if pass_mark exists)
    pass_source = pd.to_numeric(d["pass_mark_effective"], errors="coerce").fillna(pd.to_numeric(d["pass_mark"], errors="coerce"))
    if "pass_mark_ambiguous" in d.columns:
        pass_source = pass_source.mask(d["pass_mark_ambiguous"])
    if pass_source.notna().any():
        d["passed"] = (pd.to_numeric(d["marks"], errors="coerce") >= pass_source).astype(int)
    else:
        d["passed"] = np.nan

    per_test = d.groupby(["user_id", "test_id"]).agg(
        attempts=("test_id", "count"),
        avg_accuracy=("accuracy_total", "mean"),
        pass_rate=("passed", "mean"),
    ).reset_index()

    # Per-test status rule (V1)
    def _status(row):
        if row["attempts"] < 2:
            return "Low evidence"
        # At risk if poor pass rate OR poor accuracy
        if pd.notna(row["pass_rate"]) and row["pass_rate"] < 0.5:
            return "At risk"
        if pd.notna(row["avg_accuracy"]) and row["avg_accuracy"] < 0.5:
            return "At risk"
        return "On track"

    per_test["status"] = per_test.apply(_status, axis=1)

    # Aggregate to user level
    user_cov = per_test.groupby("user_id").agg(
        total_tests_covered=("test_id", "nunique"),
        at_risk_tests_count=("status", lambda x: int((x == "At risk").sum())),
        low_evidence_tests_count=("status", lambda x: int((x == "Low evidence").sum())),
    ).reset_index()

    # Coverage risk and coverage factor
    def _risk_and_factor(row):
        total = row["total_tests_covered"] if row["total_tests_covered"] else 0
        at_risk = row["at_risk_tests_count"]
        low_ev = row["low_evidence_tests_count"]
        low_ev_rate = (low_ev / total) if total > 0 else 1.0

        # Risk labels
        if at_risk >= 2 or low_ev_rate >= 0.40:
            risk = "High"
        elif at_risk == 1 or low_ev_rate >= 0.20:
            risk = "Medium"
        else:
            risk = "Low"

        # Factor (0..1): penalize risk; keep it simple and explainable
        # - each at-risk test: -0.20
        # - each low evidence test: -0.10
        # + small base, clipped
        factor = 1.0 - (0.20 * at_risk) - (0.10 * low_ev)
        factor = float(np.clip(factor, 0.40, 1.0))  # never below 0.40

        return pd.Series({"coverage_risk": risk, "coverage_factor": factor})

    user_cov = pd.concat([user_cov, user_cov.apply(_risk_and_factor, axis=1)], axis=1)
    return user_cov
