# utils/metrics.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def save_uploaded_df(df: pd.DataFrame, path="data/verify_df_fixed.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_data_from_disk_or_session(default_path="data/verify_df_fixed.csv"):
    # prefer session state if available (pages will read from it via streamlit)
    try:
        import streamlit as st
        if 'df' in st.session_state and st.session_state['df'] is not None:
            df = df.drop_duplicates(subset=['user_id', 'created_at'])
            return st.session_state['df']
    except Exception:
        pass

    # else load from disk if present
    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path, low_memory=False)
            return df
        except Exception:
            return None
    return None
# basic metrics ------------------------------------------------



# --- Speed & accuracy base features (idempotent) ---

def compute_basic_metrics2(df):
    df = df.copy()

    # Ensure numeric columns exist
    for c in ['attempted_questions','correct_answers','marks','time_taken','duration','no_of_questions','pass_mark']:
        if c not in df.columns:
            df[c] = np.nan

    # Coerce numeric
    for c in ['attempted_questions','correct_answers','marks','time_taken','duration','no_of_questions','pass_mark']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Guard zeros
    df['time_taken'] = df['time_taken'].replace(0, np.nan)
    df['duration'] = df['duration'].replace(0, np.nan)
    df['attempted_questions'] = df['attempted_questions'].replace(0, np.nan)
    df['no_of_questions'] = df['no_of_questions'].replace(0, np.nan)

    # Accuracy
    df['accuracy_attempt'] = (df['correct_answers'] / df['attempted_questions']).fillna(0)
    df['accuracy_total'] = (df['marks'] / df['no_of_questions']).fillna(0)

    # Speed per second
    df['speed_raw'] = df['attempted_questions'] / df['time_taken']      # attempted/sec
    df['adj_speed'] = df['correct_answers'] / df['time_taken']          # correct/sec

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
    df['accuracy_total'] = (df['marks'] / df['no_of_questions']).fillna(0)
    df['efficiency_ratio'] = df['accuracy_total'] / df['time_consumed'].replace(0, np.nan)
    agg = df.groupby('test_id').agg(
        mean_time=('time_taken','mean'),
        std_time=('time_taken','std'),
        mean_speed=('speed_marks','mean'),
        std_speed=('speed_marks','std'),
        mean_efficiency=('efficiency_ratio','mean'),
        mean_accuracy=('accuracy_total','mean'),
        pass_count=('marks', lambda x: (x >= df.loc[x.index,'pass_mark']).sum() if 'pass_mark' in df.columns else np.nan),
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
    if 'pass_mark' in df.columns:
        df['passed'] = (df['marks'] >= df['pass_mark']).astype(int)
    else:
        df['passed'] = (df['marks'] >= df['no_of_questions']).astype(int)
        
     # --- Handle inactive (no attempts) users ---
    df['inactive'] = (df['attempted_questions'] == 0).astype(int)
    df.loc[df['inactive'] == 1, ['marks', 'accuracy_total', 'accuracy_attempt', 'accuracy_norm']] = 0
    df['passed'] = np.where(df['marks'] >= df['pass_mark'], 1, 0)
    
        
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
    df['accuracy_total'] = (df['marks'] / df['no_of_questions']).fillna(0)

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
    for c in ['user_id','marks','pass_mark']:
        if c not in df.columns:
            df[c] = np.nan

    df['marks'] = pd.to_numeric(df['marks'], errors='coerce')
    df['pass_mark'] = pd.to_numeric(df['pass_mark'], errors='coerce')

    # passed defined as marks >= pass_mark (policy)
    df['passed'] = np.where(df['pass_mark'].notna(), (df['marks'] >= df['pass_mark']).astype(int), np.nan)

    # pass ratio requested: marks / pass_mark
    df['pass_ratio'] = (df['marks'] / df['pass_mark']).replace([np.inf, -np.inf], np.nan)
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
