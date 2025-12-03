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

# utils/metrics.py
#import pandas as pd
#import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#import streamlit as st

#@st.cache_data
#def load_data_default(path):
  #  df = pd.read_csv(path, parse_dates=True, low_memory=False)
 #   return df

#def load_data_with_upload(default_path="data/verify_df_fixed.csv"):
    #uploaded = st.sidebar.file_uploader("Upload verify_df_fixed.csv (optional)", type=["csv"])
   # if uploaded is not None:
  #      df = pd.read_csv(uploaded, low_memory=False)
  #      return df
    # fallback to repo file if exists
   # try:
      #  df = load_data_default(default_path)
       # return df
   # except Exception:
       # return None
# basic metrics ------------------------------------------------
def compute_basic_metrics1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure numeric columns exist
    for c in ['attempted_questions','correct_answers','marks','time_taken','duration','no_of_questions','pass_mark']:
        if c not in df.columns:
            df[c] = np.nan
    # guard time
    df['time_taken'] = pd.to_numeric(df['time_taken'], errors='coerce')
    df['time_taken'] = df['time_taken'].replace(0, np.nan)
    # speeds
    df['speed_raw'] = df['attempted_questions'] / df['time_taken']
    df['speed_acc_raw'] = df['correct_answers'] / df['time_taken']
    df['speed_marks'] = df['marks'] / df['time_taken']
    # accuracy
    df['accuracy_total'] = (df['correct_answers'] / df['attempted_questions']).fillna(0)
    # relative time
    df['time_consumed'] = (df['time_taken'] / df['duration']).clip(0,1)
    df['speed_rel_time'] = ((df['duration'] - df['time_taken']) / df['duration']).clip(lower=0)
    # efficiency (guard divide-by-zero)
    df['efficiency_ratio'] = df['accuracy_total'] / df['time_consumed'].replace(0, np.nan)
    # basic cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df



# --- Speed & accuracy base features (idempotent) ---
def compute_basic_metrics2(df):
    df = df.copy()
    # ensure numeric
    for c in ['attempted_questions','correct_answers','marks','time_taken','duration']:
        if c not in df.columns:
            df[c] = np.nan
    # avoid zero time issues
    df['time_taken'] = df['time_taken'].replace(0, np.nan)
    df['speed_raw'] = df['attempted_questions'] / df['time_taken']
    df['adj_speed'] = df['correct_answers'] / df['time_taken']
    df['speed_marks'] = df['marks'] / df['time_taken']
    df['speed_rel_time'] = ((df['duration'] - df['time_taken']) / df['duration']).clip(lower=0)
    df['time_consumed'] = (df['time_taken'] / df['duration']).clip(0,1)
    # accuracy
    df['accuracy_attempt'] = (df['correct_answers'] / df['attempted_questions']).fillna(0)
    df['accuracy_total'] = (df['marks'] / df['no_of_questions']).fillna(0)

    # normalized speed (safe)
    scaler = MinMaxScaler()
    df['speed_norm'] = scaler.fit_transform(df[['speed_raw']].fillna(0))
    # efficiency ratio
    df['efficiency_ratio'] = df['accuracy_total'] / df['time_consumed'].replace(0, np.nan)
    return df

# --- Per-test analytics ---
def compute_test_analytics(df):
    df = df.copy()
    df = df[df['time_taken'] > 0]
    df['speed_marks'] = df['marks'] / df['time_taken']
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
    df = df.copy()
    # pass_flag per row
    if 'pass_mark' in df.columns:
        df['passed'] = (df['marks'] >= df['pass_mark']).astype(int)
    else:
        df['passed'] = (df['marks'] >= df['no_of_questions']).astype(int)
    test_pass = df.groupby('test_id').agg(
        pass_rate=('passed','mean'),
        mean_accuracy=('accuracy_total','mean'),
        std_accuracy=('accuracy_total','std'),
        takers=('user_id','nunique')
    ).reset_index()
    # DCI: closeness between mean_accuracy and (1 - difficulty), difficulty ~ 1-pass_rate
    test_pass['difficulty'] = 1 - test_pass['pass_rate']
    test_pass['DCI'] = 1 - (abs(test_pass['mean_accuracy'] - (1 - test_pass['difficulty'])) )
    # test stability
    test_pass['stability'] = 1 / (1 + test_pass['std_accuracy'].fillna(0))
    return test_pass

# --- SAB behavioral computations (per-user) ---
def compute_sab_behavioral(df):
    df = df.copy()
    df = df[df['time_taken'] > 0]
    df['speed'] = df['correct_answers'] / df['time_taken']
    sab = df.groupby('user_id').agg(
        mean_speed=('speed','mean'),
        std_speed=('speed','std'),
        mean_accuracy=('accuracy_total','mean'),
        std_acc=('accuracy_total','std'),
        test_count=('test_id','count')
    ).reset_index()
    sab[['std_speed','std_acc']] = sab[['std_speed','std_acc']].fillna(0)
    # dampened consistency penalizing low counts
    sab['speed_consistency'] = sab.apply(
        lambda r: r['test_count'] / (r['test_count'] + (r['std_speed'] / r['mean_speed'] if r['mean_speed']>0 else 1) + 5),
        axis=1
    )
    sab['accuracy_consistency'] = sab.apply(
        lambda r: r['test_count'] / (r['test_count'] + (r['std_acc'] / r['mean_accuracy'] if r['mean_accuracy']>0 else 1) + 5),
        axis=1
    )
    sab['SAB_index'] = sab['mean_accuracy'] * sab['speed_consistency']
    # robust normalization (winsorize mean_speed)
    Q1 = sab['mean_speed'].quantile(0.25); Q3 = sab['mean_speed'].quantile(0.75)
    IQR = max(Q3 - Q1, 1e-6)
    lower = Q1 - 1.5*IQR; upper = Q3 + 1.5*IQR
    sab['mean_speed_w'] = sab['mean_speed'].clip(lower, upper)
    mu_s = sab['mean_speed_w'].mean(); sd_s = sab['mean_speed_w'].std(ddof=0)
    sab['normalized_speed'] = (sab['mean_speed_w'] - mu_s) / (sd_s if sd_s>0 else 1)
    mu_a = sab['mean_accuracy'].mean(); sd_a = sab['mean_accuracy'].std(ddof=0)
    sab['normalized_accuracy'] = (sab['mean_accuracy'] - mu_a) / (sd_a if sd_a>0 else 1)
    sab['weight'] = sab['test_count'] / (sab['test_count'] + 20)
    sab['robust_SAB_index'] = (
        ((sab['normalized_speed'] + sab['normalized_accuracy'])/2)
        * sab['speed_consistency']
        * sab['accuracy_consistency']
        * sab['weight']
    )
    sab['rank'] = sab['robust_SAB_index'].rank(ascending=False)
    maxv = sab['robust_SAB_index'].max()
    sab['robust_SAB_scaled'] = (sab['robust_SAB_index'] / maxv * 100) if maxv and not np.isnan(maxv) else 0
    return sab


def compute_SAB(df):

    df = df.copy()

    user_group = df.groupby("user_id")

    sab = user_group.agg(
        mean_speed = ("speed_acc_raw", "mean"),
        std_speed  = ("speed_acc_raw", "std"),
        mean_accuracy = ("accuracy", "mean"),
        std_acc = ("accuracy", "std"),
        test_count = ("test_id", "count")
    ).reset_index()

    sab["std_speed"] = sab["std_speed"].fillna(0)
    sab["std_acc"] = sab["std_acc"].fillna(0)

    sab["speed_consistency"] = 1 / (1 + sab["std_speed"] / sab["mean_speed"])
    sab["accuracy_consistency"] = 1 / (1 + sab["std_acc"] / sab["mean_accuracy"])

    sab["SAB_index"] = sab["mean_accuracy"] * sab["speed_consistency"]

    # Robust SAB
    mu_speed = sab["mean_speed"].mean()
    sigma_speed = sab["mean_speed"].std()

    mu_acc = sab["mean_accuracy"].mean()
    sigma_acc = sab["mean_accuracy"].std()

    sab["normalized_speed"] = (sab["mean_speed"] - mu_speed) / sigma_speed
    sab["normalized_accuracy"] = (sab["mean_accuracy"] - mu_acc) / sigma_acc

    sab["weight"] = np.log1p(sab["test_count"])

    sab["robust_SAB_index"] = (
        ((sab["normalized_speed"] + sab["normalized_accuracy"]) / 2)
        * sab["speed_consistency"]
        * sab["accuracy_consistency"]
        * sab["weight"]
    )

    sab["rank"] = sab["robust_SAB_index"].rank(ascending=False)

    return sab
