# utils/metrics.py
import pandas as pd
import numpy as np
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

# basic metrics ------------------------------------------------
def compute_basic_metrics(df: pd.DataFrame) -> pd.DataFrame:
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
