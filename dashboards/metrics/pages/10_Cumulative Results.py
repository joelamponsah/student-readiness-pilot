import pandas as pd
import numpy as np
import streamlit as st
from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics2


# try to get df from session or disk
df = load_data_from_disk_or_session()
if df is None:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
else:
    df = compute_basic_metrics2(df)
st.title("Cumulative Results")
# ----------------------------------------
# LOAD & SORT DATA
# ----------------------------------------
#df = pd.read_csv("data/processed_data.csv")

# Ensure datetime parsing
df['created_at'] = pd.to_datetime(df['created_at'])

# Sort by user + timestamp
df = df.sort_values(['user_id', 'created_at']).reset_index(drop=True)
#df = df.sort_values(['user_id', 'created_at'])


df['time_taken'] = df['time_taken'].replace(0, 1e-6)

# ----------------------------------------
# BASIC PER-TEST METRICS
# ----------------------------------------

# Accuracy metrics
df['attempted_accuracy'] = df.apply(
    lambda x: x['correct_answers'] / x['attempted_questions']
    if x['attempted_questions'] > 0 else 0,
    axis=1
)

#personal accuracy
df['total_accuracy'] = df['correct_answers'] / df['no_of_questions']        #test accuracy

# Speed metrics
df['raw_speed'] = df['attempted_questions'] / df['time_taken']         # questions/sec
df['accurate_speed'] = df['correct_answers'] / df['time_taken']        # correct/sec
df['speed_marks'] = df['marks'] / df['time_taken']                     # marks per sec

# Time consumption ratios
df['relative_speed'] = (df['duration'] - df['time_taken']) / df['duration']
df['time_consumed'] = df['time_taken'] / df['duration']

# ----------------------------------------
# ATTEMPT INDEX PER USER
# ----------------------------------------
df['attempt_idx'] = df.groupby('user_id').cumcount() + 1

# ----------------------------------------
# CUMULATIVE METRICS (THE IMPORTANT PART)
# ----------------------------------------

# Cumulative totals
df['c_correct'] = df.groupby('user_id')['correct_answers'].cumsum()
df['c_attempted'] = df.groupby('user_id')['attempted_questions'].cumsum()
df['c_time'] = df.groupby('user_id')['time_taken'].cumsum()
df['c_total_questions'] = df.groupby('user_id')['no_of_questions'].cumsum()
df['c_duration'] = df.groupby('user_id')['duration'].cumsum()

# Cumulative Test Accuracy (main longitudinal metric)
df['c_total_accuracy'] = df['c_correct'] / df['c_total_questions']

#cumulative personal accuracy 
df['c_attempt_accuracy'] = df['c_correct'] / df['c_attempted']


# Cumulative Speed (seconds per question — lower = better)
df['c_time_consumed'] = df['c_time'] / df['c_duration'] 

# Optional: cumulative accurate speed (correct answers per second)
df['c_accurate_speed'] = df['c_correct'] / df['c_time']

# Optional: cumulative raw speed (questions per second)
df['c_raw_speed'] = df['c_attempted'] / df['c_time']

# ----------------------------------------
# CLEANUP FOR DASHBOARD
# ----------------------------------------
# These columns are the most useful for your Readiness dashboard
dashboard_df = df[[
    'user_id', 'created_at', 'attempt_idx',
    
    # basic accuracy
    'attempted_accuracy', 'total_accuracy',
    
    # basic speed metrics
    'raw_speed', 'accurate_speed', 'speed_marks',
    'relative_speed', 'time_consumed',
    
    # cumulative metrics
    'c_total_accuracy','c_attempt_accuracy', 'c_time_consumed', 
    'c_accurate_speed', 'c_raw_speed',
    
    # raw fields for reference
    'attempted_questions', 'correct_answers', 'time_taken', 'duration'
]]
dashboard_df = dashboard_df.drop_duplicates(subset=['user_id', 'created_at'])
st.subheader("Cumulative Results ")
st.dataframe(dashboard_df)
