# ===============================================
# 1. IMPORTS
# ===============================================
import pandas as pd
import numpy as np
from utils.metrics import compute_basic_metrics2
# ===============================================
# 2. SAMPLE DATA (Replace with your actual df)
# ===============================================
#df = pd.DataFrame({
 #   "student_id": [1,2,3,4],
  # "attempted_questions": [40, 50, 30, 60],
   # "correct_answers": [32, 40, 20, 50],
    #"time_taken": [20, 25, 15, 30]  # minutes
#})

df = compute_basic_metrics2(df)

# ===============================================
# 3. FEATURE ENGINEERING
# ===============================================

# Raw speed (questions per minute)
#df["speed_raw"] = df["attempted_questions"] / df["time_taken"]

# Accuracy
#df["accuracy"] = df["correct_answers"] / df["no_of_questions"]

# Adjusted speed = correct answers per minute
#df["speed_acc_raw"] = df["correct_answers"] / df["time_taken"]

# Optional percent score
df["percent_score"] = df["accuracy_total"] * 100

# ===============================================
# 4. LEADERBOARD SCORE
# ===============================================
# Weighted score (you can tune weights)
W_ACCURACY = 0.7
W_SPEED = 0.3

df["leaderboard_score"] = (
    (df["accuracy_total"] * W_ACCURACY) +
    (df["adj_speed] / df["adj.speed"].max() * W_SPEED)
)

# ===============================================
# 5. SORT LEADERBOARD
# ===============================================
leaderboard_df = df.sort_values("leaderboard_score", ascending=False).reset_index(drop=True)

leaderboard_df
