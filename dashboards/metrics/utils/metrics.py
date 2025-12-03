import pandas as pd
import numpy as np

def compute_basic_metrics(df):

    df = df.copy()
    df = df[df["time_taken"] > 0]

    df["speed_raw"] = df["attempted_questions"] / df["time_taken"]
    df["speed_acc_raw"] = df["correct_answers"] / df["time_taken"]
    df["accuracy"] = df["correct_answers"] / df["attempted_questions"]

    df["efficiency_ratio"] = df["speed_acc_raw"] / df["speed_raw"]

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
