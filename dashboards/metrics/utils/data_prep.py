# utils/data_prep.py
import pandas as pd
from utils.metrics import compute_basic_metrics2
from utils.institute_standardization import standardize_institute

REQUIRED_COLS = ["user_id", "test_id", "institute"]

def prepare_fact_table(df_raw: pd.DataFrame, mapping_path: str) -> pd.DataFrame:
    df = df_raw.copy()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = standardize_institute(df=df, column="institute", mapping_path=mapping_path)
    df["institute_std"] = df["institute_std"].fillna("Unknown").astype(str)

    df = compute_basic_metrics2(df)
    return df
