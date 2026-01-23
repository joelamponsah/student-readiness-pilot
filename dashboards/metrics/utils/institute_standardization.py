import pandas as pd
import re

UNKNOWN_TOKENS = {'', 'na', 'n/a', 'none', 'null', '-', 'unknown'}

def _normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def standardize_institute(df: pd.DataFrame, column: str, mapping_path: str) -> pd.DataFrame:
    df = df.copy()

    df[f"{column}_raw"] = df[column]

    # Unknown handling
    clean = df[f"{column}_raw"].fillna("").astype(str).str.strip().str.lower()
    clean = clean.where(~clean.isin(UNKNOWN_TOKENS), other="unknown")
    df[f"{column}_clean"] = clean

    # Normalize
    df[f"{column}_norm"] = df[f"{column}_clean"].apply(_normalize_text)

    # Mapping (expects columns: institute_norm, institute_std)
    mapping = pd.read_csv(mapping_path)
    mapping_dict = dict(zip(mapping["institute_norm"], mapping["institute_std"]))

    df["institute_std"] = df[f"{column}_norm"].map(mapping_dict)
    df["institute_std"] = df["institute_std"].fillna(df[f"{column}_norm"].str.title())
    df["institute_std"] = df["institute_std"].fillna("Unknown")

    df["is_unknown_institute"] = df[f"{column}_norm"].eq("unknown")
    df["institute_std_confidence"] = df[f"{column}_norm"].isin(mapping_dict)

    return df
