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

    clean = df[f"{column}_raw"].fillna("").astype(str).str.strip().str.lower()
    clean = clean.where(~clean.isin(UNKNOWN_TOKENS), other="unknown")
    df[f"{column}_clean"] = clean

    df[f"{column}_norm"] = df[f"{column}_clean"].apply(_normalize_text)

    mapping = pd.read_csv(mapping_path)
    mapping.columns = mapping.columns.str.strip()

    # Allow flexible mapping column names
    #possible_norm = ["institute_norm", "norm", "normalized", "institute_normalized"]
    possible_std  = ["institute_std", "std", "standardized", "institute_standardized", "canonical_institute"]

    #norm_col = next((c for c in possible_norm if c in mapping.columns), None)
    std_col  = next((c for c in possible_std  if c in mapping.columns), None)

    if norm_col is None or std_col is None:
        raise KeyError(
            f"Mapping file must include norm+std columns. Found: {mapping.columns.tolist()}"
        )

    mapping_dict = dict(zip(mapping[norm_col].astype(str).str.strip(),
                            mapping[std_col].astype(str).str.strip()))

    df["institute_std"] = df[f"{column}_norm"].map(mapping_dict)
    df["institute_std"] = df["institute_std"].fillna(df[f"{column}_norm"].str.title())
    df["institute_std"] = df["institute_std"].fillna("Unknown")

    df["is_unknown_institute"] = df[f"{column}_norm"].eq("unknown")
    df["institute_std_confidence"] = df[f"{column}_norm"].isin(mapping_dict)

    return df
