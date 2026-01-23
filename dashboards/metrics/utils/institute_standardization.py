#pip install unidecode

import pandas as pd
import re
import unidecode

# -----------------------
# CONFIG
# -----------------------

UNKNOWN_TOKENS = {
    None, '', 'na', 'n/a', 'none', 'null', '-', 'unknown'
}

# -----------------------
# NORMALIZATION
# -----------------------

def normalize_text(s: str) -> str:
    s = unidecode.unidecode(str(s))
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def handle_unknowns(s):
    if s is None:
        return 'unknown'
    s = str(s).strip().lower()
    if s in UNKNOWN_TOKENS:
        return 'unknown'
    return s

# -----------------------
# MAIN STANDARDIZATION
# -----------------------

def standardize_institute(
    df: pd.DataFrame,
    column: str,
    mapping_path: str
) -> pd.DataFrame:
    """
    Standardizes institute column and adds data quality flags
    """

    df = df.copy()

    # Freeze raw
    df[f'{column}_raw'] = df[column]

    # Handle unknowns
    df[f'{column}_clean'] = df[column].apply(handle_unknowns)

    # Normalize
    df[f'{column}_norm'] = df[f'{column}_clean'].apply(normalize_text)

    # Load mapping
    mapping = pd.read_csv(mapping_path)
    mapping_dict = dict(
        zip(mapping['institute_norm'], mapping['institute_std'])
    )

    # Apply canonical mapping
    df[f'{column}_std'] = df[f'{column}_norm'].map(mapping_dict)
    df[f'{column}_std'] = df[f'{column}_std'].fillna(
        df[f'{column}_norm'].str.title()
    )

    # Quality flags
    df['is_unknown_institute'] = df[f'{column}_norm'] == 'unknown'
    df['institute_std_confidence'] = df[f'{column}_norm'].isin(mapping_dict)

    return df
