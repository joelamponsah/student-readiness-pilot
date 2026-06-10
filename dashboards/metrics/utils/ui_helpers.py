"""UI helpers for v1.3-ext2 artifact pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def fmt_count(value) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{int(value):,}"
    except Exception:
        return "N/A"


def fmt_pct(value, decimals: int = 1) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        value = float(value)
        return f"{value:.{decimals}f}%"
    except Exception:
        return "N/A"


def fmt_date(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return "N/A"
    return str(dt.date())


def evidence_badge(level) -> str:
    if pd.isna(level):
        return "Evidence: unknown"
    text = str(level).strip()
    return f"Evidence: {text}" if text else "Evidence: unknown"


def show_missing_columns(name: str, missing: list[str]) -> None:
    if missing:
        st.warning(f"{name} is missing required columns: {missing}")


def show_dq_warning(message: str) -> None:
    st.warning(f"Data Quality Warning: {message}")


def optional_filter(df: pd.DataFrame, column: str, label: str | None = None) -> pd.DataFrame:
    if column not in df.columns:
        return df
    values = sorted(df[column].dropna().astype(str).unique().tolist())
    if not values:
        return df
    selected = st.multiselect(label or column, values, default=[])
    if selected:
        return df[df[column].astype(str).isin(selected)].copy()
    return df


def min_numeric_filter(df: pd.DataFrame, column: str, label: str | None = None, default: int = 0) -> pd.DataFrame:
    if column not in df.columns:
        return df
    values = pd.to_numeric(df[column], errors="coerce")
    max_value = int(values.max()) if values.notna().any() else default
    threshold = st.number_input(label or f"Minimum {column}", min_value=0, max_value=max(max_value, 0), value=default, step=1)
    return df[values.fillna(0) >= threshold].copy()


def dataframe_with_download(df: pd.DataFrame, label: str, filename: str) -> None:
    st.dataframe(df, use_container_width=True)
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

