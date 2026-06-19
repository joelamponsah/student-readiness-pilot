import pandas as pd
import streamlit as st

from utils.artifact_loader import get_artifact_status, load_artifact
from utils.ui_helpers import fmt_pct


st.set_page_config(page_title="Data Quality & Build Health", layout="wide")
st.title("Data Quality & Build Health")
st.caption("Build 99 control checks, export manifest status, and artifact-level DQ signals for v1.3-ext2.")


def _count_values(df: pd.DataFrame | None, column: str, value: str) -> int:
    if df is None or column not in df.columns:
        return 0
    return int((df[column].astype(str).str.upper() == value.upper()).sum())


def _safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _missing_column_note(df: pd.DataFrame | None, artifact_name: str, required_columns: list[str]) -> None:
    if df is None:
        st.warning(f"{artifact_name} is missing.")
        return
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.warning(f"{artifact_name} is missing expected columns: {missing}")


status = get_artifact_status()
dq = load_artifact("dq_summary")
build = load_artifact("build_summary")
manifest = load_artifact("export_manifest")
raw = load_artifact("raw_attempts")

_missing_column_note(
    manifest,
    "v13_export_manifest.csv",
    ["artifact", "file_exists", "status", "row_count", "column_count", "modified_at", "built_at"],
)
_missing_column_note(
    dq,
    "v13_dq_summary.csv",
    ["check_name", "check_category", "value", "severity", "notes"],
)

st.subheader("Build 99 Health")

manifest_pass = _count_values(manifest, "status", "PASS")
manifest_warn = _count_values(manifest, "status", "WARN")
manifest_fail = _count_values(manifest, "status", "FAIL")
dq_warn = _count_values(dq, "severity", "WARN")
dq_error = _count_values(dq, "severity", "ERROR")

built_at = ""
if manifest is not None and "built_at" in manifest.columns and not manifest.empty:
    built_at = str(manifest["built_at"].dropna().astype(str).iloc[0]) if manifest["built_at"].notna().any() else ""

cols = st.columns(6)
cols[0].metric("Manifest PASS", manifest_pass)
cols[1].metric("Manifest WARN", manifest_warn)
cols[2].metric("Manifest FAIL", manifest_fail)
cols[3].metric("DQ WARN", dq_warn)
cols[4].metric("DQ ERROR", dq_error)
cols[5].metric("Artifacts expected", len(status))

if built_at:
    st.info(f"Build 99 timestamp: {built_at}")

if manifest_fail > 0 or dq_error > 0:
    st.error("Build health has blocking failures. Review the manifest/DQ rows below before using dashboard outputs.")
elif manifest_warn > 0 or dq_warn > 0:
    st.warning("Build health has warnings. Outputs may still be usable, but inspect the warning rows before interpretation.")
else:
    st.success("Build health checks show no manifest failures or DQ errors.")

st.subheader("Artifact Loader Status")
status_cols = ["artifact", "filename", "exists", "valid", "rows", "columns", "missing_required"]
status_show = [col for col in status_cols if col in status.columns]
st.dataframe(status[status_show], use_container_width=True)

if manifest is not None:
    st.subheader("Export Manifest")
    manifest_display_cols = [
        "artifact",
        "file_exists",
        "status",
        "row_count",
        "column_count",
        "size_bytes",
        "modified_at",
        "required_columns_ok",
        "duplicate_grain_count",
        "notes",
        "built_at",
    ]
    manifest_display_cols = [col for col in manifest_display_cols if col in manifest.columns]

    status_filter = st.multiselect(
        "Manifest status filter",
        sorted(manifest["status"].dropna().astype(str).unique()) if "status" in manifest.columns else [],
        default=sorted(manifest["status"].dropna().astype(str).unique()) if "status" in manifest.columns else [],
    )
    manifest_view = manifest.copy()
    if status_filter and "status" in manifest_view.columns:
        manifest_view = manifest_view[manifest_view["status"].astype(str).isin(status_filter)]

    st.dataframe(manifest_view[manifest_display_cols], use_container_width=True)

    if "status" in manifest.columns:
        st.write("Manifest status distribution")
        st.dataframe(
            manifest["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="artifacts"),
            use_container_width=True,
        )
else:
    st.warning("v13_export_manifest.csv is missing. Run Build 99 in the notebook and refresh artifacts.")

if dq is not None:
    st.subheader("DQ Summary Artifact")
    dq_view = dq.copy()

    if "severity" in dq_view.columns:
        severity_options = sorted(dq_view["severity"].dropna().astype(str).unique())
        selected_severities = st.multiselect("DQ severity filter", severity_options, default=severity_options)
        if selected_severities:
            dq_view = dq_view[dq_view["severity"].astype(str).isin(selected_severities)]

        st.write("DQ severity distribution")
        st.dataframe(
            dq["severity"].value_counts(dropna=False).rename_axis("severity").reset_index(name="checks"),
            use_container_width=True,
        )

    if "check_category" in dq_view.columns:
        st.write("DQ category distribution")
        st.dataframe(
            dq_view["check_category"].value_counts(dropna=False).rename_axis("check_category").reset_index(name="checks"),
            use_container_width=True,
        )

    st.dataframe(dq_view, use_container_width=True)
else:
    st.warning("v13_dq_summary.csv is missing.")

if build is not None:
    st.subheader("Build Summary")
    st.dataframe(build, use_container_width=True)
else:
    st.warning("v13_build_summary.csv is missing.")

if raw is not None:
    st.subheader("Raw Attempt DQ Signals")
    c = st.columns(4)
    c[0].metric(
        "Missing institute rate",
        fmt_pct(raw.get("missing_institute_flag", pd.Series(dtype=float)).mean() * 100)
        if "missing_institute_flag" in raw.columns
        else "N/A",
    )
    c[1].metric(
        "Generic institute rate",
        fmt_pct(raw.get("generic_institute_flag", pd.Series(dtype=float)).mean() * 100)
        if "generic_institute_flag" in raw.columns
        else "N/A",
    )
    c[2].metric(
        "Multi-class rows",
        int(raw["multi_class_mapping_flag"].fillna(False).sum())
        if "multi_class_mapping_flag" in raw.columns
        else "N/A",
    )
    c[3].metric(
        "Low repeat evidence rows",
        int(raw["low_repeat_evidence_flag"].fillna(False).sum())
        if "low_repeat_evidence_flag" in raw.columns
        else "N/A",
    )

    for column in ["question_denominator_source", "dq_status", "attempt_status", "subscription_status"]:
        if column in raw.columns:
            st.write(f"{column} distribution")
            st.dataframe(
                raw[column].value_counts(dropna=False).rename_axis(column).reset_index(name="rows"),
                use_container_width=True,
            )
else:
    st.info("Raw attempts artifact is unavailable, so raw-attempt DQ distributions are hidden.")
