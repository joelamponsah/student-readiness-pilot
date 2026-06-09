import numpy as np
import pandas as pd
import streamlit as st

from utils.metrics import get_v13_artifacts


st.set_page_config(page_title="Class Summary", layout="wide")
st.title("Class Summary")
st.caption("v1.3 Test / Exercise Readiness by class. BLS/ALS/CAS are proxy signals.")


# ---------------------------
# Helpers
# ---------------------------
def _normalise_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_metric_mean(df: pd.DataFrame, col: str | None, decimals: int = 2) -> str:
    if col is None or df is None or df.empty or col not in df.columns:
        return "N/A"
    vals = pd.to_numeric(df[col], errors="coerce")
    if not vals.notna().any():
        return "N/A"
    return f"{vals.mean():.{decimals}f}"


def _safe_metric_count_notna(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return "N/A"
    return f"{int(df[col].notna().sum()):,}"


def _safe_nunique(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(df[col].nunique(dropna=True))


def _filter_by_class_or_fallback_users(
    frame: pd.DataFrame | None,
    selected_class_id: str,
    selected_user_keys: set[str],
    frame_name: str,
) -> tuple[pd.DataFrame, bool]:
    if frame is None or frame.empty:
        return pd.DataFrame(), False

    df = frame.copy()
    if "class_id_std" in df.columns:
        class_key = _normalise_string_series(df["class_id_std"])
        return df.loc[class_key.eq(selected_class_id)].copy(), False

    if "user_id" in df.columns:
        st.warning(
            f"{frame_name} has no class_id_std column. Falling back to learner-based filtering; "
            "this may include tests taken outside the selected class."
        )
        user_key = df["user_id"].astype(str)
        return df.loc[user_key.isin(selected_user_keys)].copy(), True

    st.warning(f"{frame_name} has neither class_id_std nor user_id, so it cannot be filtered for this class.")
    return pd.DataFrame(), False


def _value_counts_table(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()
    out = (
        df[col]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="rows")
    )
    out.insert(0, "metric", label)
    return out


def _build_institute_context(df: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    """Build cautious institute context from selected-class source rows."""
    if df is None or df.empty or "institute_std" not in df.columns:
        empty = pd.DataFrame(columns=["institute_std", "rows", "share"])
        return "N/A", "missing", empty

    inst = df["institute_std"].fillna("Unknown").astype(str).str.strip()
    inst = inst.replace("", "Unknown")

    counts = inst.value_counts(dropna=False).rename_axis("institute_std").reset_index(name="rows")
    total_rows = int(counts["rows"].sum()) if not counts.empty else 0
    counts["share"] = counts["rows"] / total_rows if total_rows > 0 else np.nan
    counts = counts.sort_values(["rows", "institute_std"], ascending=[False, True], kind="mergesort")

    known = counts[~counts["institute_std"].str.lower().eq("unknown")].copy()
    if known.empty:
        return "Unknown", "low_no_known_institute", counts

    top = known.iloc[0]
    top_label = str(top["institute_std"])
    top_share = float(top["share"])
    known_count = int(len(known))

    if top_share >= 0.70 and known_count == 1:
        return top_label, "high_single_institute", counts
    if top_share >= 0.70:
        return f"{top_label} (dominant, mixed)", "medium_dominant_mixed", counts
    return "Mixed / unclear", "low_mixed_institutes", counts


# ---------------------------
# Load shared artifacts
# ---------------------------
raw_df, artifacts = get_v13_artifacts()
if raw_df is None or not artifacts:
    st.warning("No raw_attempts.csv input or shared v1.3 artifacts are available.")
    st.stop()

user_test_summary = artifacts.get("user_test_summary")
group_summary = artifacts.get("group_summary")
readiness_user = artifacts.get("readiness_user")
proxy_sequence = artifacts.get("proxy_sequence")
dq_attempts = artifacts.get("dq_attempts")
smoke_report = artifacts.get("smoke_report")

required_frames = {
    "user_test_summary": user_test_summary,
    "group_summary": group_summary,
    "readiness_user": readiness_user,
    "proxy_sequence": proxy_sequence,
    "dq_attempts": dq_attempts,
    "smoke_report": smoke_report,
}
missing_frames = [name for name, frame in required_frames.items() if frame is None]
if missing_frames:
    st.warning(f"The shared artifact bundle is incomplete for class summary inspection. Missing: {missing_frames}")
    st.stop()

if user_test_summary.empty or "class_id_std" not in user_test_summary.columns:
    st.warning("class_id_std is missing from user_test_summary, so class summaries cannot be built yet.")
    st.stop()

class_df = user_test_summary.loc[user_test_summary["class_id_std"].notna()].copy()
if class_df.empty:
    st.warning("No class_id_std values are available in user_test_summary.")
    st.stop()

class_df["class_id_std"] = _normalise_string_series(class_df["class_id_std"])
class_df = class_df[class_df["class_id_std"] != ""].copy()
if class_df.empty:
    st.warning("No usable class_id_std values are available in user_test_summary.")
    st.stop()

sab_col = _first_existing_column(
    readiness_user,
    ["robust_SAB_scaled", "robust_sab_scaled", "robust_SAB_index", "robust_SAB"],
)

st.markdown(
    """
### Boundary
- BLS / ALS / CAS are proxies only in v1.3.
- CAS Proxy is not true CAS yet.
- Difficulty / DCI is context, not score correction.
- Institute context is cautious because source class joins may be mixed.
- No topic / subject / year-group inference is performed here.
- No final unified readiness score is created.
"""
)

# ---------------------------
# Class selector
# ---------------------------
class_meta = (
    class_df.groupby("class_id_std", dropna=False)
    .agg(
        learner_count=("user_id", "nunique"),
        test_count=("test_id", "nunique"),
    )
    .reset_index()
)

class_meta["class_label"] = class_meta.apply(
    lambda r: (
        f"{r['class_id_std']} "
        f"(learners: {int(r['learner_count'])} | source tests: {int(r['test_count'])})"
    ),
    axis=1,
)
class_meta = class_meta.sort_values(
    ["learner_count", "test_count", "class_id_std"],
    ascending=[False, False, True],
    kind="mergesort",
)

selected_class_label = st.selectbox("Select class", class_meta["class_label"].tolist())
selected_class_id = class_meta.loc[class_meta["class_label"] == selected_class_label, "class_id_std"].iloc[0]

st.caption(
    "Class Summary only includes rows tagged with the selected class_id_std. "
    "Learner-level readiness is overall learner readiness for learners in this class. "
    "The dropdown source-test count is before the temporary class-test confidence filter."
)

# ---------------------------
# Class rows + temporary class-test confidence filter
# ---------------------------
class_rows_all = class_df[class_df["class_id_std"].eq(selected_class_id)].copy()
selected_user_keys_all = set(class_rows_all["user_id"].dropna().astype(str).unique().tolist())
selected_class_learner_count = len(selected_user_keys_all)

# Institute context is calculated from source class rows before heuristic filtering.
selected_institute_label, institute_confidence, institute_context = _build_institute_context(class_rows_all)

if {"test_id", "user_id"}.issubset(class_rows_all.columns) and selected_class_learner_count > 0:
    test_confidence = (
        class_rows_all.groupby("test_id", dropna=False)
        .agg(
            class_test_learner_count=("user_id", "nunique"),
            class_test_row_count=("user_id", "size"),
        )
        .reset_index()
    )
    test_confidence["class_test_learner_share"] = test_confidence["class_test_learner_count"] / selected_class_learner_count
    test_confidence["class_test_included"] = (
        (test_confidence["class_test_learner_count"] >= 3)
        | (test_confidence["class_test_learner_share"] >= 0.20)
    )
    included_test_ids = set(test_confidence.loc[test_confidence["class_test_included"], "test_id"].tolist())
    class_rows = class_rows_all[class_rows_all["test_id"].isin(included_test_ids)].copy()

    excluded_test_count = int((~test_confidence["class_test_included"]).sum())
    excluded_row_count = int(len(class_rows_all) - len(class_rows))

    st.info(
        "Heuristic class-test filter is active. "
        "The page only includes tests attempted by at least 3 learners or at least 20% of selected class learners. "
        "This is not official class assignment data."
    )
    st.caption(
        f"Filtered out {excluded_test_count:,} low-confidence test(s) "
        f"and {excluded_row_count:,} learner-test row(s) for this class."
    )

    with st.expander("Low-confidence tests filtered out"):
        filtered_out_tests = test_confidence.loc[~test_confidence["class_test_included"]].sort_values(
            ["class_test_learner_count", "class_test_learner_share"],
            ascending=[False, False],
            kind="mergesort",
        )
        st.dataframe(filtered_out_tests, use_container_width=True)
else:
    class_rows = class_rows_all.copy()
    test_confidence = pd.DataFrame()
    st.warning(
        "Could not apply heuristic class-test filter because test_id/user_id is missing "
        "or selected class learner count is zero."
    )

if class_rows.empty:
    st.warning(
        "No class rows remain after the heuristic class-test filter. "
        "Lower the threshold later only after validating class-test assignment data."
    )
    st.stop()

selected_user_keys = set(class_rows["user_id"].dropna().astype(str).unique().tolist())
included_test_ids = set(class_rows["test_id"].dropna().tolist()) if "test_id" in class_rows.columns else set()

readiness_user_work = readiness_user.copy()
if "user_id" in readiness_user_work.columns:
    readiness_user_work["_user_id_key"] = readiness_user_work["user_id"].astype(str)
    readiness_class = readiness_user_work[readiness_user_work["_user_id_key"].isin(selected_user_keys)].copy()
else:
    readiness_class = pd.DataFrame()

proxy_class_all, proxy_fallback_used = _filter_by_class_or_fallback_users(
    proxy_sequence,
    selected_class_id,
    selected_user_keys_all,
    "proxy_sequence",
)
dq_class_all, dq_fallback_used = _filter_by_class_or_fallback_users(
    dq_attempts,
    selected_class_id,
    selected_user_keys_all,
    "dq_attempts",
)

if included_test_ids and "test_id" in proxy_class_all.columns:
    proxy_class = proxy_class_all[proxy_class_all["test_id"].isin(included_test_ids)].copy()
else:
    proxy_class = proxy_class_all.copy()

if included_test_ids and "test_id" in dq_class_all.columns:
    dq_class = dq_class_all[dq_class_all["test_id"].isin(included_test_ids)].copy()
else:
    dq_class = dq_class_all.copy()

st.subheader(f"Selected Class: {selected_class_id}")
source_test_count = _safe_nunique(class_rows_all, "test_id")
filtered_test_count = _safe_nunique(class_rows, "test_id")
source_learner_count = _safe_nunique(class_rows_all, "user_id")
filtered_learner_count = _safe_nunique(class_rows, "user_id")
st.caption(
    f"Source view: {source_learner_count:,} learners | {source_test_count:,} tests. "
    f"Filtered class-readiness view: {filtered_learner_count:,} learners | {filtered_test_count:,} tests."
)
st.caption("Custom selected-test groups will be added after source-backed class summaries are validated.")

if proxy_fallback_used or dq_fallback_used:
    st.info(
        "One or more artifacts used learner-based fallback filtering because class_id_std was unavailable. "
        "Validate these results carefully."
    )

# ---------------------------
# Institute context
# ---------------------------
st.subheader("Institute Context")
known_mask = ~institute_context["institute_std"].str.lower().eq("unknown") if not institute_context.empty else pd.Series(dtype=bool)
known_institute_count = int(known_mask.sum()) if not institute_context.empty else 0
known_coverage = float(institute_context.loc[known_mask, "share"].sum()) * 100 if not institute_context.empty else np.nan

inst_cols = st.columns(4)
inst_cols[0].metric("Institute", selected_institute_label)
inst_cols[1].metric("Institute confidence", institute_confidence)
inst_cols[2].metric("Known institutes", f"{known_institute_count:,}" if not institute_context.empty else "N/A")
inst_cols[3].metric("Known institute coverage", f"{known_coverage:.1f}%" if pd.notna(known_coverage) else "N/A")

with st.expander("Institute distribution for selected class"):
    if not institute_context.empty:
        st.dataframe(institute_context, use_container_width=True)
    else:
        st.info("No institute context is available for this class.")

# ---------------------------
# Overview KPIs
# ---------------------------
st.subheader("Class Overview")

overview1 = st.columns(4)
overview1[0].metric("Institute", selected_institute_label)
overview1[1].metric("Learners", f"{class_rows['user_id'].nunique():,}" if "user_id" in class_rows.columns else "N/A")
overview1[2].metric("Tests/exercises", f"{class_rows['test_id'].nunique():,}" if "test_id" in class_rows.columns else "N/A")
overview1[3].metric("User-test groups", f"{len(class_rows):,}")

overview2 = st.columns(4)
overview2[0].metric("Avg readiness probability %", _safe_metric_mean(readiness_class, "readiness_probability_pct"))
overview2[1].metric("BLS rows", _safe_metric_count_notna(class_rows, "bls_score_pct"))
overview2[2].metric("Current ALS rows", _safe_metric_count_notna(class_rows, "current_als_score_pct"))
overview2[3].metric("Avg robust SAB", _safe_metric_mean(readiness_class, sab_col) if sab_col else "N/A")

overview3 = st.columns(4)
overview3[0].metric("Potential ALS rows", _safe_metric_count_notna(class_rows, "potential_als_score_pct"))
overview3[1].metric("Avg BLS %", _safe_metric_mean(class_rows, "bls_score_pct"))
overview3[2].metric("Avg Current ALS %", _safe_metric_mean(class_rows, "current_als_score_pct"))
overview3[3].metric("Avg learning gain %", _safe_metric_mean(class_rows, "learning_gain_pct"))

overview4 = st.columns(4)
overview4[0].metric("Avg CAS proxy %", _safe_metric_mean(class_rows, "cas_proxy_score_pct"))

if sab_col:
    st.caption(f"Robust SAB source column: `{sab_col}` from readiness_user.")
    if sab_col in readiness_user.columns:
        st.caption(f"robust SAB non-null users in readiness_user: {int(readiness_user[sab_col].notna().sum()):,}")
else:
    st.caption("Robust SAB source column was not found in readiness_user.")

# ---------------------------
# Readiness distribution
# ---------------------------
st.subheader("Class Readiness Distribution")
dist_frames = []
for column in ["exam_status", "risk_band", "coverage_risk"]:
    dist = _value_counts_table(readiness_class, column, column)
    if not dist.empty:
        dist_frames.append(dist)

if dist_frames:
    st.dataframe(pd.concat(dist_frames, ignore_index=True, sort=False), use_container_width=True)
else:
    st.info("No readiness distribution fields are available for this class.")

st.caption("Readiness distribution uses learner-level overall readiness for learners in this class.")

# ---------------------------
# Class test / exercise table
# ---------------------------
st.subheader("Class Test / Exercise Table")
group_cols = ["test_id"]
if "test_name" in class_rows.columns:
    group_cols.append("test_name")

test_table = class_rows.groupby(group_cols, dropna=False).agg(learner_count=("user_id", "nunique")).reset_index()

mean_cols = {
    "mean_bls_score_pct": "bls_score_pct",
    "mean_current_als_score_pct": "current_als_score_pct",
    "mean_potential_als_score_pct": "potential_als_score_pct",
    "mean_learning_gain_pct": "learning_gain_pct",
    "cas_proxy_score_pct": "cas_proxy_score_pct",
}
for out_col, src_col in mean_cols.items():
    if src_col in class_rows.columns:
        tmp = class_rows.groupby(group_cols, dropna=False)[src_col].mean().reset_index(name=out_col)
        test_table = test_table.merge(tmp, on=group_cols, how="left")
    else:
        test_table[out_col] = np.nan

if "attempt_count" in class_rows.columns:
    repeated = (
        class_rows.groupby(group_cols, dropna=False)["attempt_count"]
        .apply(lambda s: int((pd.to_numeric(s, errors="coerce") >= 2).sum()))
        .reset_index(name="repeated_group_count")
    )
    test_table = test_table.merge(repeated, on=group_cols, how="left")
else:
    test_table["repeated_group_count"] = np.nan

if "proxy_evidence_band" in class_rows.columns:
    evidence = (
        class_rows.groupby(group_cols, dropna=False)["proxy_evidence_band"]
        .agg(
            high_rate=lambda s: float((_normalise_string_series(s).str.lower() == "high").mean()),
            medium_rate=lambda s: float((_normalise_string_series(s).str.lower() == "medium").mean()),
            low_rate=lambda s: float((_normalise_string_series(s).str.lower() == "low").mean()),
        )
        .reset_index()
    )
    test_table = test_table.merge(evidence, on=group_cols, how="left")
else:
    test_table["high_rate"] = np.nan
    test_table["medium_rate"] = np.nan
    test_table["low_rate"] = np.nan

for col in ["difficulty_label", "DCI", "test_stability"]:
    if col in class_rows.columns:
        context = class_rows[group_cols + [col]].drop_duplicates(group_cols, keep="first")
        test_table = test_table.merge(context, on=group_cols, how="left")

if "institute_std" in class_rows.columns:
    institute_context_by_test = (
        class_rows[group_cols + ["institute_std"]]
        .dropna(subset=["institute_std"])
        .drop_duplicates(group_cols, keep="first")
    )
    test_table = test_table.merge(institute_context_by_test, on=group_cols, how="left")

display_cols = [
    c for c in [
        "test_id",
        "test_name",
        "institute_std",
        "learner_count",
        "mean_bls_score_pct",
        "mean_current_als_score_pct",
        "mean_potential_als_score_pct",
        "mean_learning_gain_pct",
        "cas_proxy_score_pct",
        "repeated_group_count",
        "high_rate",
        "medium_rate",
        "low_rate",
        "difficulty_label",
        "DCI",
        "test_stability",
    ]
    if c in test_table.columns
]
st.dataframe(test_table[display_cols].sort_values(["learner_count", "test_id"], ascending=[False, True], kind="mergesort"), use_container_width=True)

# ---------------------------
# Learner table
# ---------------------------
st.subheader("Learner Table")
learner_base_agg = {"number_of_tests": ("test_id", "nunique")}
learner_base_agg["learner_id_display"] = ("learner_id_display", "first") if "learner_id_display" in class_rows.columns else ("user_id", "first")
if "institute_std" in class_rows.columns:
    learner_base_agg["institute_std"] = ("institute_std", "first")

learner_table = class_rows.groupby("user_id", dropna=False).agg(**learner_base_agg).reset_index()

learner_mean_cols = {
    "avg_bls_score_pct": "bls_score_pct",
    "avg_current_als_score_pct": "current_als_score_pct",
    "avg_potential_als_score_pct": "potential_als_score_pct",
    "avg_learning_gain_pct": "learning_gain_pct",
    "avg_cas_proxy_score_pct": "cas_proxy_score_pct",
}
for out_col, src_col in learner_mean_cols.items():
    if src_col in class_rows.columns:
        tmp = class_rows.groupby("user_id", dropna=False)[src_col].mean().reset_index(name=out_col)
        learner_table = learner_table.merge(tmp, on="user_id", how="left")

readiness_merge_cols = ["user_id"]
for col in ["readiness_probability_pct", "exam_status", "risk_band"]:
    if col in readiness_class.columns:
        readiness_merge_cols.append(col)
if sab_col and sab_col in readiness_class.columns:
    readiness_merge_cols.append(sab_col)

if len(readiness_merge_cols) > 1:
    readiness_for_merge = readiness_class[readiness_merge_cols].drop_duplicates("user_id", keep="first")
    learner_table = learner_table.merge(readiness_for_merge, on="user_id", how="left")

if sab_col and sab_col in learner_table.columns:
    learner_table = learner_table.rename(columns={sab_col: "robust_sab"})

dq_learner = pd.DataFrame({"user_id": class_rows["user_id"].drop_duplicates().tolist()})
if not dq_class.empty and "user_id" in dq_class.columns:
    if "finished_at" in dq_class.columns:
        finished_missing = dq_class.groupby("user_id", dropna=False)["finished_at"].apply(lambda s: int(s.isna().sum())).reset_index(name="missing_finished_at_count")
        dq_learner = dq_learner.merge(finished_missing, on="user_id", how="left")
    if "completion_status" in dq_class.columns:
        unknown_usable = dq_class.groupby("user_id", dropna=False)["completion_status"].apply(lambda s: int((s == "unknown_but_usable").sum())).reset_index(name="unknown_but_usable_count")
        dq_learner = dq_learner.merge(unknown_usable, on="user_id", how="left")

learner_table = learner_table.merge(dq_learner, on="user_id", how="left")
for col in ["missing_finished_at_count", "unknown_but_usable_count"]:
    if col in learner_table.columns:
        learner_table[col] = learner_table[col].fillna(0).astype(int)

learner_display_cols = [
    c for c in [
        "learner_id_display",
        "institute_std",
        "user_id",
        "readiness_probability_pct",
        "exam_status",
        "risk_band",
        "robust_sab",
        "number_of_tests",
        "avg_bls_score_pct",
        "avg_current_als_score_pct",
        "avg_potential_als_score_pct",
        "avg_learning_gain_pct",
        "avg_cas_proxy_score_pct",
        "missing_finished_at_count",
        "unknown_but_usable_count",
    ]
    if c in learner_table.columns
]

sort_cols = [c for c in ["number_of_tests", "avg_current_als_score_pct"] if c in learner_table.columns]
learner_table_display = learner_table[learner_display_cols].sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort") if sort_cols else learner_table[learner_display_cols]
st.dataframe(learner_table_display, use_container_width=True)

# ---------------------------
# Learner x Test comparison
# ---------------------------
st.subheader("Learner × Test Comparison")
learner_test_cols = [
    c for c in [
        "learner_id_display",
        "institute_std",
        "user_id",
        "test_id",
        "test_name",
        "bls_score_pct",
        "current_als_score_pct",
        "potential_als_score_pct",
        "learning_gain_pct",
        "potential_gain_pct",
        "cas_proxy_score_pct",
        "proxy_evidence_band",
        "completion_status_mix",
        "difficulty_label",
        "DCI",
        "test_stability",
    ]
    if c in class_rows.columns
]
if learner_test_cols:
    learner_test_table = class_rows[learner_test_cols].copy()
    sort_cols = [c for c in ["learner_id_display", "user_id", "test_id"] if c in learner_test_table.columns]
    if sort_cols:
        learner_test_table = learner_test_table.sort_values(sort_cols, kind="mergesort")
    st.dataframe(learner_test_table, use_container_width=True)
else:
    st.info("No learner-test comparison columns are available for this class.")

# ---------------------------
# Attempt trace
# ---------------------------
with st.expander("Attempt trace for selected class"):
    trace = proxy_class.copy()
    trace_candidate_cols = [
        "learner_id_display",
        "institute_std",
        "user_id",
        "test_id",
        "test_name",
        "created_at",
        "marks",
        "accuracy_total",
        "v13_score_pct",
        "completion_status",
        "dq_included",
        "dq_bucket",
        "dq_eligible_published",
        "dq_eligible_proxy_sequence",
    ]
    trace_cols = list(dict.fromkeys(col for col in trace_candidate_cols if col in trace.columns))
    if trace_cols:
        sort_cols = [c for c in ["user_id", "test_id", "created_at"] if c in trace.columns]
        trace_display = trace[trace_cols].copy()
        if sort_cols:
            trace_display = trace_display.sort_values(sort_cols, kind="mergesort")
        st.dataframe(trace_display, use_container_width=True)
    else:
        st.info("No trace fields are available for this class.")

# ---------------------------
# DQ caveats
# ---------------------------
st.subheader("DQ Caveats")
dq_class_total = len(dq_class)
dq_included = int(dq_class["dq_included"].fillna(False).sum()) if not dq_class.empty and "dq_included" in dq_class.columns else 0
dq_excluded = int((~dq_class["dq_included"].fillna(False)).sum()) if not dq_class.empty and "dq_included" in dq_class.columns else 0
missing_finished = int(dq_class["finished_at"].isna().sum()) if not dq_class.empty and "finished_at" in dq_class.columns else 0
unknown_but_usable = int((dq_class["completion_status"] == "unknown_but_usable").sum()) if not dq_class.empty and "completion_status" in dq_class.columns else 0

dq_cols = st.columns(4)
dq_cols[0].metric("Total rows", f"{dq_class_total:,}")
dq_cols[1].metric("Included rows", f"{dq_included:,}")
dq_cols[2].metric("Excluded rows", f"{dq_excluded:,}")
dq_cols[3].metric("Missing finished_at", f"{missing_finished:,}")

dq_summary = pd.DataFrame([
    {"measure": "unknown_but_usable rows", "value": unknown_but_usable},
    {"measure": "missing finished_at rows", "value": missing_finished},
])
st.dataframe(dq_summary, use_container_width=True)

if not dq_class.empty and "exclusion_reason" in dq_class.columns:
    exclusion_filter = dq_class["dq_bucket"].eq("excluded") if "dq_bucket" in dq_class.columns else ~dq_class["dq_included"].fillna(False)
    exclusion_counts = dq_class.loc[exclusion_filter, "exclusion_reason"].value_counts(dropna=False).rename_axis("reason").reset_index(name="rows")
    if not exclusion_counts.empty:
        st.subheader("Exclusion Reasons")
        st.dataframe(exclusion_counts, use_container_width=True)

# ---------------------------
# Smoke report
# ---------------------------
with st.expander("Smoke report"):
    if smoke_report is not None and not smoke_report.empty:
        st.dataframe(smoke_report.T, use_container_width=True)
    else:
        st.info("No smoke_report is available.")

with st.expander("Group summary preview"):
    if group_summary is not None and not group_summary.empty:
        group_preview = group_summary.copy()
        if "class_id_std" in group_preview.columns:
            group_preview["_class_id_key"] = _normalise_string_series(group_preview["class_id_std"])
            group_preview = group_preview[group_preview["_class_id_key"].eq(selected_class_id)].drop(columns=["_class_id_key"])
        if included_test_ids and "test_id" in group_preview.columns:
            group_preview = group_preview[group_preview["test_id"].isin(included_test_ids)]

        preferred_group_cols = [
            "group_level",
            "group_value",
            "class_id_std",
            "institute_std",
            "subscriber_id",
            "test_id",
            "learner_count",
            "repeated_group_count",
            "mean_bls_score_pct",
            "mean_current_als_score_pct",
            "mean_potential_als_score_pct",
            "mean_learning_gain_pct",
            "cas_proxy_score_pct",
            "formula_readiness_avg",
            "robust_SAB_avg",
            "high_evidence_rate",
            "medium_evidence_rate",
            "low_evidence_rate",
            "difficulty_label",
            "DCI",
            "test_stability",
        ]
        display_group_cols = [c for c in preferred_group_cols if c in group_preview.columns]
        if display_group_cols:
            st.dataframe(group_preview[display_group_cols].head(100), use_container_width=True)
        else:
            st.dataframe(group_preview.head(100), use_container_width=True)
    else:
        st.info("No group_summary is available.")

st.caption("This page consumes shared v1.3 artifacts only. Page-level aggregation is for display only.")
