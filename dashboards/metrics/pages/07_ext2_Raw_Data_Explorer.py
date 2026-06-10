import streamlit as st

from utils.artifact_contracts import ARTIFACTS
from utils.artifact_loader import load_artifact, validate_artifact_columns
from utils.ui_helpers import dataframe_with_download, optional_filter


st.set_page_config(page_title="Raw Data Explorer", layout="wide")
st.title("Raw Data Explorer")
st.caption("Technical debugging and evidence inspection only.")

artifact_name = st.selectbox("Artifact", list(ARTIFACTS.keys()), format_func=lambda name: ARTIFACTS[name]["filename"])
df = load_artifact(artifact_name)

if df is None:
    st.warning(f"Missing artifact: {ARTIFACTS[artifact_name]['filename']}")
    st.stop()

validation = validate_artifact_columns(artifact_name, df)
st.write("Required-column status:", "PASS" if validation["valid"] else "WARNING")
if validation["missing_required"]:
    st.warning(f"Missing required columns: {validation['missing_required']}")

cols = st.columns(3)
cols[0].metric("Rows", len(df))
cols[1].metric("Columns", len(df.columns))
cols[2].metric("Required missing", len(validation["missing_required"]))

st.write("Columns")
st.dataframe({"column": list(df.columns)}, use_container_width=True)

filtered = df.copy()
for col in ["institute_std", "class_id", "user_id", "test_id"]:
    filtered = optional_filter(filtered, col, col)

search = st.text_input("Search")
if search:
    mask = filtered.astype(str).apply(lambda s: s.str.contains(search, case=False, na=False)).any(axis=1)
    filtered = filtered[mask].copy()

dataframe_with_download(filtered.head(5000), "Download filtered preview CSV", f"{artifact_name}_preview.csv")

