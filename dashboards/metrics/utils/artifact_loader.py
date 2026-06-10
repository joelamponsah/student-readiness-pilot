"""Cached artifact loader for the v1.3-ext2 Streamlit dashboard."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
import sys
import gdown

import pandas as pd
import streamlit as st

from utils.artifact_contracts import ARTIFACTS


PRIMARY_ARTIFACT_DIR = Path("/content/drive/MyDrive/Learn Smarter framework/datasets/outputs/data")
FALLBACK_ARTIFACT_DIR = Path("data/processed")

# Use /tmp so downloaded Drive artifacts are not written into the GitHub repo.
ZIP_ROOT_DIR = Path("/tmp/lr_ext2_artifacts")
ZIP_ARTIFACT_DIR = ZIP_ROOT_DIR / "processed"
ZIP_DOWNLOAD_PATH = ZIP_ROOT_DIR / "artifacts.zip"


def _has_known_artifact(path: Path) -> bool:
    return path.exists() and any((path / spec["filename"]).exists() for spec in ARTIFACTS.values())


def primary_artifact_dirs() -> list[Path]:
    """Drive/env locations that should be checked before GitHub repo fallback."""
    paths: list[Path] = []

    env_path = os.getenv("LR_ARTIFACT_DIR")
    if env_path:
        env_dir = Path(env_path)
        paths.extend([env_dir, env_dir / "processed"])

    paths.extend([PRIMARY_ARTIFACT_DIR, PRIMARY_ARTIFACT_DIR / "processed"])
    return paths


def fallback_artifact_dirs() -> list[Path]:
    """Repo-local fallbacks checked only after Drive/ZIP options."""
    return [FALLBACK_ARTIFACT_DIR]


def candidate_artifact_dirs() -> list[Path]:
    """All directories, in final resolver priority order."""
    return primary_artifact_dirs() + fallback_artifact_dirs()


def resolve_artifact_dir() -> Path:
    """
    Resolve artifact directory in this order:

    1. LR_ARTIFACT_DIR
    2. LR_ARTIFACT_DIR/processed
    3. Colab Google Drive parent path
    4. Colab Google Drive processed child
    5. Google Drive ZIP via LR_ARTIFACT_ZIP_GDRIVE_ID
    6. GitHub/local data/processed fallback
    """

    # 1–4: preferred Drive/env paths
    for path in primary_artifact_dirs():
        if _has_known_artifact(path):
            return path

    # 5: Google Drive ZIP download/extract
    zip_path = _resolve_gdrive_zip_artifacts()
    if zip_path is not None and _has_known_artifact(zip_path):
        return zip_path

    # 6: repo-local fallback, for GitHub sample artifacts only
    for path in fallback_artifact_dirs():
        if _has_known_artifact(path):
            return path

    if not os.getenv("LR_ARTIFACT_ZIP_GDRIVE_ID"):
        st.info(
            "No known v1.3-ext2 artifact CSVs found. "
            "Set LR_ARTIFACT_DIR, LR_ARTIFACT_ZIP_GDRIVE_ID, or add sample files to data/processed."
        )

    # Return an existing directory for useful status output, even if empty.
    for path in candidate_artifact_dirs():
        if path.exists():
            return path

    return Path(os.getenv("LR_ARTIFACT_DIR", str(PRIMARY_ARTIFACT_DIR)))


@st.cache_resource(show_spinner=False)
def _resolve_gdrive_zip_artifacts() -> Path | None:
    file_id = os.getenv("LR_ARTIFACT_ZIP_GDRIVE_ID")
    if not file_id:
        return None

    if _has_known_artifact(ZIP_ARTIFACT_DIR):
        return ZIP_ARTIFACT_DIR

    ZIP_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    try:
    
    except ImportError:
        st.warning(
    "LR_ARTIFACT_ZIP_GDRIVE_ID is set, but `gdown` is not installed. "
    f"Python executable: {sys.executable}. "
    "Confirm gdown is in the root requirements.txt of the deployed branch."
)
        return None

    try:
        output = gdown.download(id=file_id, output=str(ZIP_DOWNLOAD_PATH), quiet=True)
    except Exception as exc:
        st.warning(f"Google Drive artifact ZIP download failed: {exc}")
        return None

    if not output or not ZIP_DOWNLOAD_PATH.exists():
        st.warning(
            "Google Drive artifact ZIP download failed. "
            "Check LR_ARTIFACT_ZIP_GDRIVE_ID and file sharing permissions."
        )
        return None

    try:
        # Clear extracted contents, but keep ZIP root.
        if ZIP_ARTIFACT_DIR.exists():
            shutil.rmtree(ZIP_ARTIFACT_DIR)

        ZIP_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(ZIP_DOWNLOAD_PATH) as zf:
            zf.extractall(ZIP_ARTIFACT_DIR)

    except zipfile.BadZipFile:
        st.warning("Google Drive artifact file downloaded, but it is not a valid ZIP archive.")
        return None
    except Exception as exc:
        st.warning(f"Google Drive artifact ZIP extraction failed: {exc}")
        return None

    _flatten_extracted_artifacts(ZIP_ARTIFACT_DIR)

    if not _has_known_artifact(ZIP_ARTIFACT_DIR):
        st.warning("Google Drive artifact ZIP extracted, but no known v1.3-ext2 artifact CSVs were found.")
        return None

    return ZIP_ARTIFACT_DIR


def _flatten_extracted_artifacts(base: Path) -> None:
    expected_filenames = {spec["filename"] for spec in ARTIFACTS.values()}

    for path in list(base.rglob("*.csv")):
        if path.name in expected_filenames and path.parent != base:
            target = base / path.name
            if not target.exists():
                shutil.copy2(path, target)


def artifact_path(name: str) -> Path:
    spec = ARTIFACTS[name]
    return resolve_artifact_dir() / spec["filename"]


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text, low_memory=False)


def available_artifacts() -> list[str]:
    base = resolve_artifact_dir()
    return [name for name, spec in ARTIFACTS.items() if (base / spec["filename"]).exists()]


def validate_artifact_columns(name: str, df: pd.DataFrame) -> dict:
    spec = ARTIFACTS[name]
    required = spec.get("required", [])
    optional = spec.get("optional", [])

    columns = set(df.columns)
    missing_required = [col for col in required if col not in columns]
    missing_optional = [col for col in optional if col not in columns]

    return {
        "artifact": name,
        "filename": spec["filename"],
        "valid": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


def load_artifact(name: str) -> pd.DataFrame | None:
    if name not in ARTIFACTS:
        raise KeyError(f"Unknown artifact: {name}")

    path = artifact_path(name)

    if not path.exists():
        return None

    try:
        return _read_csv_cached(str(path))
    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return None


def load_required_artifact(name: str) -> pd.DataFrame:
    df = load_artifact(name)

    if df is None:
        st.warning(f"Missing required artifact: {ARTIFACTS[name]['filename']}")
        st.stop()

    validation = validate_artifact_columns(name, df)

    if not validation["valid"]:
        st.warning(
            f"{ARTIFACTS[name]['filename']} is missing required columns: "
            f"{validation['missing_required']}"
        )

    return df


def get_artifact_status() -> pd.DataFrame:
    base = resolve_artifact_dir()
    rows = []

    for name, spec in ARTIFACTS.items():
        path = base / spec["filename"]
        exists = path.exists()

        row = {
            "artifact": name,
            "filename": spec["filename"],
            "description": spec["description"],
            "exists": exists,
            "path": str(path),
            "valid": False,
            "rows": None,
            "columns": None,
            "missing_required": [],
        }

        if exists:
            df = load_artifact(name)
            if df is not None:
                validation = validate_artifact_columns(name, df)
                row.update(
                    {
                        "valid": validation["valid"],
                        "rows": validation["row_count"],
                        "columns": validation["column_count"],
                        "missing_required": validation["missing_required"],
                    }
                )

        rows.append(row)

    return pd.DataFrame(rows)
