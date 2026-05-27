"""Compatibility wrapper for the legacy User Summary page.

The old page had drifted from the maintained DQ-gated implementation.
Keep this entry point so existing Streamlit navigation/bookmarks still work.
"""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).with_name("7_User_Summary.py")), run_name="__main__")
