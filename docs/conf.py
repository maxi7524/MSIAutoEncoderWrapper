"""Sphinx configuration for the MSIAutoEncoderWrapper documentation."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "MSIAutoEncoderWrapper"
author = "MSIAutoEncoderWrapper contributors"

extensions = ["myst_nb"]
source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.header", "mystnb.unknown_mime_type"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 4
nb_execution_mode = "off"

html_theme = "furo"
html_title = "MSIAutoEncoderWrapper"
