"""Sphinx configuration for the MSIAutoEncoderWrapper documentation."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "MSIAutoEncoderWrapper"
author = "MSIAutoEncoderWrapper contributors"

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 4

html_theme = "furo"
html_title = "MSIAutoEncoderWrapper"
