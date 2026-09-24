"""Print the self-contained Reveal.js deck to a slide-sized PDF."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

PRESENTATION_DIR = Path(__file__).resolve().parent
HTML_PATH = PRESENTATION_DIR / "AtlasMSI_interactive.html"
PDF_PATH = PRESENTATION_DIR / "segmentation_model_full_conceptual.pdf"


def _find_browser() -> str:
    """Resolve Chrome Headless Shell from Quarto, PATH, or its default install path."""
    configured = os.environ.get("QUARTO_CHROMIUM")
    if configured and Path(configured).is_file():
        return configured

    for executable in ("chrome-headless-shell", "chromium", "google-chrome", "google-chrome-stable"):
        candidate = shutil.which(executable)
        if candidate:
            return candidate

    default_install = (
        Path.home()
        / ".local/share/quarto/chrome-headless-shell/chrome-headless-shell-linux64/chrome-headless-shell"
    )
    if default_install.is_file():
        return str(default_install)

    raise FileNotFoundError(
        "Chrome Headless Shell was not found. Install it with "
        "`quarto install chrome-headless-shell` or set QUARTO_CHROMIUM."
    )


def build_pdf() -> Path:
    """Print the standalone Reveal.js presentation through its PDF print stylesheet."""
    if not HTML_PATH.is_file():
        raise FileNotFoundError(f"Standalone HTML not found: {HTML_PATH}. Run build_standalone.py first.")

    browser = _find_browser()
    url = f"{HTML_PATH.as_uri()}?print-pdf"
    command = [
        browser,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        url,
    ]
    subprocess.run(command, cwd=PRESENTATION_DIR, check=True)
    return PDF_PATH


if __name__ == "__main__":
    print(build_pdf())
