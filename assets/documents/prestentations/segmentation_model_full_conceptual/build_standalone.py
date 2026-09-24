"""Build a portable, interactive, single-file Reveal.js presentation."""

from __future__ import annotations

import base64
import hashlib
import html
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import unquote

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PRESENTATION_DIR = Path(__file__).resolve().parent
OUTPUT_NAME = "AtlasMSI_interactive.html"
BUILD_NAME = "AtlasMSI_interactive_build.html"
PLOTLY_URL = "https://cdn.plot.ly/plotly-3.7.0.min.js"
PLOTLY_SHA256_BASE64 = "jvTGqxNp8AGWEcvNLVuKr+8j5dGe9Yw51LQkmDH+IYA="

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from msi_autoencoder_wrapper.utils.logger import get_custom_logger  # noqa: E402

logger = get_custom_logger(__name__)

PLOTLY_SCRIPT_PATTERN = re.compile(
    r'<script\b(?=[^>]*\bsrc=["\']https://cdn\.plot\.ly/plotly-3\.7\.0\.min\.js'
    r'["\'])[^>]*>\s*</script>',
    re.IGNORECASE | re.DOTALL,
)
EMBEDDED_IFRAME_PATTERN = re.compile(
    r'(?P<prefix><iframe\b[^>]*?)\s+'
    r'src="(?P<uri>data:text/html,[^"]+)"(?P<suffix>[^>]*)>',
    re.IGNORECASE | re.DOTALL,
)


def _download_plotly() -> str:
    """Download and verify the pinned Plotly browser bundle.

    :return: Verified UTF-8 Plotly JavaScript source.
    :rtype: str
    :raises OSError: If the asset cannot be downloaded.
    :raises ValueError: If its content does not match the pinned digest.
    """
    request = urllib.request.Request(
        PLOTLY_URL,
        headers={"User-Agent": "AtlasMSI-standalone-builder"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        plotly_bytes = response.read()

    digest = base64.b64encode(hashlib.sha256(plotly_bytes).digest()).decode("ascii")
    if digest != PLOTLY_SHA256_BASE64:
        raise ValueError("Downloaded Plotly bundle does not match the pinned SHA-256.")

    plotly_source = plotly_bytes.decode("utf-8")
    if "</script" in plotly_source.lower():
        raise ValueError("Plotly source contains an unsafe closing script token.")
    return plotly_source


def _make_iframe_self_contained(document: str, plotly_source: str) -> str:
    """Inline the chart library and convert its data URL frame to ``srcdoc``.

    :param document: Quarto-rendered standalone HTML.
    :type document: str
    :param plotly_source: Verified Plotly JavaScript source.
    :type plotly_source: str
    :return: HTML with the chart frame executable from a local file.
    :rtype: str
    :raises ValueError: If Quarto's expected embedded iframe is missing.
    """
    iframe_match = EMBEDDED_IFRAME_PATTERN.search(document)
    if iframe_match is None:
        raise ValueError("Quarto output does not contain the expected embedded chart iframe.")

    nested_document = unquote(iframe_match.group("uri").partition(",")[2])
    nested_document, script_count = PLOTLY_SCRIPT_PATTERN.subn(
        lambda _: f"<script>\n{plotly_source}\n</script>",
        nested_document,
        count=1,
    )
    if script_count != 1:
        raise ValueError("Expected exactly one Plotly CDN script in the chart iframe.")
    if "Plotly.newPlot" not in nested_document:
        raise ValueError("The embedded chart does not contain its Plotly figure code.")

    replacement = (
        iframe_match.group("prefix")
        + ' srcdoc="'
        + html.escape(nested_document, quote=True)
        + '"'
        + iframe_match.group("suffix")
        + ">"
    )
    return document[: iframe_match.start()] + replacement + document[iframe_match.end() :]


def build_standalone() -> Path:
    """Render the presentation and write one self-contained HTML file.

    :return: Path to the generated HTML file.
    :rtype: pathlib.Path
    :raises subprocess.CalledProcessError: If Quarto rendering fails.
    :raises OSError: If the output cannot be read or written.
    :raises ValueError: If required presentation resources are missing.
    """
    logger.info("Downloading and verifying Plotly JavaScript.")
    plotly_source = _download_plotly()

    build_path = PRESENTATION_DIR / BUILD_NAME
    output_path = PRESENTATION_DIR / OUTPUT_NAME
    command = [
        "quarto",
        "render",
        "slides.qmd",
        "--to",
        "revealjs",
        "--output",
        BUILD_NAME,
        "-M",
        "embed-resources:true",
        "--quiet",
    ]
    logger.info("Rendering self-contained Reveal.js resources with Quarto.")
    try:
        result = subprocess.run(
            command,
            cwd=PRESENTATION_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error(
                "Quarto render failed with exit code %s: %s",
                result.returncode,
                result.stderr[-4000:],
            )
            raise subprocess.CalledProcessError(
                result.returncode,
                command,
                output=result.stdout,
                stderr=result.stderr,
            )

        rendered_html = build_path.read_text(encoding="utf-8")
        standalone_html = _make_iframe_self_contained(rendered_html, plotly_source)
        temporary_path = output_path.with_suffix(".tmp")
        temporary_path.write_text(standalone_html, encoding="utf-8")
        temporary_path.replace(output_path)
    finally:
        build_path.unlink(missing_ok=True)

    logger.info("Wrote standalone interactive HTML to %s.", output_path)
    return output_path


if __name__ == "__main__":
    build_standalone()
