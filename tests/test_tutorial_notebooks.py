"""Structural validation for the ordered user tutorial notebooks."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIRECTORY = REPOSITORY_ROOT / "assets" / "notebooks" / "tutorials"
EXPECTED_TUTORIALS = [
    "01_workspace_and_models.ipynb",
    "02_readers_binners_and_coordinates.ipynb",
    "03_model_configuration_and_training.ipynb",
    "04_autoencoder_and_latent_space.ipynb",
    "05_multi_image_models_todo.ipynb",
]
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def test_tutorials_are_clean_valid_notebooks_with_python_syntax() -> None:
    """Every tutorial is valid JSON with clean outputs and valid Python cells."""
    paths = sorted(TUTORIAL_DIRECTORY.glob("*.ipynb"))

    assert [path.name for path in paths] == EXPECTED_TUTORIALS
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path}:cell-{cell_index}",
            )


def test_local_tutorial_links_resolve_and_readme_lists_every_tutorial() -> None:
    """Local notebook links resolve and the main README links the full sequence."""
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    for tutorial_name in EXPECTED_TUTORIALS:
        assert f"assets/notebooks/tutorials/{tutorial_name}" in readme

    for path in TUTORIAL_DIRECTORY.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for target in MARKDOWN_LINK.findall(markdown):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target).resolve().exists(), (
                f"Broken local link in {path.name}: {target}"
            )
