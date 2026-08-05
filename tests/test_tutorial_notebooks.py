"""Structural validation for the ordered user tutorial notebooks."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIRECTORY = REPOSITORY_ROOT / "docs" / "tutorials"
EXPECTED_TUTORIALS = [
    "autoencoder/autoencoder_01_model_configuration_and_training.ipynb",
    "autoencoder/autoencoder_02_inference_and_latent_space.ipynb",
    "cli-and-configuration/cli_configuration_01_validate_and_plan.ipynb",
    "cohort-models/cohort_models_01_multi_image_models.ipynb",
    "dataset-management/dataset_management_01_pride_explorer.ipynb",
    "dataset-management/dataset_management_02_metaspace_explorer.ipynb",
    "dataset-management/dataset_management_03_metaspace_download_and_merge.ipynb",
    "workspace-and-contexts/workspace_01_workspace_and_models.ipynb",
    "workspace-and-contexts/workspace_02_readers_binners_and_coordinates.ipynb",
]
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def test_tutorials_are_valid_notebooks_with_python_syntax() -> None:
    """Every tutorial is valid JSON and contains syntactically valid Python cells."""
    paths = sorted(TUTORIAL_DIRECTORY.rglob("*.ipynb"))

    assert [path.relative_to(TUTORIAL_DIRECTORY).as_posix() for path in paths] == (
        EXPECTED_TUTORIALS
    )
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if source.lstrip().startswith("%%"):
                continue
            ast.parse(
                source,
                filename=f"{path}:cell-{cell_index}",
            )


def test_local_tutorial_links_resolve_and_readme_lists_every_tutorial() -> None:
    """Local notebook links resolve and the main README links the full sequence."""
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    for tutorial_name in EXPECTED_TUTORIALS:
        assert f"docs/tutorials/{tutorial_name}" in readme

    for path in TUTORIAL_DIRECTORY.rglob("*.ipynb"):
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
