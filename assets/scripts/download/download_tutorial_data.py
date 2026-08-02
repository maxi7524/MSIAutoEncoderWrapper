"""Prepare the datasets used by documentation and tutorial notebooks."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DATASETS = REPOSITORY_ROOT / "data" / "tutorial_workspace" / "datasets"
EXPECTED_DATASETS = ("example_1", "example_2")


def main() -> None:
    """Report the pending external tutorial-data source configuration."""
    missing = [
        name for name in EXPECTED_DATASETS if not (TUTORIAL_DATASETS / name).is_dir()
    ]
    if not missing:
        print(f"Tutorial datasets are already available in: {TUTORIAL_DATASETS}")
        return

    raise RuntimeError(
        "Tutorial datasets are missing: "
        f"{', '.join(missing)}. TODO: configure the maintainer-provided Google "
        "Drive source before enabling automatic downloads."
    )


if __name__ == "__main__":
    main()
