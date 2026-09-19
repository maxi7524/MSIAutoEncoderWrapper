"""Canonical storage of precompute provenance, model selections, and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import AnalysisContext
    from .contracts import ArtifactSpec, PrecomputeStrategy


class ArtifactStore:
    """Own precompute-control artifacts while preserving notebook result layout."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def prepare(self, context: "AnalysisContext", strategy: "PrecomputeStrategy") -> None:
        """Create the control directory and every configured notebook result folder."""
        self.root.mkdir(parents=True, exist_ok=True)
        for plugin in strategy.stages:
            if not plugin.is_enabled(context):
                continue
            for spec in plugin.provides:
                if spec.analysis_name is None:
                    continue
                configured = context.settings["analyses"][spec.analysis_name]
                Path(configured["output_directory"]).mkdir(parents=True, exist_ok=True)

    def write_plan(self, strategy: "PrecomputeStrategy", stage_names: list[str]) -> None:
        """Persist the exact, validated stage order selected for this run."""
        payload = {"strategy": strategy.name, "model_type": strategy.model_type, "stages": stage_names}
        (self.root / "execution_plan.json").write_text(json.dumps(payload, indent=2) + "\n")

    def verify(self, context: "AnalysisContext", spec: "ArtifactSpec") -> None:
        """Check that a plugin produced the files its contract promises.

        :raises FileNotFoundError: If a declared output is absent after a plugin run.
        """
        if spec.analysis_name is not None:
            directory = Path(context.settings["analyses"][spec.analysis_name]["output_directory"])
        elif spec.root_setting is not None:
            directory = Path(context.settings[spec.root_setting])
        else:
            return
        if spec.path_kind == "file":
            if not directory.is_file():
                raise FileNotFoundError(
                    f"Plugin output '{spec.name}' is incomplete: expected '{directory}'."
                )
            return
        for name in spec.required_files:
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Plugin output '{spec.name}' is incomplete: expected '{path}'."
                )
