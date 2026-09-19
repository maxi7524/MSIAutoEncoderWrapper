"""Common-strategy adapter for campaign-local annotation-evidence caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...precompute.core.contracts import AnalysisPlugin, ArtifactSpec
from ..experiments import predictive_precompute
from .precompute import EvidencePrecompute, EvidenceStatistics, build_evidence_grid, build_population_dataset
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

# These are the threshold sweep values consumed by the maintained evidence
# notebooks. The configured campaign value is added separately below.
_DEFAULT_RELATIVE_THRESHOLDS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
_DEFAULT_BIN_RADII = (0, 1, 2)


def _configuration(settings: dict[str, Any]) -> tuple[Path, tuple[float, ...], tuple[int, ...]]:
    """Resolve one campaign's exact evidence-cache requirements.

    :param settings: Fully resolved heads-analysis settings.
    :type settings: dict[str, typing.Any]
    :return: Cache location, exact threshold boundaries, and measured bin radii.
    :rtype: tuple[pathlib.Path, tuple[float, ...], tuple[int, ...]]
    :raises ValueError: If the campaign enables evidence notebooks without a complete
        signal-evidence configuration.
    """
    configured = settings.get("evidence")
    if not isinstance(configured, dict):
        raise ValueError("annotation_evidence_cache requires an 'evidence' mapping.")
    try:
        relative_threshold = float(configured["relative_threshold"])
        bin_radius = int(configured["bin_radius"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "annotation_evidence_cache requires evidence.relative_threshold and evidence.bin_radius."
        ) from error
    if not 0.0 <= relative_threshold <= 1.0:
        raise ValueError("evidence.relative_threshold must be within [0, 1].")
    if bin_radius < 0:
        raise ValueError("evidence.bin_radius must be nonnegative.")

    candidates = tuple(sorted({*_DEFAULT_RELATIVE_THRESHOLDS, relative_threshold}))
    radii = tuple(sorted({*_DEFAULT_BIN_RADII, bin_radius}))
    return Path(settings["annotation_evidence_cache"]), candidates, radii


def _is_compatible(
    statistics: EvidenceStatistics,
    *,
    target_field: str,
    candidate_thresholds: tuple[float, ...],
    bin_radii: tuple[int, ...],
) -> bool:
    """Return whether a persisted evidence pass satisfies this campaign contract."""
    if statistics.metadata.get("target_field") != target_field:
        return False
    if not set(bin_radii).issubset(statistics.bin_radii):
        return False
    return all(np.isclose(statistics.grid.relative_edges, value).any() for value in candidate_thresholds)


def annotation_evidence_plugin() -> AnalysisPlugin:
    """Return the optional evidence-population stage for predictive-head campaigns.

    The stage is enabled only when a campaign declares ``annotation_evidence_cache``.
    It never borrows another campaign's archive: the configured path is the sole
    persisted artifact. A stale archive is rebuilt when it lacks the campaign's
    exact threshold boundary or required dilation radius.
    """

    def run(context) -> None:
        settings = context.settings
        cache_path, candidate_thresholds, bin_radii = _configuration(settings)
        target_field = str(settings.get("target_field", "molecule"))

        if cache_path.is_file():
            cached = EvidenceStatistics.load(cache_path)
            if _is_compatible(
                cached,
                target_field=target_field,
                candidate_thresholds=candidate_thresholds,
                bin_radii=bin_radii,
            ):
                logger.info("Reusing compatible annotation-evidence cache at %s.", cache_path)
                return
            logger.info("Rebuilding incompatible annotation-evidence cache at %s.", cache_path)

        _, dataset = build_population_dataset(settings["experiment_config"])
        device = predictive_precompute.resolve_device(settings, allow_cpu=context.allow_cpu)
        precompute = EvidencePrecompute(
            dataset,
            target_field=target_field,
            bin_radii=bin_radii,
            grid=build_evidence_grid(candidate_thresholds),
            candidate_relative_thresholds=candidate_thresholds,
            batch_size=min(int(settings["batch_size"]), 512),
            device=device,
        )
        precompute.run(progress=True).save(cache_path)

    return AnalysisPlugin(
        name="autoencoder.evidence.annotation_population",
        requires=("model_catalog",),
        provides=(
            ArtifactSpec(
                "annotation_evidence",
                root_setting="annotation_evidence_cache",
                path_kind="file",
            ),
        ),
        run=run,
        is_enabled_by=lambda context: bool(context.settings.get("annotation_evidence_cache")),
    )
