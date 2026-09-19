"""Configured precomputation for the initial deconvolution feasibility study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from msi_dataset_manager.annotations import CandidateCatalogReader

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.models.architectures.types.deconvolution import (
    GlobalCandidateDictionary,
    SyntheticDeconvolutionConfig,
    catalogue_condition_counts,
    identifiability_experiment,
    projected_gradient_convergence,
    projected_gradient_gradcheck,
)
from msi_autoencoder_wrapper.models.architectures.types.deconvolution.evaluation.catalogue_precompute import (
    catalogue_path,
    precompute_catalogue,
)
from ...core.contracts import AnalysisPlugin, ArtifactSpec, PrecomputeStrategy


def read_settings(path: Path | str) -> dict[str, Any]:
    """Load initial deconvolution settings and resolve the repository root.

    :param path: YAML settings file stored beside the notebook collection.
    :type path: pathlib.Path | str
    :return: Parsed settings with absolute ``settings_path`` and repository root.
    :rtype: dict[str, typing.Any]
    :raises FileNotFoundError: If ``path`` does not exist.
    :raises ValueError: If the settings are not located in this repository.
    """
    settings_path = Path(path).resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"No analysis settings at '{settings_path}'.")
    root = next((parent for parent in settings_path.parents if (parent / "pyproject.toml").is_file()), None)
    if root is None:
        raise ValueError("Analysis settings must live inside the repository.")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    if not isinstance(settings, dict):
        raise ValueError("Deconvolution analysis settings must be a YAML mapping.")
    settings["settings_path"] = str(settings_path)
    settings["repository_root"] = str(root)
    return settings


def _inventory(_: dict[str, Any]) -> pd.DataFrame:
    """Expose the deterministic baseline as one model-family inventory record."""
    return pd.DataFrame(
        [
            {
                "model_id": "nonnegative_projected_gradient",
                "source": "torch_synthetic",
                "grid_id": "initial_feasibility",
                "task_id": "projected_gradient",
                "repetition": 0,
                "label": "Projected-gradient baseline",
                "ready": True,
            }
        ]
    )


def _analysis_settings(context: Any, name: str) -> Mapping[str, Any]:
    """Return one required named analysis configuration.

    :param context: Common precompute context.
    :param name: Analysis YAML key.
    :type context: typing.Any
    :type name: str
    :return: Configured analysis mapping.
    :rtype: collections.abc.Mapping[str, typing.Any]
    :raises ValueError: If the named analysis is not configured.
    """
    value = context.settings.get("analyses", {}).get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Initial deconvolution analysis '{name}' must be configured.")
    return value


def _condition(context: Any, analysis_name: str) -> Mapping[str, Any]:
    """Resolve the explicit candidate-universe condition selected by one analysis."""
    condition_name = _analysis_settings(context, analysis_name).get("condition")
    for candidate in context.settings.get("metaspace_conditions", []):
        if isinstance(candidate, Mapping) and candidate.get("name") == condition_name:
            filters = candidate.get("filters")
            if isinstance(filters, Mapping):
                return filters
    raise ValueError(
        f"Analysis '{analysis_name}' references unknown METASPACE condition {condition_name!r}."
    )


def _candidate_catalog(context: Any) -> CandidateCatalogReader:
    """Materialize and memoize the dataset-wide candidate-ion catalogue."""
    def build_reader() -> CandidateCatalogReader:
        path = precompute_catalogue(
            context.settings,
            settings_path=Path(context.settings["settings_path"]),
        )
        return CandidateCatalogReader(path)

    return context.resources.get_or_create("deconvolution_candidate_catalog", build_reader)


def _binner(context: Any) -> Any:
    """Build and memoize the explicitly configured spectral binner."""
    def build_binner() -> Any:
        definition = context.settings.get("binner")
        if not isinstance(definition, Mapping):
            raise ValueError("Deconvolution analysis requires a binner mapping.")
        strategy = definition.get("strategy")
        parameters = definition.get("parameters")
        if not isinstance(strategy, str) or not isinstance(parameters, Mapping):
            raise ValueError("binner requires string strategy and parameters mapping.")
        BinnerManager.discover_strategies()
        return BinnerManager.get_binner(strategy, **dict(parameters))

    return context.resources.get_or_create("deconvolution_binner", build_binner)


def _synthetic_config(context: Any) -> SyntheticDeconvolutionConfig:
    """Create the deterministic sparse-mixture configuration from YAML."""
    definition = context.settings.get("synthetic")
    if not isinstance(definition, Mapping):
        raise ValueError("Deconvolution analysis requires a synthetic mapping.")
    return SyntheticDeconvolutionConfig(**dict(definition))


def _selected_dictionary(
    context: Any,
    *,
    analysis_name: str,
    candidate_count: int,
    seed: int,
) -> GlobalCandidateDictionary:
    """Render only a deterministic local subset of the global candidate universe.

    The candidate catalogue remains global.  The dense ``(M, C)`` Torch matrix is
    intentionally constructed only after a seeded subset selection, because the full
    catalogue can contain hundreds of thousands of ions and is not a valid dense
    baseline allocation.
    """
    return GlobalCandidateDictionary.from_candidate_catalog(
        _candidate_catalog(context),
        _binner(context),
        filters=dict(_condition(context, analysis_name)),
        candidate_limit=candidate_count,
        selection_seed=seed,
    )


def _write_metadata(directory: Path, payload: Mapping[str, Any]) -> None:
    """Write stable, human-readable provenance for one notebook result directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_selected_candidates(directory: Path, dictionary: GlobalCandidateDictionary) -> None:
    """Persist exact selected catalogue records in their Torch-column order."""
    records = [
        {
            "column_index": index,
            "candidate_ion_key": record.get("candidate_ion_key"),
            "compound_key": record.get("compound_key"),
            "formula": record.get("formula"),
            "adduct": record.get("adduct"),
            "theoretical_mz": record.get("theoretical_mz"),
            "record_json": json.dumps(dict(record), sort_keys=True, default=str),
        }
        for index, record in enumerate(dictionary.candidate_ions)
    ]
    pd.DataFrame(records).to_csv(directory / "selected_candidates.csv", index=False)


def _candidate_coverage(context: Any) -> None:
    """Count global candidate populations under every configured condition."""
    output = Path(_analysis_settings(context, "candidate_coverage")["output_directory"])
    catalogue = _candidate_catalog(context)
    table = catalogue_condition_counts(catalogue, context.settings["metaspace_conditions"])
    table.to_csv(output / "candidate_coverage.csv", index=False)
    _write_metadata(
        output,
        {
            "method": "candidate_catalogue_condition_counts",
            "candidate_catalogue": str(catalogue_path(context.settings, notebook_dir=Path(context.settings["settings_path"]).parent)),
            "conditions": context.settings["metaspace_conditions"],
        },
    )


def _projected_gradient_baseline(context: Any) -> None:
    """Validate gradients and convergence of the Torch baseline on a fixed subset."""
    configuration = context.settings.get("baseline")
    if not isinstance(configuration, Mapping):
        raise ValueError("Deconvolution analysis requires a baseline method mapping.")
    output = Path(_analysis_settings(context, "baseline")["output_directory"])
    seed = int(configuration["seed"])
    dictionary = _selected_dictionary(
        context,
        analysis_name="baseline",
        candidate_count=int(configuration["candidate_count"]),
        seed=seed,
    )
    convergence = projected_gradient_convergence(
        dictionary,
        batch_size=int(configuration["batch_size"]),
        seed=seed,
        iteration_counts=tuple(int(value) for value in configuration["iteration_counts"]),
        synthetic_config=_synthetic_config(context),
        l1_weight=float(configuration.get("l1_weight", 0.0)),
    )
    convergence.to_csv(output / "convergence.csv", index=False)
    pd.DataFrame(
        [{"gradcheck_passed": projected_gradient_gradcheck(dictionary)}]
    ).to_csv(output / "gradient_validation.csv", index=False)
    _write_selected_candidates(output, dictionary)
    _write_metadata(
        output,
        {
            "method": "torch_nonnegative_projected_gradient",
            "condition": dict(_condition(context, "baseline")),
            "candidate_count": dictionary.candidate_count,
            "seed": seed,
            "synthetic": dict(context.settings["synthetic"]),
        },
    )


def _identifiability(context: Any) -> None:
    """Measure small-dictionary recovery from a reproducible global candidate pool."""
    configuration = context.settings.get("identifiability")
    if not isinstance(configuration, Mapping):
        raise ValueError("Deconvolution analysis requires an identifiability method mapping.")
    output = Path(_analysis_settings(context, "identifiability")["output_directory"])
    counts = tuple(int(value) for value in configuration["candidate_counts"])
    seed = int(configuration["seed"])
    dictionary = _selected_dictionary(
        context,
        analysis_name="identifiability",
        candidate_count=max(counts),
        seed=seed,
    )
    table = identifiability_experiment(
        dictionary,
        candidate_counts=counts,
        repeats=int(configuration["repeats"]),
        batch_size=int(configuration["batch_size"]),
        seed=seed,
        synthetic_config=_synthetic_config(context),
        solver_iterations=int(configuration["solver_iterations"]),
    )
    table.to_csv(output / "identifiability.csv", index=False)
    _write_selected_candidates(output, dictionary)
    _write_metadata(
        output,
        {
            "method": "torch_projected_gradient_identifiability",
            "condition": dict(_condition(context, "identifiability")),
            "candidate_pool_count": dictionary.candidate_count,
            "candidate_counts": counts,
            "seed": seed,
            "synthetic": dict(context.settings["synthetic"]),
        },
    )


def build_strategy() -> PrecomputeStrategy:
    """Build the explicit three-stage initial deconvolution workflow."""
    return PrecomputeStrategy(
        name="deconvolution.initial_analysis",
        model_type="deconvolution",
        stages=(
            AnalysisPlugin(
                name="deconvolution.candidate_coverage",
                requires=("model_catalog",),
                provides=(
                    ArtifactSpec("candidate_catalogue", root_setting="repository_root"),
                    ArtifactSpec(
                        "candidate_coverage",
                        "candidate_coverage",
                        ("candidate_coverage.csv", "metadata.json"),
                    ),
                ),
                run=_candidate_coverage,
            ),
            AnalysisPlugin(
                name="deconvolution.projected_gradient_baseline",
                requires=("model_catalog", "candidate_catalogue"),
                provides=(
                    ArtifactSpec(
                        "projected_gradient_baseline",
                        "baseline",
                        ("convergence.csv", "gradient_validation.csv", "selected_candidates.csv", "metadata.json"),
                    ),
                ),
                run=_projected_gradient_baseline,
            ),
            AnalysisPlugin(
                name="deconvolution.identifiability",
                requires=("model_catalog", "candidate_catalogue"),
                provides=(
                    ArtifactSpec(
                        "identifiability",
                        "identifiability",
                        ("identifiability.csv", "selected_candidates.csv", "metadata.json"),
                    ),
                ),
                run=_identifiability,
            ),
        ),
        load_settings=read_settings,
        inventory=_inventory,
    )
