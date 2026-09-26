"""Reusable small-scale experiments for deconvolution validation notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import torch

from ..data.dictionary import GlobalCandidateDictionary
from ..data.synthetic import SyntheticDeconvolutionConfig, SyntheticDeconvolutionGenerator
from ..solvers.projected_gradient import NonnegativeProjectedGradientSolver


def catalogue_condition_counts(
    candidate_catalog: Any,
    conditions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Count candidate ions under explicit METASPACE-compatible conditions.

    :param candidate_catalog: Reader exposing ``get_candidate_ions(filters=...)``.
    :param conditions: Condition records containing a unique ``name`` and a
        ``filters`` mapping accepted by the candidate catalogue.
    :type candidate_catalog: Any
    :type conditions: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]]
    :return: One row per condition with ion, compound, formula, and adduct counts.
    :rtype: pandas.DataFrame
    :raises ValueError: If the records omit required condition fields.
    """
    get_ions = getattr(candidate_catalog, "get_candidate_ions", None)
    if not callable(get_ions):
        raise ValueError("candidate_catalog must expose get_candidate_ions().")
    catalogue_path = getattr(candidate_catalog, "path", None)
    if catalogue_path is not None and Path(catalogue_path).is_file():
        return _sqlite_catalogue_condition_counts(Path(catalogue_path), conditions)
    rows = []
    for condition in conditions:
        name = condition.get("name")
        filters = condition.get("filters")
        if not isinstance(name, str) or not name:
            raise ValueError("Every condition requires a non-empty name.")
        if not isinstance(filters, Mapping):
            raise ValueError("Every condition requires a filters mapping.")
        ions = get_ions(filters=dict(filters))
        rows.append(
            {
                "condition": name,
                "polarity": filters.get("polarity"),
                "mz_min": filters.get("mz_min"),
                "mz_max": filters.get("mz_max"),
                "adduct": filters.get("adduct"),
                "candidate_ion_count": len(ions),
                "compound_count": len({str(ion["compound_key"]) for ion in ions}),
                "formula_count": len({str(ion["formula"]) for ion in ions}),
                "adduct_count": len({str(ion["adduct"]) for ion in ions}),
            }
        )
    return pd.DataFrame(rows)


def _sqlite_catalogue_condition_counts(
    path: Path,
    conditions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Aggregate candidate counts in SQLite without materializing ion records.

    :param path: Materialized candidate-catalogue SQLite path.
    :param conditions: Explicit METASPACE-compatible filter conditions.
    :type path: pathlib.Path
    :type conditions: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]]
    :return: One aggregate record per condition.
    :rtype: pandas.DataFrame
    """
    rows = []
    with sqlite3.connect(path) as connection:
        for condition in conditions:
            name = condition.get("name")
            filters = condition.get("filters")
            if not isinstance(name, str) or not name or not isinstance(filters, Mapping):
                raise ValueError("Every condition requires a non-empty name and filters mapping.")
            clauses: list[str] = []
            values: list[object] = []
            for key, column in {
                "provider": "compound.provider",
                "adduct": "ion.adduct",
                "polarity": "ion.polarity",
                "formula": "ion.formula",
            }.items():
                if filters.get(key) is not None:
                    clauses.append(f"{column} = ?")
                    values.append(str(filters[key]))
            if filters.get("mz_min") is not None:
                clauses.append("ion.theoretical_mz >= ?")
                values.append(float(filters["mz_min"]))
            if filters.get("mz_max") is not None:
                clauses.append("ion.theoretical_mz <= ?")
                values.append(float(filters["mz_max"]))
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            query = f"""
                SELECT COUNT(*) AS candidate_ion_count,
                       COUNT(DISTINCT ion.compound_key) AS compound_count,
                       COUNT(DISTINCT ion.formula) AS formula_count,
                       COUNT(DISTINCT ion.adduct) AS adduct_count
                FROM candidate_ions AS ion
                JOIN candidate_compounds AS compound USING (compound_key)
                {where}
            """
            counts = connection.execute(query, values).fetchone()
            rows.append(
                {
                    "condition": name,
                    "polarity": filters.get("polarity"),
                    "mz_min": filters.get("mz_min"),
                    "mz_max": filters.get("mz_max"),
                    "adduct": filters.get("adduct"),
                    "candidate_ion_count": int(counts[0]),
                    "compound_count": int(counts[1]),
                    "formula_count": int(counts[2]),
                    "adduct_count": int(counts[3]),
                }
            )
    return pd.DataFrame(rows)


def projected_gradient_convergence(
    dictionary: GlobalCandidateDictionary,
    *,
    batch_size: int,
    seed: int,
    iteration_counts: Sequence[int],
    synthetic_config: SyntheticDeconvolutionConfig,
    l1_weight: float = 0.0,
) -> pd.DataFrame:
    """Measure per-spectrum solver convergence on one deterministic batch.

    :param dictionary: Global candidate dictionary.
    :param batch_size: Number of fixed synthetic spectra.
    :param seed: Generator seed defining the analyzed batch.
    :param iteration_counts: Positive projected-gradient iteration counts.
    :param synthetic_config: Mixture-generation configuration.
    :param l1_weight: Sparse penalty used by every solver run.
    :type dictionary: GlobalCandidateDictionary
    :type batch_size: int
    :type seed: int
    :type iteration_counts: collections.abc.Sequence[int]
    :type synthetic_config: SyntheticDeconvolutionConfig
    :type l1_weight: float
    :return: Long-form per-spectrum convergence observations.
    :rtype: pandas.DataFrame
    """
    if not iteration_counts or any(count < 1 for count in iteration_counts):
        raise ValueError("iteration_counts must contain positive integers.")
    batch = SyntheticDeconvolutionGenerator(
        dictionary,
        synthetic_config,
        seed=seed,
    ).generate(batch_size)
    if batch.abundances is None or batch.presence is None:
        raise RuntimeError("Synthetic deconvolution generation must provide exact targets.")
    rows = []
    for iterations in sorted(set(int(count) for count in iteration_counts)):
        result = NonnegativeProjectedGradientSolver(
            iterations=iterations,
            l1_weight=l1_weight,
        )(batch.spectra, dictionary)
        abundance_error = (result.abundances - batch.abundances).abs().mean(dim=1)  # (B,)
        reconstruction_mse = result.residual.square().mean(dim=1)  # (B,)
        support_exact = (result.abundances > 1e-8).eq(batch.presence).all(dim=1)  # (B,)
        for sample_index in range(batch.batch_size):
            rows.append(
                {
                    "iterations": iterations,
                    "sample_index": sample_index,
                    "objective": float(result.objective[sample_index]),
                    "abundance_mae": float(abundance_error[sample_index]),
                    "reconstruction_mse": float(reconstruction_mse[sample_index]),
                    "support_exact": bool(support_exact[sample_index]),
                }
            )
    return pd.DataFrame(rows)


def sample_global_subdictionary(
    dictionary: GlobalCandidateDictionary,
    *,
    candidate_count: int,
    seed: int,
) -> GlobalCandidateDictionary:
    """Select a deterministic candidate subset from a global dictionary.

    :param dictionary: Full dataset-wide candidate dictionary.
    :param candidate_count: Number of globally indexed candidate columns to retain.
    :param seed: Local Torch random seed defining the selected global indices.
    :type dictionary: GlobalCandidateDictionary
    :type candidate_count: int
    :type seed: int
    :return: Candidate subdictionary with matrix shape ``(M, candidate_count)``.
    :rtype: GlobalCandidateDictionary
    :raises ValueError: If ``candidate_count`` is outside the global dictionary range.
    """
    if candidate_count < 1 or candidate_count > dictionary.candidate_count:
        raise ValueError("candidate_count must be within the global dictionary size.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(
        dictionary.candidate_count,
        generator=generator,
        device="cpu",
    )[:candidate_count]  # (C_subset,)
    return dictionary.select_columns(indices)


def projected_gradient_gradcheck(
    dictionary: GlobalCandidateDictionary,
    *,
    step_size: float = 0.1,
    iterations: int = 3,
) -> bool:
    """Verify autograd through projected-gradient updates by finite differences.

    :param dictionary: Candidate dictionary whose first active columns define a
        small differentiable test case.
    :param step_size: Fixed positive update step away from ReLU kinks.
    :param iterations: Number of differentiable projected-gradient updates.
    :type dictionary: GlobalCandidateDictionary
    :type step_size: float
    :type iterations: int
    :return: Whether PyTorch's finite-difference gradient check succeeds.
    :rtype: bool
    :raises ValueError: If the dictionary has fewer than one candidate.
    """
    if dictionary.candidate_count < 1:
        raise ValueError("gradcheck requires at least one candidate.")
    matrix = dictionary.matrix.detach().to(dtype=torch.float64, device="cpu")  # (M, C)
    abundance = torch.ones((1, matrix.shape[1]), dtype=torch.float64)  # (B=1, C)
    spectra = (abundance @ matrix.transpose(0, 1)).requires_grad_(True)  # (B=1, M)
    solver = NonnegativeProjectedGradientSolver(
        iterations=iterations,
        step_size=step_size,
    )

    def reconstruction(values: torch.Tensor) -> torch.Tensor:
        return solver(values, matrix).reconstruction  # (B=1, M)

    return bool(torch.autograd.gradcheck(reconstruction, (spectra,), fast_mode=True))


def identifiability_experiment(
    dictionary: GlobalCandidateDictionary,
    *,
    candidate_counts: Sequence[int],
    repeats: int,
    batch_size: int,
    seed: int,
    synthetic_config: SyntheticDeconvolutionConfig,
    solver_iterations: int,
) -> pd.DataFrame:
    """Measure recovery and dictionary degeneracy for random global subsets.

    :param dictionary: Full dataset-wide candidate dictionary.
    :param candidate_counts: Global-subset sizes such as ``(10, 50, 100)``.
    :param repeats: Independent deterministic candidate subset draws per size.
    :param batch_size: Number of synthetic spectra per subset draw.
    :param seed: Base local random seed.
    :param synthetic_config: Sparse mixture-generation settings.
    :param solver_iterations: Projected-gradient iteration count.
    :type dictionary: GlobalCandidateDictionary
    :type candidate_counts: collections.abc.Sequence[int]
    :type repeats: int
    :type batch_size: int
    :type seed: int
    :type synthetic_config: SyntheticDeconvolutionConfig
    :type solver_iterations: int
    :return: One aggregate recovery record per subset draw.
    :rtype: pandas.DataFrame
    """
    if repeats < 1 or batch_size < 1 or solver_iterations < 1:
        raise ValueError("repeats, batch_size, and solver_iterations must be positive.")
    if not candidate_counts or any(count < 1 or count > dictionary.candidate_count for count in candidate_counts):
        raise ValueError("candidate_counts must be valid non-empty global subset sizes.")
    rows = []
    for candidate_count in candidate_counts:
        for repeat in range(repeats):
            subdictionary = sample_global_subdictionary(
                dictionary,
                candidate_count=candidate_count,
                seed=seed + 10_000 * candidate_count + repeat,
            )
            generator = SyntheticDeconvolutionGenerator(
                subdictionary,
                synthetic_config,
                seed=seed + repeat + candidate_count,
            )
            batch = generator.generate(batch_size)
            if batch.abundances is None or batch.presence is None:
                raise RuntimeError("Synthetic deconvolution generation must provide exact targets.")
            result = NonnegativeProjectedGradientSolver(iterations=solver_iterations)(
                batch.spectra,
                subdictionary,
            )
            normalized = subdictionary.matrix / subdictionary.matrix.norm(dim=0).clamp_min(1e-12)  # (M, C_subset)
            coherence = normalized.transpose(0, 1) @ normalized  # (C_subset, C_subset)
            off_diagonal = coherence - torch.eye(
                candidate_count,
                dtype=coherence.dtype,
                device=coherence.device,
            )  # (C_subset, C_subset)
            rows.append(
                {
                    "candidate_count": candidate_count,
                    "repeat": repeat,
                    "dictionary_rank": int(torch.linalg.matrix_rank(subdictionary.matrix)),
                    "max_pairwise_coherence": float(off_diagonal.abs().max()),
                    "abundance_mae": float((result.abundances - batch.abundances).abs().mean()),
                    "reconstruction_mse": float(result.residual.square().mean()),
                    "support_exact_fraction": float(
                        (result.abundances > 1e-8).eq(batch.presence).all(dim=1).float().mean()
                    ),
                }
            )
    return pd.DataFrame(rows)
