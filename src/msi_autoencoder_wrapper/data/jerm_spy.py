"""Campaign-global static nearest-spy precompute for JERM."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from .supervision_sampling import collect_supervision_masks
from ..utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


def prepare_jerm_static_spy_cache(
    dataset: Any,
    target_field: str,
    *,
    spectrum_batch_size: int = 256,
    positive_chunk_size: int = 64,
    unlabelled_chunk_size: int = 2048,
) -> tuple[torch.Tensor, ...]:
    """Create or load JERM's static nearest-spy source IDs.

    The nearest-spy relation depends only on the fixed train spectra and the
    fixed P/U/N_sim partition.  It is therefore computed once during campaign
    resolution rather than after every posterior epoch.

    :param dataset: Dataset exposing a train partition and model-input spectra.
    :type dataset: Any
    :param target_field: Molecular multi-label target field.
    :type target_field: str
    :param spectrum_batch_size: Source spectra loaded per I/O batch.
    :type spectrum_batch_size: int
    :param positive_chunk_size: Positive rows compared in one distance block.
    :type positive_chunk_size: int
    :param unlabelled_chunk_size: Unlabelled rows compared in one distance block.
    :type unlabelled_chunk_size: int
    :return: One tensor of stable source IDs for every target column.
    :rtype: tuple[torch.Tensor, ...]
    """
    source_ids, positives, unlabelled = _partition_masks(dataset, target_field)
    cache_path = _cache_path(dataset, target_field, source_ids, positives, unlabelled)
    cached = _load(cache_path, source_ids, positives.shape[1])
    if cached is not None:
        return cached

    spectrum_getter = getattr(dataset, "get_evidence_spectra", None)
    if not callable(spectrum_getter):
        raise TypeError("JERM static precompute requires dataset.get_evidence_spectra().")

    # Stable model-space input matrix
    ## The source IDs are sorted before loading, so cache rows and JERM epoch
    ## records use exactly the same lookup order.
    spectra = torch.cat(
        [
            spectrum_getter(source_ids[start:stop].tolist())
            for start in range(0, len(source_ids), spectrum_batch_size)
            for stop in (min(start + spectrum_batch_size, len(source_ids)),)
        ],
        dim=0,
    ).to(dtype=torch.float32, device=_compute_device(dataset))  # (N, M)
    spy_ids = _nearest_spy_source_ids(
        spectra,
        source_ids,
        positives,
        unlabelled,
        positive_chunk_size=positive_chunk_size,
        unlabelled_chunk_size=unlabelled_chunk_size,
    )
    _store(cache_path, source_ids, spy_ids)
    logger.info(
        "Prepared JERM static nearest-spy cache: spectra=%s classes=%s spies=%s.",
        len(source_ids),
        positives.shape[1],
        sum(int(values.numel()) for values in spy_ids),
    )
    return spy_ids


def load_jerm_static_spy_cache(
    dataset: Any,
    target_field: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Load precomputed JERM train masks and nearest-spy source IDs.

    :return: Sorted source IDs, P mask, U mask, and per-class spy source IDs.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]
    :raises FileNotFoundError: If campaign precompute has not been run.
    """
    source_ids, positives, unlabelled = _partition_masks(dataset, target_field)
    cache_path = _cache_path(dataset, target_field, source_ids, positives, unlabelled)
    cached = _load(cache_path, source_ids, positives.shape[1])
    if cached is None:
        raise FileNotFoundError(
            "JERM static nearest-spy cache is missing. Resolve the campaign before "
            "launching JERM workers."
        )
    return source_ids, positives, unlabelled, cached


def _partition_masks(
    dataset: Any,
    target_field: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted train IDs and fixed mutually exclusive P/U masks."""
    partition = dataset.create_partitions().train
    owner = partition.dataset if isinstance(partition, Subset) else partition
    indices = list(partition.indices) if isinstance(partition, Subset) else list(range(len(partition)))
    source_ids = torch.tensor(
        [int(owner.get_sample_id(index)) for index in indices], dtype=torch.long
    )  # (N,)
    targets, availability, simulated_negative = collect_supervision_masks(
        partition, target_field
    )
    positives = availability & (targets > 0.5)  # (N, C)
    unlabelled = availability & ~positives & ~simulated_negative  # (N, C)
    order = source_ids.argsort()
    return source_ids[order], positives[order], unlabelled[order]


def _nearest_spy_source_ids(
    spectra: torch.Tensor,
    source_ids: torch.Tensor,
    positives: torch.Tensor,
    unlabelled: torch.Tensor,
    *,
    positive_chunk_size: int,
    unlabelled_chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    """Find each positive row's nearest unlabelled row without dense storage."""
    if spectra.ndim != 2 or source_ids.shape != spectra.shape[:1]:
        raise ValueError("JERM spectra and source IDs must align as (N, M) and (N,).")
    if positives.shape != unlabelled.shape or positives.shape[:1] != spectra.shape[:1]:
        raise ValueError("JERM P/U masks must align with spectrum rows.")
    resolved: list[torch.Tensor] = []
    for column in range(positives.shape[1]):
        positive_rows = positives[:, column].nonzero(as_tuple=True)[0].to(spectra.device)
        unlabelled_rows = unlabelled[:, column].nonzero(as_tuple=True)[0].to(spectra.device)
        if positive_rows.numel() == 0 or unlabelled_rows.numel() == 0:
            resolved.append(torch.empty(0, dtype=torch.long))
            continue
        nearest_rows: list[torch.Tensor] = []
        for positive_start in range(0, positive_rows.numel(), positive_chunk_size):
            selected_positive = positive_rows[
                positive_start : positive_start + positive_chunk_size
            ]
            best_distance = torch.full(
                (selected_positive.numel(),),
                torch.inf,
                dtype=spectra.dtype,
                device=spectra.device,
            )
            best_row = torch.full(
                (selected_positive.numel(),), -1, dtype=torch.long, device=spectra.device
            )
            for unlabelled_start in range(0, unlabelled_rows.numel(), unlabelled_chunk_size):
                candidates = unlabelled_rows[
                    unlabelled_start : unlabelled_start + unlabelled_chunk_size
                ]
                distances = torch.cdist(
                    spectra[candidates], spectra[selected_positive]
                )  # (N_U_chunk, N_P_chunk)
                candidate_distance, candidate_position = distances.min(dim=0)
                update = candidate_distance < best_distance
                best_distance[update] = candidate_distance[update]
                best_row[update] = candidates[candidate_position[update]]
            nearest_rows.append(best_row)
        rows = torch.cat(nearest_rows).unique(sorted=True)
        resolved.append(source_ids[rows.to(device="cpu")].clone())
    return tuple(resolved)


def _cache_path(
    dataset: Any,
    target_field: str,
    source_ids: torch.Tensor,
    positives: torch.Tensor,
    unlabelled: torch.Tensor,
) -> Path:
    """Return the immutable cache location for one JERM train contract."""
    wrapper = getattr(getattr(dataset, "active_context", None), "_wrapper", None)
    workspace = getattr(wrapper, "workspace", None)
    project_path = getattr(workspace, "project_path_resolved", None)
    if not project_path:
        raise ValueError("JERM static precompute requires a workspace project path.")
    binner = getattr(getattr(dataset, "active_context", None), "binner", None)
    binner_config = binner.get_config() if callable(getattr(binner, "get_config", None)) else None
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "target_field": target_field,
                "normalization": getattr(dataset, "normalization", None),
                "binner": binner_config,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    for value in (source_ids, positives, unlabelled):
        digest.update(value.detach().to(device="cpu").numpy().tobytes())
    return Path(project_path) / "cache" / "evidence" / "jerm_static_spy" / f"{digest.hexdigest()}.npz"


def _compute_device(dataset: Any) -> torch.device:
    """Use the campaign GPU for static distance search when it is available."""
    wrapper = getattr(getattr(dataset, "active_context", None), "_wrapper", None)
    requested = torch.device(getattr(wrapper, "device", "cpu"))
    if requested.type == "cuda" and torch.cuda.is_available():
        return requested
    return torch.device("cpu")


def _load(
    path: Path,
    source_ids: torch.Tensor,
    class_count: int,
) -> tuple[torch.Tensor, ...] | None:
    """Load ragged per-class source-ID lists from an existing cache."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            cached_ids = payload["source_ids"]
            offsets = payload["offsets"]
            values = payload["spy_source_ids"]
        if not np.array_equal(cached_ids, source_ids.numpy()) or len(offsets) != class_count + 1:
            return None
        return tuple(
            torch.from_numpy(values[offsets[index] : offsets[index + 1]].copy())
            for index in range(class_count)
        )
    except (OSError, KeyError, ValueError) as error:
        logger.warning("Ignoring unreadable JERM static-spy cache %s: %s", path, error)
        return None


def _store(path: Path, source_ids: torch.Tensor, spy_ids: tuple[torch.Tensor, ...]) -> None:
    """Atomically store ragged per-class spy source IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([values.numel() for values in spy_ids], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    values = np.concatenate(
        [value.detach().cpu().numpy() for value in spy_ids],
        dtype=np.int64,
    ) if spy_ids else np.empty(0, dtype=np.int64)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{path.stem}.", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(
                temporary,
                source_ids=source_ids.detach().cpu().numpy(),
                offsets=offsets,
                spy_source_ids=values,
            )
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
