"""Reliable negatives selected by low local signal evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import torch

from ...annotation_evidence import NEGATIVE, IonCatalogue, SignalEvidencePolicy
from ..base import SimulatedNegativeStrategy
from ..manager import SimulatedNegativeManager
from ....utils.logger import get_custom_logger


logger = get_custom_logger(__name__)


@SimulatedNegativeManager.register_strategy("SignalEvidenceSimulatedNegative")
class SignalEvidenceSimulatedNegative(SimulatedNegativeStrategy):
    """Mark unannotated low-evidence ion positions as reliable negatives.

    The strategy applies the notebook rule on spectra after the dataset's own
    binning and normalization. An entry is ``N_sim`` precisely when it is not
    annotated positive and its local maximum within ``bin_radius`` is less than
    or equal to ``relative_threshold`` times the spectrum maximum.

    :param relative_threshold: Relative local-evidence threshold in ``[0, 1]``.
    :type relative_threshold: float
    :param bin_radius: Number of neighbouring bins included on each side.
    :type bin_radius: int
    :param batch_size: Number of spectra evaluated per CPU evidence batch.
    :type batch_size: int
    :param cache_directory: Workspace-relative directory for compressed masks;
        ``None`` retains only the process-local dataset cache.
    :type cache_directory: str | None
    """

    def __init__(
        self,
        *,
        relative_threshold: float,
        bin_radius: int = 1,
        batch_size: int = 256,
        cache_directory: str | None = "cache/evidence/simulated_negatives",
    ) -> None:
        if not math.isfinite(relative_threshold) or not 0.0 <= relative_threshold <= 1.0:
            raise ValueError("relative_threshold must be finite and in [0, 1].")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if cache_directory is not None and (
            not isinstance(cache_directory, str) or not cache_directory.strip()
        ):
            raise ValueError("cache_directory must be a nonempty string or None.")
        self.relative_threshold = float(relative_threshold)
        self.bin_radius = bin_radius
        self.batch_size = batch_size
        self.cache_directory = cache_directory
        self.policy = SignalEvidencePolicy(
            relative_threshold=self.relative_threshold,
            bin_radius=self.bin_radius,
        )

    def build_mask(
        self,
        dataset: Any,
        target_field: str,
        source_indices: Sequence[int],
        targets: torch.Tensor,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return the evidence-selected ``N_sim`` mask.

        :param dataset: Dataset exposing ``get_evidence_spectra``.
        :type dataset: Any
        :param target_field: Annotation target field.
        :type target_field: str
        :param source_indices: Reader-level spectrum identifiers.
        :type source_indices: Sequence[int]
        :param targets: Canonical annotations with shape ``(N, C)``.
        :type targets: torch.Tensor
        :param availability: Candidate mask with shape ``(N, C)``.
        :type availability: torch.Tensor
        :return: Evidence-derived reliable-negative mask with shape ``(N, C)``.
        :rtype: torch.Tensor
        """
        if targets.ndim != 2 or availability.shape != targets.shape:
            raise ValueError("Evidence simulated negatives require targets and masks with shape (N, C).")
        if len(source_indices) != targets.shape[0]:
            raise ValueError("source_indices must align with the target rows.")
        spectrum_getter = getattr(dataset, "get_evidence_spectra", None)
        if not callable(spectrum_getter):
            raise TypeError("The dataset does not expose get_evidence_spectra().")
        catalogue = IonCatalogue.from_dataset(dataset, target_field)
        if len(catalogue.bins) != targets.shape[1]:
            raise ValueError("Evidence catalogue does not align with target columns.")

        cache_path = self._cache_path(dataset, target_field, source_indices, targets, catalogue)
        cached = self._load_cached_mask(cache_path, source_indices, targets.shape[1])
        if cached is not None:
            return cached  # (N, C)

        # Evidence materialization
        ## Compute states in bounded batches while preserving source ordering.
        masks: list[torch.Tensor] = []
        for start in range(0, len(source_indices), self.batch_size):
            stop = min(start + self.batch_size, len(source_indices))
            spectra = spectrum_getter(source_indices[start:stop])  # (B, M)
            states = self.policy.classify(
                spectra,
                targets[start:stop].to(dtype=spectra.dtype),
                availability[start:stop],
                catalogue,
            )  # (B, C)
            masks.append(states == NEGATIVE)  # (B, C)
        resolved = (
            torch.cat(masks, dim=0)
            if masks
            else torch.zeros_like(availability, dtype=torch.bool)
        )  # (N, C)
        self._store_cached_mask(cache_path, source_indices, resolved)
        return resolved

    def _cache_path(
        self,
        dataset: Any,
        target_field: str,
        source_indices: Sequence[int],
        targets: torch.Tensor,
        catalogue: IonCatalogue,
    ) -> Path | None:
        """Return the deterministic workspace cache path for one mask population."""
        if self.cache_directory is None:
            return None
        wrapper = getattr(getattr(dataset, "active_context", None), "_wrapper", None)
        project_path = getattr(wrapper, "_project_path", None)
        if not project_path:
            return None
        binner = getattr(getattr(dataset, "active_context", None), "binner", None)
        binner_config = (
            binner.get_config()
            if callable(getattr(binner, "get_config", None))
            else type(binner).__qualname__
        )
        payload = {
            "strategy": type(self).__name__,
            "target_field": target_field,
            "relative_threshold": self.relative_threshold,
            "bin_radius": self.bin_radius,
            "normalization": getattr(dataset, "normalization", None),
            "binner": binner_config,
            "class_names": catalogue.class_names,
            "bins": catalogue.bins,
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        digest.update(np.asarray(source_indices, dtype=np.int64).tobytes())
        digest.update(targets.detach().to(dtype=torch.uint8, device="cpu").numpy().tobytes())
        return Path(project_path) / self.cache_directory / f"{digest.hexdigest()}.npz"

    @staticmethod
    def _load_cached_mask(
        cache_path: Path | None,
        source_indices: Sequence[int],
        class_count: int,
    ) -> torch.Tensor | None:
        """Load a matching bit-packed mask, returning ``None`` on a cache miss."""
        if cache_path is None or not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                cached_indices = payload["source_indices"]
                if not np.array_equal(cached_indices, np.asarray(source_indices, dtype=np.int64)):
                    return None
                packed = payload["packed_mask"]
            unpacked = np.unpackbits(packed, axis=1, count=class_count).astype(np.bool_)
            if unpacked.shape != (len(source_indices), class_count):
                return None
            logger.info("Reused simulated-negative evidence cache: %s.", cache_path)
            return torch.from_numpy(unpacked)  # (N, C)
        except (OSError, KeyError, ValueError) as error:
            logger.warning("Ignoring unreadable simulated-negative cache %s: %s", cache_path, error)
            return None

    @staticmethod
    def _store_cached_mask(
        cache_path: Path | None,
        source_indices: Sequence[int],
        mask: torch.Tensor,
    ) -> None:
        """Atomically persist one compressed reliable-negative mask."""
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        packed = np.packbits(mask.detach().to(dtype=torch.uint8, device="cpu").numpy(), axis=1)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                prefix=f".{cache_path.stem}.",
                dir=cache_path.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                np.savez_compressed(
                    temporary,
                    source_indices=np.asarray(source_indices, dtype=np.int64),
                    packed_mask=packed,
                )
            os.replace(temporary_name, cache_path)
            logger.info("Stored simulated-negative evidence cache: %s.", cache_path)
        except OSError as error:
            logger.warning("Could not store simulated-negative cache %s: %s", cache_path, error)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)
