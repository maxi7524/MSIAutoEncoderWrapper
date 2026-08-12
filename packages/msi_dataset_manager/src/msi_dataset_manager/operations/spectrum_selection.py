"""Select annotated spectra and optional reproducible unannotated controls."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable, List, Optional, Sequence

from ..utils.exceptions import raise_validation_error


def select_merge_spectrum_ids(
    *,
    candidate_ids: Sequence[int],
    annotated_ids: Iterable[int],
    unannotated_ratio: Optional[float] = None,
    unannotated_amount: Optional[int] = None,
    random_seed: int = 0,
    seed_namespace: str = "",
) -> List[int]:
    """Select annotated IDs and an optional sample without annotations.

    :param candidate_ids: Valid source spectrum IDs eligible for selection.
    :type candidate_ids: Sequence[int]
    :param annotated_ids: Spectrum IDs linked to at least one molecule.
    :type annotated_ids: Iterable[int]
    :param unannotated_ratio: Requested unannotated count relative to the
        annotated count. For example, ``3.0`` requests three unannotated
        spectra per annotated spectrum.
    :type unannotated_ratio: float | None
    :param unannotated_amount: Requested absolute unannotated count.
    :type unannotated_amount: int | None
    :param random_seed: Reproducible base seed.
    :type random_seed: int
    :param seed_namespace: Dataset-specific value preventing identical samples
        for different datasets that share a base seed.
    :type seed_namespace: str
    :return: Sorted annotated IDs followed by sorted sampled unannotated IDs.
    :rtype: List[int]
    :raises ValidationError: If a limit is negative or candidates repeat.

    When both limits are omitted, all available unannotated spectra are kept.
    An explicit amount or ratio of zero keeps none. When both limits are
    supplied, the larger requested count is used and capped by availability.
    """
    candidates = [int(value) for value in candidate_ids]
    if len(candidates) != len(set(candidates)):
        raise_validation_error("MergeSelection", "candidate spectrum IDs must be unique.")
    if unannotated_ratio is not None and float(unannotated_ratio) < 0:
        raise_validation_error("MergeSelection", "unannotated_ratio cannot be negative.")
    if unannotated_amount is not None and int(unannotated_amount) < 0:
        raise_validation_error("MergeSelection", "unannotated_amount cannot be negative.")

    annotated_set = {int(value) for value in annotated_ids}
    selected_annotated = [value for value in candidates if value in annotated_set]
    available_unannotated = [value for value in candidates if value not in annotated_set]
    ratio_count = (
        math.floor(len(selected_annotated) * float(unannotated_ratio))
        if unannotated_ratio is not None
        else 0
    )
    amount_count = int(unannotated_amount) if unannotated_amount is not None else 0
    # Unannotated-spectrum semantics
    ## No limit means all available controls; an explicit zero means none.
    requested_count = (
        len(available_unannotated)
        if unannotated_ratio is None and unannotated_amount is None
        else min(max(ratio_count, amount_count), len(available_unannotated))
    )
    if requested_count == 0:
        return selected_annotated

    seed_material = f"{int(random_seed)}:{seed_namespace}".encode("utf-8")
    resolved_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    sampled = random.Random(resolved_seed).sample(
        available_unannotated,
        requested_count,
    )

    # TODO: Add a configurable background-detection strategy based on a
    # dedicated segmentation model. Unannotated spectra must not be assumed
    # to represent background.
    return selected_annotated + sorted(sampled)
