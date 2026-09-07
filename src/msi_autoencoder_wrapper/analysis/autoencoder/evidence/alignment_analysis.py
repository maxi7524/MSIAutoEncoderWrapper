"""Whether an annotated m/z actually lands on the bin that carries its peak.

``bin_radius`` exists because the mapping from an annotated m/z to a spectral bin is
not exact: the annotation carries a theoretical mass, the bin is 0.55 Da wide, and
the acquisition has its own calibration. If the peak systematically sits one bin
away from the mapped one, a radius of zero measures background at the mapped bin and
declares a negative for an ion that is plainly present one bin over. That is a
labelling error, not a modelling choice, and it is what this module measures.

Two independent pieces of evidence are combined:

*geometric* — where inside its bin the annotated m/z falls, from the annotation and
the binner alone, with no spectra involved;

*empirical* — where the strongest intensity inside the annotation window actually
sits, measured over annotated entries and compared against the same displacement
distribution over unannotated entries. Unannotated entries are the null: with no
real peak at the ion, the strongest bin in the window is placed by noise, so its
displacement distribution is roughly flat.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....utils.logger import get_custom_logger
from .precompute import EvidenceStatistics

logger = get_custom_logger(__name__)


def bin_offset_records(statistics: EvidenceStatistics, *, bin_step: float) -> list[dict[str, Any]]:
    """Geometric placement of every annotated m/z inside its mapped bin.

    Reports the signed distance from the annotated m/z to the centre of the bin it
    maps to, in Da and in bin widths. A distribution centred on zero and bounded by
    half a bin width means the mapping is doing what it claims; a systematic offset
    or values beyond half a bin width mean the mapping itself is misaligned, which no
    dilation radius can repair.

    :param statistics: A completed population pass.
    :param bin_step: Width of one spectral bin, in Da, from the binning configuration.
    :return: One record per ion with a resolvable annotated m/z.
    :rtype: list[dict[str, Any]]
    """
    records: list[dict[str, Any]] = []
    for position, name in enumerate(statistics.class_names):
        mz = float(statistics.class_mz[position])
        centre = float(statistics.class_bin_centre[position])
        if not np.isfinite(mz) or not np.isfinite(centre):
            continue
        offset = mz - centre
        records.append({
            "class_name": name,
            "mz": mz,
            "bin_centre": centre,
            "offset_da": offset,
            "offset_bins": offset / bin_step,
            "mapped_bins": int(statistics.class_bin_counts[position]),
        })
    logger.info("Resolved the mapped-bin geometry of %s ion(s).", len(records))
    return records


def peak_displacement_records(statistics: EvidenceStatistics) -> list[dict[str, Any]]:
    """Where the strongest bin of the annotation window sits, annotated versus not.

    For every displacement in the widest measured window, reports the share of
    annotated entries whose strongest bin sits there, and the same share over
    unannotated entries. The comparison, not the annotated share alone, carries the
    information: a peak genuinely at the mapped bin produces a sharp excess at
    displacement zero over the unannotated reference, while a systematic calibration
    offset produces that excess at a nonzero displacement.

    Entries whose window is entirely zero are excluded — their strongest bin is an
    arbitrary tie, and including them would flatten both distributions.

    :param statistics: A completed population pass.
    :return: One record per displacement.
    :rtype: list[dict[str, Any]]
    """
    annotated = statistics.offset_annotated.sum(axis=0).astype(np.float64)  # (D,)
    unannotated = statistics.offset_unannotated.astype(np.float64)  # (D,)
    annotated_total = float(annotated.sum())
    unannotated_total = float(unannotated.sum())

    records: list[dict[str, Any]] = []
    for position, offset in enumerate(statistics.offset_values):
        annotated_share = annotated[position] / annotated_total if annotated_total else float("nan")
        unannotated_share = unannotated[position] / unannotated_total if unannotated_total else float("nan")
        records.append({
            "offset_bins": int(offset),
            "annotated_entries": float(annotated[position]),
            "unannotated_entries": float(unannotated[position]),
            "annotated_share": annotated_share,
            "unannotated_share": unannotated_share,
            "excess_over_unannotated": annotated_share - unannotated_share,
        })
    return records


def class_displacement_records(statistics: EvidenceStatistics) -> list[dict[str, Any]]:
    """Per-ion displacement of the strongest bin, to find individually misaligned ions.

    An ion whose annotated entries put the strongest bin consistently one position
    away from the mapped bin is misaligned on its own, independently of the cohort.
    ``dominant_offset`` is the displacement holding the largest share of that ion's
    annotated entries, and ``centre_share`` the share sitting exactly on the mapped
    bin.

    :param statistics: A completed population pass.
    :return: One record per ion with at least one informative annotated entry.
    :rtype: list[dict[str, Any]]
    """
    offsets = np.asarray(statistics.offset_values)  # (D,)
    centre_position = int(np.flatnonzero(offsets == 0)[0])

    records: list[dict[str, Any]] = []
    for position, name in enumerate(statistics.class_names):
        counts = statistics.offset_annotated[position].astype(np.float64)  # (D,)
        total = float(counts.sum())
        if total <= 0:
            continue
        dominant = int(np.argmax(counts))
        records.append({
            "class_name": name,
            "mz": float(statistics.class_mz[position]),
            "informative_entries": total,
            "dominant_offset": int(offsets[dominant]),
            "dominant_share": float(counts[dominant] / total),
            "centre_share": float(counts[centre_position] / total),
        })
    misaligned = sum(1 for record in records if record["dominant_offset"] != 0)
    logger.info(
        "Measured peak displacement for %s ion(s); %s place their strongest bin off-centre.",
        len(records), misaligned,
    )
    return records


def bin_occupancy_records(statistics: EvidenceStatistics) -> list[dict[str, Any]]:
    """How many ion targets share each spectral bin.

    At a fixed bin width the mapping from ions to bins is not injective: isobaric
    ions, and ions whose exact masses differ by less than the bin width, land on the
    same coordinate. Their evidence is then *identical by construction* — the rule
    reads one number for all of them — so no threshold can assign them different
    states, and any per-ion evidence statistic is shared among them rather than
    measured independently. This is a property of the binning configuration, not of
    the evidence rule, and it bounds what the rule can possibly resolve.

    :param statistics: A completed population pass.
    :return: One record per occupied bin, with the ions mapped to it.
    :rtype: list[dict[str, Any]]
    """
    occupancy: dict[int, list[int]] = {}
    for position, centre in enumerate(statistics.class_bin_centre):
        occupancy.setdefault(int(round(float(centre) * 1e6)), []).append(position)

    records: list[dict[str, Any]] = []
    for key, positions in occupancy.items():
        records.append({
            "bin_centre": key / 1e6,
            "ions": len(positions),
            "class_names": "|".join(statistics.class_names[position] for position in positions),
        })
    shared = sum(record["ions"] for record in records if record["ions"] > 1)
    logger.info(
        "Bin occupancy: %s ion(s) share a bin with at least one other over %s occupied bin(s).",
        shared, len(records),
    )
    return sorted(records, key=lambda record: -record["ions"])
