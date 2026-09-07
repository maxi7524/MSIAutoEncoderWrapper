"""One streaming pass producing every statistic needed to select evidence thresholds.

The quantity the whole package is built around is the *relative evidence*

.. math::

    r_{ic} \\;=\\; \\frac{\\max_{b \\in \\mathcal{B}_c^{(\\rho)}} x_{ib}}{\\max_b x_{ib}},

where :math:`x_i` is the TIC-normalized binned spectrum of pixel :math:`i`,
:math:`\\mathcal{B}_c` the bins the ion :math:`c` maps to and :math:`\\rho` the
``bin_radius`` dilation. With ``absolute_threshold`` :math:`\\alpha = 0` the rule in
:class:`~msi_autoencoder_wrapper.data.annotation_evidence.SignalEvidencePolicy`
reduces exactly to a threshold on that single scalar: an unannotated entry becomes
an operational negative iff :math:`r_{ic} \\le \\beta`. Every threshold sweep in this
package is therefore a cumulative sum over a histogram of :math:`r`, not a repeated
pass over the image.

Because the sweep is exact only at histogram boundaries, the grid returned by
:func:`build_evidence_grid` carries the candidate thresholds as explicit edges;
:func:`cumulative_below` refuses any threshold that is not one of them.

REMARK: The full annotated kidney population is ~4.2e5 pixels x ~5e2 ions, so the
dense evidence matrix would be ~8e8 entries. Nothing here materializes it: the pass
accumulates histograms and per-pixel reductions only, which keeps the persisted
artifact in the low megabytes and makes a re-run of the analysis cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ....data.annotation_evidence import IonCatalogue
from ....data.collators import RawSpectrumCollator
from ....data.datasets import RawDatasetView
from ....data.preprocessing import BatchPreprocessor
from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

# Evidence-state codes, re-exported from the training-time implementation so the
# analysis can never drift from the semantics the criterions actually use.
from ....data.annotation_evidence import NEGATIVE, POSITIVE, UNLABELLED  # noqa: E402


def build_population_dataset(
    config_path: str | Path,
    *,
    use_configured_subset: bool = False,
    split_seed: int | None = None,
) -> tuple[Any, Any]:
    """Build the campaign's dataset without constructing or training any model.

    The dataset is resolved through the campaign's own planning entry point, so the
    reader, binner, annotation mapping, normalization and target policy are exactly
    the ones the configured runs will use. No model weights are created.

    :param config_path: Experiment YAML holding ``task.parameters.factory_parameters``.
    :type config_path: str | pathlib.Path
    :param use_configured_subset: Keep the campaign's ``dataset.parameters.subset``
        block. ``False`` removes it, giving the complete annotated pixel population;
        the evidence rule is a property of the data, not of a training subsample.
    :type use_configured_subset: bool
    :param split_seed: Split seed forwarded to the planning pipeline. Defaults to the
        campaign's ``seeds.common_seeds.split``. It only affects partitioning, which
        this package never requests.
    :type split_seed: int | None
    :return: The planning wrapper and its dataset.
    :rtype: tuple[Any, Any]
    :raises ValidationError: If the configuration carries no factory parameters.
    """
    # Campaign configuration
    ## The analysis reads the same file the campaign runner reads, never a copy of it.
    config = yaml.safe_load(Path(config_path).read_text())
    parameters = config.get("task", {}).get("parameters", {})
    factory_parameters = parameters.get("factory_parameters")
    if not isinstance(factory_parameters, Mapping):
        raise_validation_error(
            "EvidencePrecompute",
            f"{config_path} has no task.parameters.factory_parameters block.",
        )
    factory_parameters = dict(factory_parameters)
    dataset_parameters = dict(factory_parameters["dataset"]["parameters"])
    if not use_configured_subset:
        dataset_parameters.pop("subset", None)
    factory_parameters["dataset"] = dict(factory_parameters["dataset"])
    factory_parameters["dataset"]["parameters"] = dataset_parameters

    seed = int(
        split_seed
        if split_seed is not None
        else config.get("seeds", {}).get("common_seeds", {}).get("split", 42)
    )

    # REMARK: `_build_planning_pipeline` is the campaign's own resolver. Reusing it is
    # deliberate: any independent reconstruction of reader/binner/annotation wiring
    # would risk analysing a population the training runs never see.
    from ....runtime.workflows.configured import _build_planning_pipeline

    wrapper, dataset = _build_planning_pipeline(factory_parameters, split_seed=seed)
    logger.info(
        "Resolved the evidence population: %s spectra, configured_subset=%s, split_seed=%s.",
        len(dataset), use_configured_subset, seed,
    )
    return wrapper, dataset


@dataclass(frozen=True)
class EvidenceGrid:
    """Histogram boundaries on which every threshold sweep is exact.

    Bucket ``k`` of a histogram built on ``edges`` holds the values in
    ``(edges[k - 1], edges[k]]``, bucket ``0`` holds the values equal to ``0`` and the
    final bucket ``len(edges)`` holds anything above ``edges[-1]``. A cumulative sum
    up to and including bucket ``k`` is therefore exactly the count of entries at or
    below ``edges[k]``, which is what the evidence rule tests.

    :param relative_edges: Ascending boundaries for the relative evidence ``r``,
        starting at ``0`` and containing every candidate relative threshold.
    :param absolute_edges: Ascending boundaries for the absolute evidence ``s``,
        i.e. the TIC-normalized intensity itself.
    :param maximum_edges: Ascending boundaries for the per-pixel spectrum maximum,
        used only for the joint ``(r, max)`` histogram behind the absolute-threshold
        interaction.
    """

    relative_edges: np.ndarray
    absolute_edges: np.ndarray
    maximum_edges: np.ndarray

    @property
    def relative_buckets(self) -> int:
        """Number of histogram buckets on the relative axis."""
        return int(self.relative_edges.size) + 1

    @property
    def absolute_buckets(self) -> int:
        """Number of histogram buckets on the absolute axis."""
        return int(self.absolute_edges.size) + 1

    @property
    def maximum_buckets(self) -> int:
        """Number of histogram buckets on the spectrum-maximum axis."""
        return int(self.maximum_edges.size) + 1


def build_evidence_grid(
    candidate_relative_thresholds: Sequence[float] = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
    candidate_absolute_thresholds: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3),
    *,
    relative_decades: tuple[float, float] = (-6.0, 0.0),
    absolute_decades: tuple[float, float] = (-8.0, 0.0),
    maximum_decades: tuple[float, float] = (-4.0, 0.0),
    points_per_decade: int = 40,
) -> EvidenceGrid:
    """Build logarithmic histogram boundaries containing the candidate thresholds.

    :param candidate_relative_thresholds: Relative thresholds that must be exactly
        representable, i.e. the values a campaign configuration may set.
    :param candidate_absolute_thresholds: Absolute thresholds that must be exactly
        representable, on the TIC-normalized intensity scale.
    :param relative_decades: Base-10 exponent range covered on the relative axis.
    :param absolute_decades: Base-10 exponent range covered on the absolute axis.
    :param maximum_decades: Base-10 exponent range covered on the spectrum-maximum axis.
    :param points_per_decade: Resolution of the logarithmic part of each axis.
    :return: Grid whose edges start at zero and end at one.
    :rtype: EvidenceGrid
    """

    def _axis(decades: tuple[float, float], required: Sequence[float]) -> np.ndarray:
        low, high = decades
        count = int(round((high - low) * points_per_decade)) + 1
        logarithmic = np.logspace(low, high, count, dtype=np.float64)
        required_values = np.asarray([value for value in required if value > 0.0], dtype=np.float64)
        edges = np.concatenate(([0.0], logarithmic, required_values, [1.0]))
        return np.unique(np.round(edges, 12))

    return EvidenceGrid(
        relative_edges=_axis(relative_decades, candidate_relative_thresholds),
        absolute_edges=_axis(absolute_decades, candidate_absolute_thresholds),
        maximum_edges=_axis(maximum_decades, ()),
    )


def evidence_signals(
    spectra: torch.Tensor,
    bin_index: torch.Tensor,
    bin_valid: torch.Tensor,
    bin_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the local peak evidence and its ratio to the spectrum maximum.

    Reproduces the measurement inside
    :meth:`~msi_autoencoder_wrapper.data.annotation_evidence.SignalEvidencePolicy.classify`
    exactly: the spectrum is max-pooled over a window of ``2 * bin_radius + 1`` bins
    and each ion takes the largest pooled value over the bins it maps to. Splitting
    it out keeps the equivalence directly testable against the training-time rule.

    :param spectra: Nonnegative model inputs, ``(B, M)``.
    :param bin_index: Padded ion-to-bin map, ``(C, K)``.
    :param bin_valid: Validity of each padded position, ``(C, K)``.
    :param bin_radius: Dilation radius applied before the per-ion maximum.
    :return: Absolute evidence ``(B, C)`` and its ratio to the spectrum maximum
        ``(B, C)``; the ratio is zero wherever the spectrum is identically zero.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    pooled = F.max_pool1d(
        spectra.unsqueeze(1), 2 * bin_radius + 1, stride=1, padding=bin_radius
    ).squeeze(1)  # (B, M)
    gathered = pooled[:, bin_index.reshape(-1)].reshape(
        spectra.shape[0], bin_index.shape[0], bin_index.shape[1]
    )  # (B, C, K)
    signal = (
        gathered.masked_fill(~bin_valid.unsqueeze(0), float("-inf")).amax(dim=2).clamp_min(0.0)
    )  # (B, C)
    maximum = spectra.amax(dim=1)  # (B,)
    safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))  # (B,)
    return signal, signal / safe_maximum.unsqueeze(1)


def cumulative_below(counts: np.ndarray, edges: np.ndarray, threshold: float) -> np.ndarray:
    """Count entries at or below one exact histogram boundary.

    :param counts: Histogram counts whose last axis runs over buckets.
    :param edges: The boundaries the histogram was built on.
    :param threshold: A value that must appear in ``edges`` (``0`` is always valid).
    :return: Counts reduced over the bucket axis.
    :rtype: numpy.ndarray
    :raises ValidationError: If the threshold is not a boundary, which would make the
        answer an interpolation rather than a count.
    """
    if threshold == 0.0:
        return counts[..., 0]
    position = int(np.searchsorted(edges, threshold))
    if position >= edges.size or not np.isclose(edges[position], threshold, rtol=1e-9, atol=0.0):
        raise_validation_error(
            "EvidenceStatistics",
            f"Threshold {threshold} is not a histogram boundary; rebuild the grid with it.",
        )
    return counts[..., : position + 1].sum(axis=-1)


@dataclass
class EvidenceStatistics:
    """Everything one population pass measures about the evidence rule.

    Histograms carry a trailing bucket axis interpreted through
    :class:`EvidenceGrid`. ``annotated`` means the ion is annotated in that pixel
    (state ``P`` regardless of signal); ``unannotated`` means it is not, i.e. exactly
    the entries the evidence rule splits into ``N`` and ``U``.

    :param class_names: Ion identities in target-column order.
    :param class_bin_counts: Number of spectral bins each ion maps to, ``(C,)``.
    :param class_mz: Annotated m/z of each ion, ``(C,)``.
    :param class_bin_centre: Centre of the ion's first mapped bin, ``(C,)``.
    :param bin_radii: Dilation radii the pass measured, ``(R,)``.
    :param annotated_relative: Relative-evidence histogram of annotated entries,
        ``(R, C, K_r)``.
    :param unannotated_relative: The same for unannotated entries, ``(R, C, K_r)``.
    :param annotated_absolute: Absolute-evidence histogram of annotated entries,
        ``(R, C, K_a)``.
    :param unannotated_absolute: The same for unannotated entries, ``(R, C, K_a)``.
    :param background_relative: Histogram of every bin's intensity relative to its
        own spectrum maximum, ``(K_r,)`` — the reference a per-ion distribution must
        be read against.
    :param background_absolute: Histogram of every bin's TIC-normalized intensity,
        ``(K_a,)``.
    :param group_relative: Relative-evidence histogram per source dataset and state,
        ``(G, R, 2, K_r)`` with state order ``(unannotated, annotated)``.
    :param joint_relative_maximum: Joint histogram of relative evidence and the
        pixel's spectrum maximum over unannotated entries, ``(R, K_r, K_m)``.
    :param offset_values: Bin displacements the alignment histogram covers, ``(D,)``.
    :param offset_annotated: Per ion, how often the strongest bin inside the widest
        measured window sits at each displacement from the mapped bin, counted over
        annotated entries with nonzero intensity in that window, ``(C, D)``.
    :param offset_unannotated: The same pooled over ions, for unannotated entries,
        ``(D,)`` — the reference an annotated displacement must beat.
    :param group_names: Source-dataset identifiers, ``(G,)``.
    :param spectrum_ids: Reader-level pixel identifiers, ``(N,)``.
    :param spectrum_group_index: Index into ``group_names`` per pixel, ``(N,)``.
    :param spectrum_maximum: Maximum TIC-normalized intensity per pixel, ``(N,)``.
    :param spectrum_nonzero_bins: Number of nonzero bins per pixel, ``(N,)``.
    :param spectrum_annotated_count: Number of annotated ions per pixel, ``(N,)``.
    :param spectrum_available_count: Number of ions with an available label, ``(N,)``.
    :param spectrum_negative_count: Operational negatives per pixel at each candidate
        threshold, ``(N, R, T)``.
    :param candidate_relative_thresholds: The thresholds behind that last array, ``(T,)``.
    :param grid: The histogram boundaries.
    :param metadata: Provenance of the pass (configuration, population size, ...).
    """

    class_names: tuple[str, ...]
    class_bin_counts: np.ndarray
    class_mz: np.ndarray
    class_bin_centre: np.ndarray
    bin_radii: tuple[int, ...]
    annotated_relative: np.ndarray
    unannotated_relative: np.ndarray
    annotated_absolute: np.ndarray
    unannotated_absolute: np.ndarray
    background_relative: np.ndarray
    background_absolute: np.ndarray
    group_relative: np.ndarray
    joint_relative_maximum: np.ndarray
    offset_values: np.ndarray
    offset_annotated: np.ndarray
    offset_unannotated: np.ndarray
    group_names: tuple[str, ...]
    spectrum_ids: np.ndarray
    spectrum_group_index: np.ndarray
    spectrum_maximum: np.ndarray
    spectrum_nonzero_bins: np.ndarray
    spectrum_annotated_count: np.ndarray
    spectrum_available_count: np.ndarray
    spectrum_negative_count: np.ndarray
    candidate_relative_thresholds: np.ndarray
    grid: EvidenceGrid
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def class_count(self) -> int:
        """Number of ion target columns."""
        return len(self.class_names)

    @property
    def spectrum_count(self) -> int:
        """Number of pixels in the measured population."""
        return int(self.spectrum_ids.size)

    def radius_index(self, bin_radius: int) -> int:
        """Return the axis position of one measured dilation radius.

        :raises ValidationError: If the radius was not part of the pass.
        """
        if int(bin_radius) not in self.bin_radii:
            raise_validation_error(
                "EvidenceStatistics",
                f"bin_radius={bin_radius} was not measured; available: {self.bin_radii}.",
            )
        return self.bin_radii.index(int(bin_radius))

    def save(self, path: str | Path) -> Path:
        """Persist the pass as one compressed archive.

        :param path: Destination ``.npz`` path; parent directories are created.
        :return: The written path.
        :rtype: pathlib.Path
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            class_names=np.asarray(self.class_names, dtype=object),
            class_bin_counts=self.class_bin_counts,
            class_mz=self.class_mz,
            class_bin_centre=self.class_bin_centre,
            bin_radii=np.asarray(self.bin_radii, dtype=np.int64),
            annotated_relative=self.annotated_relative,
            unannotated_relative=self.unannotated_relative,
            annotated_absolute=self.annotated_absolute,
            unannotated_absolute=self.unannotated_absolute,
            background_relative=self.background_relative,
            background_absolute=self.background_absolute,
            group_relative=self.group_relative,
            joint_relative_maximum=self.joint_relative_maximum,
            offset_values=self.offset_values,
            offset_annotated=self.offset_annotated,
            offset_unannotated=self.offset_unannotated,
            group_names=np.asarray(self.group_names, dtype=object),
            spectrum_ids=self.spectrum_ids,
            spectrum_group_index=self.spectrum_group_index,
            spectrum_maximum=self.spectrum_maximum,
            spectrum_nonzero_bins=self.spectrum_nonzero_bins,
            spectrum_annotated_count=self.spectrum_annotated_count,
            spectrum_available_count=self.spectrum_available_count,
            spectrum_negative_count=self.spectrum_negative_count,
            candidate_relative_thresholds=self.candidate_relative_thresholds,
            relative_edges=self.grid.relative_edges,
            absolute_edges=self.grid.absolute_edges,
            maximum_edges=self.grid.maximum_edges,
            metadata=np.asarray(yaml.safe_dump(self.metadata), dtype=object),
        )
        logger.info("Stored the evidence statistics at %s.", destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceStatistics":
        """Restore a persisted pass.

        :param path: Archive written by :meth:`save`.
        :rtype: EvidenceStatistics
        """
        with np.load(Path(path), allow_pickle=True) as archive:
            return cls(
                class_names=tuple(str(name) for name in archive["class_names"]),
                class_bin_counts=archive["class_bin_counts"],
                class_mz=archive["class_mz"],
                class_bin_centre=archive["class_bin_centre"],
                bin_radii=tuple(int(value) for value in archive["bin_radii"]),
                annotated_relative=archive["annotated_relative"],
                unannotated_relative=archive["unannotated_relative"],
                annotated_absolute=archive["annotated_absolute"],
                unannotated_absolute=archive["unannotated_absolute"],
                background_relative=archive["background_relative"],
                background_absolute=archive["background_absolute"],
                group_relative=archive["group_relative"],
                joint_relative_maximum=archive["joint_relative_maximum"],
                offset_values=archive["offset_values"],
                offset_annotated=archive["offset_annotated"],
                offset_unannotated=archive["offset_unannotated"],
                group_names=tuple(str(name) for name in archive["group_names"]),
                spectrum_ids=archive["spectrum_ids"],
                spectrum_group_index=archive["spectrum_group_index"],
                spectrum_maximum=archive["spectrum_maximum"],
                spectrum_nonzero_bins=archive["spectrum_nonzero_bins"],
                spectrum_annotated_count=archive["spectrum_annotated_count"],
                spectrum_available_count=archive["spectrum_available_count"],
                spectrum_negative_count=archive["spectrum_negative_count"],
                candidate_relative_thresholds=archive["candidate_relative_thresholds"],
                grid=EvidenceGrid(
                    relative_edges=archive["relative_edges"],
                    absolute_edges=archive["absolute_edges"],
                    maximum_edges=archive["maximum_edges"],
                ),
                metadata=yaml.safe_load(str(archive["metadata"])) or {},
            )


class EvidencePrecompute:
    """Measure the evidence population of one dataset in a single streaming pass.

    The pass reproduces the training input pipeline exactly — the campaign's binner
    and normalization applied through
    :class:`~msi_autoencoder_wrapper.data.preprocessing.BatchPreprocessor`, the same
    object the trainer feeds to the evidence criterions — and then accumulates the
    histograms described in :class:`EvidenceStatistics` instead of a loss.

    :param dataset: Dataset built by :func:`build_population_dataset`.
    :param target_field: Ion target field carrying the annotations.
    :param bin_radii: Dilation radii to measure jointly in one pass.
    :param grid: Histogram boundaries; defaults to :func:`build_evidence_grid`.
    :param candidate_relative_thresholds: Thresholds for which per-pixel negative
        counts are retained. Must all be boundaries of ``grid.relative_edges``.
    :param batch_size: Pixels per read. Only affects runtime, not results.
    :param num_workers: DataLoader workers for spectrum reading.
    :param device: Device the histogram reductions run on.
    :param group_fields: Metadata field identifying the source acquisition of a pixel.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        target_field: str = "molecule",
        bin_radii: Sequence[int] = (0, 1, 2),
        grid: EvidenceGrid | None = None,
        candidate_relative_thresholds: Sequence[float] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05),
        batch_size: int = 512,
        num_workers: int = 0,
        device: str | torch.device = "cpu",
        group_fields: str = "dataset_id",
    ) -> None:
        self.dataset = dataset
        self.target_field = str(target_field)
        self.bin_radii = tuple(int(radius) for radius in bin_radii)
        if not self.bin_radii or any(radius < 0 for radius in self.bin_radii):
            raise_validation_error("EvidencePrecompute", "bin_radii must be nonnegative and nonempty.")
        self.grid = grid if grid is not None else build_evidence_grid()
        self.candidate_relative_thresholds = np.asarray(candidate_relative_thresholds, dtype=np.float64)
        for threshold in self.candidate_relative_thresholds:
            cumulative_below(np.zeros(self.grid.relative_buckets), self.grid.relative_edges, float(threshold))
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.device = torch.device(device)
        self.group_fields = str(group_fields)

        # Ion catalogue
        ## Built by the training-time implementation so ion-to-bin mapping cannot diverge.
        self.catalogue = IonCatalogue.from_dataset(dataset, self.target_field)
        self._bin_index, self._bin_valid = self._padded_bin_index(self.catalogue)
        logger.info(
            "Evidence catalogue: %s ions over %s bins, radii=%s.",
            len(self.catalogue.class_names), self.catalogue.feature_count, self.bin_radii,
        )

    @staticmethod
    def _padded_bin_index(catalogue: IonCatalogue) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack the ragged ion-to-bin map into a rectangular gather index.

        :return: Index matrix ``(C, K)`` and its validity mask ``(C, K)``.
        """
        width = max(len(bins) for bins in catalogue.bins)
        index = np.zeros((len(catalogue.bins), width), dtype=np.int64)  # (C, K)
        valid = np.zeros((len(catalogue.bins), width), dtype=bool)  # (C, K)
        for row, bins in enumerate(catalogue.bins):
            index[row, : len(bins)] = np.asarray(bins, dtype=np.int64)
            valid[row, : len(bins)] = True
        return torch.from_numpy(index), torch.from_numpy(valid)

    def _iter_batches(self) -> Iterator[Any]:
        """Yield preprocessed batches through the trainer's own raw-read path."""
        loader = DataLoader(
            RawDatasetView(self.dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=RawSpectrumCollator(self.dataset.get_target_schemas()),
            persistent_workers=False,
        )
        preprocessor = BatchPreprocessor(self.dataset, self.device, self.device)
        for raw_batch in loader:
            yield preprocessor(raw_batch)

    def _resolve_group_lookup(self) -> tuple[tuple[str, ...], np.ndarray]:
        """Map every reader spectrum identifier to its source acquisition.

        The lookup is built once over the reader's complete identifier space rather
        than over dataset positions, so batches can be resolved from the sample
        identifiers they already carry, independently of any subset or split.

        :return: Ordered group names and a lookup array indexed by spectrum id.
        :rtype: tuple[tuple[str, ...], numpy.ndarray]
        """
        reader = getattr(self.dataset.active_context, "annotation_reader", None)
        bulk_getter = getattr(reader, "get_spectrum_groups", None)
        total = int(self.dataset.active_context.get_data_reader("image").GetNumberOfSpectra())
        if callable(bulk_getter):
            raw_groups = list(bulk_getter(list(range(total)), group_fields=self.group_fields))
        else:
            logger.warning("Annotation reader exposes no bulk group API; every pixel is pooled.")
            raw_groups = [("__all_samples__",)] * total
        labels = ["|".join(str(part) for part in np.atleast_1d(group)) for group in raw_groups]
        names = tuple(sorted(set(labels)))
        positions = {name: position for position, name in enumerate(names)}
        logger.info("Resolved %s source acquisition group(s) over %s reader spectra.", len(names), total)
        return names, np.asarray([positions[label] for label in labels], dtype=np.int64)

    def _class_mz(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the annotated m/z and first mapped bin centre of every ion.

        :return: ``(C,)`` annotated m/z and ``(C,)`` bin centre, ``nan`` where the
            reader exposes no raw m/z for an identity.
        """
        index = self.dataset.get_mapped_annotation_index()
        axis = np.asarray(index.coordinate_axis, dtype=np.float64)
        centre = np.asarray(
            [axis[int(bins[0])] if len(bins) else np.nan for bins in self.catalogue.bins],
            dtype=np.float64,
        )  # (C,)

        reader = getattr(self.dataset.active_context, "annotation_reader", None)
        raw_getter = getattr(reader, "get_spectrum_annotation_index", None)
        mz_by_name: dict[str, float] = {}
        if callable(raw_getter):
            raw_index = raw_getter(None)
            identities = np.asarray(
                ["|".join(identity) for identity in raw_index.annotation_identities], dtype=object
            )
            names = identities[np.asarray(raw_index.annotation_indices, dtype=np.int64)]
            values = np.asarray(raw_index.mz_values, dtype=np.float64)
            for name, value in zip(names, values):
                mz_by_name.setdefault(str(name), float(value))
        mz = np.asarray(
            [mz_by_name.get(name, np.nan) for name in self.catalogue.class_names], dtype=np.float64
        )  # (C,)
        return mz, centre

    def run(self, *, progress: bool = True) -> EvidenceStatistics:
        """Stream the population and accumulate every evidence statistic.

        :param progress: Show a per-batch progress bar. The full kidney population is
            hundreds of thousands of imzML reads, so the bar is on by default.
        :return: The measured statistics.
        :rtype: EvidenceStatistics
        :raises ValidationError: If a batch carries no targets for the ion field.
        """
        # Static tensors
        ## Shared by every batch; moved to the reduction device once.
        radius_count = len(self.bin_radii)
        class_count = len(self.catalogue.class_names)
        bin_index = self._bin_index.to(self.device)  # (C, K)
        bin_valid = self._bin_valid.to(self.device)  # (C, K)
        relative_edges = torch.as_tensor(self.grid.relative_edges, dtype=torch.float32, device=self.device)
        absolute_edges = torch.as_tensor(self.grid.absolute_edges, dtype=torch.float32, device=self.device)
        maximum_edges = torch.as_tensor(self.grid.maximum_edges, dtype=torch.float32, device=self.device)
        candidates = torch.as_tensor(
            self.candidate_relative_thresholds, dtype=torch.float32, device=self.device
        )  # (T,)
        relative_buckets = self.grid.relative_buckets
        absolute_buckets = self.grid.absolute_buckets
        maximum_buckets = self.grid.maximum_buckets

        group_names, group_lookup = self._resolve_group_lookup()
        group_lookup_tensor = torch.from_numpy(group_lookup).to(self.device)  # (S,)

        # Accumulators
        ## Histograms stay on the reduction device; per-pixel reductions collect on CPU.
        annotated_relative = torch.zeros((radius_count, class_count, relative_buckets), dtype=torch.int64, device=self.device)
        unannotated_relative = torch.zeros_like(annotated_relative)
        annotated_absolute = torch.zeros((radius_count, class_count, absolute_buckets), dtype=torch.int64, device=self.device)
        unannotated_absolute = torch.zeros_like(annotated_absolute)
        background_relative = torch.zeros(relative_buckets, dtype=torch.int64, device=self.device)
        background_absolute = torch.zeros(absolute_buckets, dtype=torch.int64, device=self.device)
        group_relative = torch.zeros((len(group_names), radius_count, 2, relative_buckets), dtype=torch.int64, device=self.device)
        joint_relative_maximum = torch.zeros((radius_count, relative_buckets, maximum_buckets), dtype=torch.int64, device=self.device)

        # Annotation-to-bin alignment
        ## Measured once, in the widest window any configured radius could dilate to.
        ## REMARK: The window is centred on the ion's first mapped bin. Every ion in
        ## this cohort maps to exactly one bin, so that is the mapped bin itself; for
        ## a multi-bin ion the displacement is reported relative to its first bin.
        offset_radius = max(self.bin_radii)
        offset_values = np.arange(-offset_radius, offset_radius + 1, dtype=np.int64)  # (D,)
        offset_width = int(offset_values.size)
        ## REMARK: The window is clamped to the spectral axis, so an ion within
        ## `offset_radius` bins of either end repeats its edge bin instead of reading
        ## outside the axis. Those ions can only bias their own displacement toward the
        ## edge; `class_displacement_records` reports per ion, so they stay identifiable.
        window_index = (
            bin_index[:, :1] + torch.as_tensor(offset_values, device=self.device).reshape(1, -1)
        ).clamp(0, self.catalogue.feature_count - 1)  # (C, D)
        offset_annotated = torch.zeros((class_count, offset_width), dtype=torch.int64, device=self.device)
        offset_unannotated = torch.zeros(offset_width, dtype=torch.int64, device=self.device)

        spectrum_ids: list[np.ndarray] = []
        spectrum_groups: list[np.ndarray] = []
        spectrum_maximum: list[np.ndarray] = []
        spectrum_nonzero: list[np.ndarray] = []
        spectrum_annotated: list[np.ndarray] = []
        spectrum_available: list[np.ndarray] = []
        spectrum_negative: list[np.ndarray] = []

        # Population pass
        ## One read per pixel; every radius is measured from the same read.
        processed = 0
        batches = self._iter_batches()
        total_batches = int(np.ceil(len(self.dataset) / self.batch_size))
        if progress:
            batches = tqdm(batches, total=total_batches, desc="evidence population", unit="batch")
        for batch in batches:
            spectra = batch[1].to(self.device)  # (B, M)
            targets = batch[2].get(self.target_field)
            masks = batch[3].get(self.target_field)
            if targets is None or masks is None:
                raise_validation_error(
                    "EvidencePrecompute", f"Batch carries no '{self.target_field}' targets."
                )
            targets = targets.to(self.device)  # (B, C)
            mask = masks.to(self.device)  # (B, C)
            if mask.ndim == 1:
                mask = mask.unsqueeze(1).expand_as(targets)  # (B, C)
            annotated = (targets > 0.5) & mask.bool()  # (B, C)
            unannotated = (~(targets > 0.5)) & mask.bool()  # (B, C)
            batch_size = int(spectra.shape[0])
            sample_ids = batch[0].to(self.device).reshape(-1).long()  # (B,)
            batch_groups = group_lookup_tensor[sample_ids]  # (B,)

            ### Spectrum scale: the denominator of every relative quantity below.
            maximum = spectra.amax(dim=1)  # (B,)
            safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))  # (B,)

            ### Background reference over every bin of every pixel.
            background_relative += torch.bincount(
                torch.searchsorted(relative_edges, (spectra / safe_maximum.unsqueeze(1)).reshape(-1).contiguous()),
                minlength=relative_buckets,
            )
            background_absolute += torch.bincount(
                torch.searchsorted(absolute_edges, spectra.reshape(-1).contiguous()),
                minlength=absolute_buckets,
            )

            ### Displacement of the strongest bin inside the annotation window.
            window = spectra[:, window_index.reshape(-1)].reshape(
                batch_size, class_count, offset_width
            )  # (B, C, D)
            informative = window.amax(dim=2) > 0  # (B, C)
            displacement = window.argmax(dim=2)  # (B, C)
            offset_annotated += torch.bincount(
                (
                    torch.arange(class_count, device=self.device).unsqueeze(0).expand_as(displacement)[annotated & informative]
                    * offset_width
                    + displacement[annotated & informative]
                ),
                minlength=class_count * offset_width,
            ).reshape(class_count, offset_width)
            offset_unannotated += torch.bincount(
                displacement[unannotated & informative], minlength=offset_width
            )

            maximum_bucket = torch.searchsorted(maximum_edges, maximum.contiguous())  # (B,)
            negative_counts = torch.zeros(
                (batch_size, radius_count, candidates.numel()), dtype=torch.int32, device=self.device
            )  # (B, R, T)

            for radius_position, radius in enumerate(self.bin_radii):
                #### Local peak evidence, identical to SignalEvidencePolicy.classify.
                signal, relative = evidence_signals(spectra, bin_index, bin_valid, radius)  # (B, C)

                #### Bucket assignment shared by every histogram of this radius.
                relative_bucket = torch.searchsorted(relative_edges, relative.contiguous())  # (B, C)
                absolute_bucket = torch.searchsorted(absolute_edges, signal.contiguous())  # (B, C)
                class_axis = torch.arange(class_count, device=self.device).unsqueeze(0).expand_as(relative_bucket)

                for state_mask, relative_target, absolute_target in (
                    (annotated, annotated_relative, annotated_absolute),
                    (unannotated, unannotated_relative, unannotated_absolute),
                ):
                    if not bool(state_mask.any()):
                        continue
                    relative_target[radius_position] += torch.bincount(
                        (class_axis[state_mask] * relative_buckets + relative_bucket[state_mask]),
                        minlength=class_count * relative_buckets,
                    ).reshape(class_count, relative_buckets)
                    absolute_target[radius_position] += torch.bincount(
                        (class_axis[state_mask] * absolute_buckets + absolute_bucket[state_mask]),
                        minlength=class_count * absolute_buckets,
                    ).reshape(class_count, absolute_buckets)

                #### Per-acquisition histograms keep the threshold sweep stratifiable.
                group_axis = batch_groups.unsqueeze(1).expand_as(relative_bucket)  # (B, C)
                for state_position, state_mask in ((0, unannotated), (1, annotated)):
                    if not bool(state_mask.any()):
                        continue
                    group_relative[:, radius_position, state_position] += torch.bincount(
                        (group_axis[state_mask] * relative_buckets + relative_bucket[state_mask]),
                        minlength=len(group_names) * relative_buckets,
                    ).reshape(len(group_names), relative_buckets)

                #### Joint (r, spectrum maximum) histogram behind the absolute threshold.
                maximum_axis = maximum_bucket.unsqueeze(1).expand_as(relative_bucket)  # (B, C)
                joint_relative_maximum[radius_position] += torch.bincount(
                    (relative_bucket[unannotated] * maximum_buckets + maximum_axis[unannotated]),
                    minlength=relative_buckets * maximum_buckets,
                ).reshape(relative_buckets, maximum_buckets)

                #### Per-pixel operational-negative counts at the candidate thresholds.
                below = (relative.unsqueeze(2) <= candidates.reshape(1, 1, -1)) & unannotated.unsqueeze(2)  # (B, C, T)
                negative_counts[:, radius_position] = below.sum(dim=1, dtype=torch.int32)  # (B, T)

            spectrum_ids.append(sample_ids.cpu().numpy().astype(np.int64))
            spectrum_groups.append(batch_groups.cpu().numpy().astype(np.int32))
            spectrum_maximum.append(maximum.detach().cpu().numpy().astype(np.float32))
            spectrum_nonzero.append((spectra > 0).sum(dim=1).cpu().numpy().astype(np.int32))
            spectrum_annotated.append(annotated.sum(dim=1).cpu().numpy().astype(np.int32))
            spectrum_available.append(mask.bool().sum(dim=1).cpu().numpy().astype(np.int32))
            spectrum_negative.append(negative_counts.cpu().numpy())
            processed += batch_size

        if processed != len(self.dataset):
            raise_validation_error(
                "EvidencePrecompute",
                f"Read {processed} of {len(self.dataset)} pixels; the pass is incomplete.",
            )
        logger.info("Completed the evidence pass over %s pixels.", processed)

        class_mz, class_bin_centre = self._class_mz()
        return EvidenceStatistics(
            class_names=tuple(self.catalogue.class_names),
            class_bin_counts=np.asarray([len(bins) for bins in self.catalogue.bins], dtype=np.int32),
            class_mz=class_mz,
            class_bin_centre=class_bin_centre,
            bin_radii=self.bin_radii,
            annotated_relative=annotated_relative.cpu().numpy(),
            unannotated_relative=unannotated_relative.cpu().numpy(),
            annotated_absolute=annotated_absolute.cpu().numpy(),
            unannotated_absolute=unannotated_absolute.cpu().numpy(),
            background_relative=background_relative.cpu().numpy(),
            background_absolute=background_absolute.cpu().numpy(),
            group_relative=group_relative.cpu().numpy(),
            joint_relative_maximum=joint_relative_maximum.cpu().numpy(),
            offset_values=offset_values,
            offset_annotated=offset_annotated.cpu().numpy(),
            offset_unannotated=offset_unannotated.cpu().numpy(),
            group_names=group_names,
            spectrum_ids=np.concatenate(spectrum_ids),
            spectrum_group_index=np.concatenate(spectrum_groups),
            spectrum_maximum=np.concatenate(spectrum_maximum),
            spectrum_nonzero_bins=np.concatenate(spectrum_nonzero),
            spectrum_annotated_count=np.concatenate(spectrum_annotated),
            spectrum_available_count=np.concatenate(spectrum_available),
            spectrum_negative_count=np.concatenate(spectrum_negative),
            candidate_relative_thresholds=self.candidate_relative_thresholds,
            grid=self.grid,
            metadata={
                "target_field": self.target_field,
                "bin_radii": list(self.bin_radii),
                "feature_count": int(self.catalogue.feature_count),
                "population_size": int(processed),
                "batch_size": self.batch_size,
                "group_fields": self.group_fields,
                "state_codes": {"negative": NEGATIVE, "positive": POSITIVE, "unlabelled": UNLABELLED},
            },
        )

    def load_or_run(self, cache_path: str | Path, *, progress: bool = True) -> EvidenceStatistics:
        """Return a cached pass when present, otherwise run and store one.

        :param cache_path: ``.npz`` archive holding a previous pass.
        :param progress: Forwarded to :meth:`run`.
        :rtype: EvidenceStatistics
        """
        path = Path(cache_path)
        if path.is_file():
            logger.info("Reusing the cached evidence pass at %s.", path)
            return EvidenceStatistics.load(path)
        statistics = self.run(progress=progress)
        statistics.save(path)
        return statistics
