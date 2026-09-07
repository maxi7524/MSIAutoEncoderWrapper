"""Prediction and latent-geometry evaluation of every model in a sweep campaign.

A sweep evaluates the same quantities on tens of trained models that differ only in
one objective term. Two properties dominate the implementation:

1. **The split is decoded once, not once per model.** ``PixelDataset.__getitem__``
   decodes a spectrum from the raw reader on every access, so iterating a dataloader
   per model re-pays that cost for every model in the grid. Every function here
   consumes an already-materialized :class:`MaterializedSplit` instead of a dataset.
2. **Outputs are long-form.** One row per ``(model, split, scope, metric)`` rather
   than a wide per-model table, so repetitions stay individually visible and the
   distribution of a metric across seeds can be plotted from the stored records
   without recomputation.

Metric and geometry computations themselves are delegated to the project's existing
implementations; nothing here defines a metric of its own. Head metrics go through
:mod:`..heads.batch_metrics`, which evaluates every class in a few tensor operations on
the accelerator and is asserted to agree with the scikit-learn reference in
:mod:`..heads.metrics`; latent statistics go through :mod:`..latent.sphere_geometry`.
The batch path matters at sweep scale: the per-class reference costs about 4.5 s for a
single 1000-pixel, 500-class evaluation, which the accelerator returns in about 11 ms.
"""

from __future__ import annotations

import hashlib
import json

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ....utils.logger import get_custom_logger
from ..heads.batch_metrics import evaluate_head_batch, per_class_metrics_batch
from ..latent import sphere_geometry as geometry
from ..latent.batch_geometry import knn_overlap_batch, two_nn_intrinsic_dimension_batch
from .penalty_sweep import PenaltySweepCell

logger = get_custom_logger(__name__)

ALL_CLASSES_SCOPE = "all classes"
FAIR_SCOPE = "train-active and test-active"


def link_campaign_scratch_workspace(
    scratch_workspace: "Path | str", workspace: "Path | str"
) -> "Path":
    """Recreate a campaign's compute-node workspace path as a link to the local one.

    Every model saved by a SLURM campaign records the absolute reader and annotation
    paths of the compute node that trained it (e.g.
    ``/tmp/<user>/msi-wrapper/<campaign>/workspace``). Those paths do not exist in a
    local checkout, so ``load_configuration`` fails while resolving the annotation
    store. Linking the recorded path to the real workspace makes the saved config
    resolvable without rewriting stored artifacts.

    Loading a model through ``models_manager.load_model`` does not need this: it
    reconstructs only the architecture and weights and never re-resolves the model's
    own saved reader paths. Only the one ``load_configuration`` call that establishes
    the dataset context does.

    :param scratch_workspace: Absolute workspace path recorded in the saved configs.
    :type scratch_workspace: pathlib.Path | str
    :param workspace: This checkout's real workspace directory.
    :type workspace: pathlib.Path | str
    :return: The scratch path, now resolvable.
    :rtype: pathlib.Path
    :raises FileExistsError: If the scratch path already exists as a real directory,
        which this function must not replace.
    """
    scratch_path = Path(scratch_workspace)
    if scratch_path.is_symlink():
        logger.debug("Scratch workspace link already present: %s", scratch_path)
        return scratch_path
    if scratch_path.exists():
        raise FileExistsError(
            f"'{scratch_path}' exists as a real path; refusing to replace it with a "
            "link to the local workspace."
        )
    scratch_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_path.symlink_to(Path(workspace).resolve())
    logger.info("Linked campaign scratch workspace %s -> %s", scratch_path, workspace)
    return scratch_path


@dataclass(frozen=True)
class MaterializedSplit:
    """One dataset split decoded once into resident tensors.

    :param name: Split name (``train``, ``validation``, ``test``).
    :type name: str
    :param spectra: Decoded input spectra, shape ``(N, M)``.
    :type spectra: torch.Tensor
    :param targets: Binary multi-label targets, shape ``(N, C)``.
    :type targets: numpy.ndarray
    :param mask: Target-availability mask, shape ``(N, C)``.
    :type mask: numpy.ndarray
    :param indices: Dataset indices actually decoded, in ascending order. Persisted
        alongside results so the realized sample is reproducible independently of the
        random stream that produced it.
    :type indices: numpy.ndarray
    :param total: Size of the full split before subsampling.
    :type total: int
    :param fraction: Requested subsample fraction.
    :type fraction: float
    :param seed: Seed of the generator that drew ``indices``.
    :type seed: int
    """

    name: str
    spectra: torch.Tensor
    targets: np.ndarray
    mask: np.ndarray
    indices: np.ndarray
    total: int
    fraction: float
    seed: int

    @property
    def sampled(self) -> int:
        """Number of decoded pixels."""
        return int(self.indices.size)


def _resolve_dataset(dataset: Any) -> tuple[Any, Optional[np.ndarray]]:
    """Unwrap a ``Subset`` chain to the dataset that owns the batched read path.

    ``create_partitions`` returns ``torch.utils.data.Subset`` views, which forward
    ``__getitem__`` but expose none of the underlying dataset's batch methods. The
    batched decode therefore has to address the owning dataset directly, which means
    translating split-local positions into that dataset's own indices.

    :param dataset: A dataset or a chain of ``Subset`` views over one.
    :type dataset: Any
    :return: The owning dataset and, when the input was a view, the index map from
        split-local positions to that dataset's indices.
    :rtype: tuple[Any, numpy.ndarray | None]
    """
    mapping: Optional[np.ndarray] = None
    current = dataset
    while hasattr(current, "dataset") and hasattr(current, "indices"):
        level = np.asarray(current.indices, dtype=np.int64)
        mapping = level if mapping is None else level[mapping]
        current = current.dataset
    return current, mapping


def _decode_metadata(
    dataset: Any,
    split_name: str,
    target_field: str,
    fraction: float,
    seed: int,
    total: int,
) -> Dict[str, Any]:
    """Describe everything that determines a materialized split's contents.

    A cached split is only reusable when every one of these matches, so the record is
    compared field by field on load rather than reduced to a single opaque key. A
    silently stale cache would be worse than no cache at all: it would return the
    wrong pixels under the right name.
    """
    binner = dataset.active_context.binner
    return {
        "split": split_name,
        "target_field": target_field,
        "fraction": float(fraction),
        "seed": int(seed),
        "total": int(total),
        "normalization": getattr(dataset, "normalization", None),
        "binner_type": type(binner).__name__,
        "binner_config": getattr(binner, "_config", None),
    }


def _cache_paths(cache_directory: Path, metadata: Dict[str, Any]) -> tuple[Path, Path]:
    """Return the array and sidecar paths for one decode configuration."""
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    stem = f"{metadata['split']}_{digest}"
    return cache_directory / f"{stem}.npz", cache_directory / f"{stem}.json"


def materialize_split(
    partitions: Any,
    split_name: str,
    target_field: str,
    *,
    fraction: float = 1.0,
    seed: int = 42,
    decode_batch_size: int = 512,
    cache_directory: Optional[Any] = None,
) -> MaterializedSplit:
    """Decode a split once, in physical read order, for reuse by every model.

    Decoding dominates the cost of a sweep analysis, and three things make it slow if
    done naively. Reading one spectrum at a time seeks around the file, since a
    dataset's index order is not the file's storage order; binning and normalizing one
    spectrum at a time pays per-call overhead on work that is defined on a batch; and
    every notebook that needs the same split decodes it again from scratch.

    All three are addressed here. Spectra are requested through ``get_raw_batch``,
    whose reader sorts a batch by physical intensity offset before reading, so access
    becomes sequential. Binning and normalization run once per batch through the same
    ``binner.transform`` and ``normalize_batch`` the training pipeline uses, so no
    second implementation exists. With ``cache_directory`` set, the decoded arrays are
    written once and reused by later runs and by other notebooks.

    :param partitions: Dataset partitions exposing one dataset per split name (see
        ``active_dataset.create_partitions()``).
    :type partitions: Any
    :param split_name: Split attribute to materialize (``train``/``test``/...).
    :type split_name: str
    :param target_field: Target specification name to extract (e.g. ``molecule``).
    :type target_field: str
    :param fraction: Fraction of the split's pixels to decode, in ``(0, 1]``.
    :type fraction: float
    :param seed: Seed of the subsampling generator. The realized indices are also
        returned, since an identical seed does not guarantee an identical stream
        across library versions.
    :type seed: int
    :param decode_batch_size: Number of pixels requested per reader batch.
    :type decode_batch_size: int
    :param cache_directory: Directory for the decoded arrays; ``None`` disables
        caching. A cached split is reused only when the split, fraction, seed, target
        field, split size, normalization and binner configuration all match.
    :type cache_directory: pathlib.Path | str | None
    :return: The decoded split.
    :rtype: MaterializedSplit
    :raises ValueError: If ``fraction`` is outside ``(0, 1]``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")

    split_view = getattr(partitions, split_name)
    dataset, index_map = _resolve_dataset(split_view)
    total = len(split_view)
    sample_size = max(1, int(round(total * fraction)))

    ## Deterministic index draw, sorted so the request follows dataset order
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(total, size=sample_size, replace=False))

    metadata = _decode_metadata(dataset, split_name, target_field, fraction, seed, total)
    array_path = sidecar_path = None
    if cache_directory is not None:
        cache_root = Path(cache_directory)
        cache_root.mkdir(parents=True, exist_ok=True)
        array_path, sidecar_path = _cache_paths(cache_root, metadata)

        ### A cache hit still verifies the stored record, not only the file name
        if array_path.is_file() and sidecar_path.is_file():
            stored = json.loads(sidecar_path.read_text())
            if stored == metadata:
                with np.load(array_path) as archive:
                    cached_indices = archive["indices"]
                    if np.array_equal(cached_indices, indices):
                        logger.info(
                            "Reusing cached split '%s' (%s pixels) from %s.",
                            split_name,
                            cached_indices.size,
                            array_path,
                        )
                        return MaterializedSplit(
                            name=split_name,
                            spectra=torch.from_numpy(archive["spectra"]).float(),
                            targets=archive["targets"],
                            mask=archive["mask"],
                            indices=cached_indices,
                            total=total,
                            fraction=fraction,
                            seed=seed,
                        )

    ## Batched decode: one reader call, one binning call and one normalization per chunk
    binner = dataset.active_context.binner
    spectra_chunks, target_chunks, mask_chunks = [], [], []
    for start in range(0, indices.size, decode_batch_size):
        positions = indices[start : start + decode_batch_size]
        chunk = (positions if index_map is None else index_map[positions]).tolist()
        raw_batch = dataset.get_raw_batch(chunk)
        dense = binner.transform(raw_batch).spectra  # (B, M)
        normalized = dataset.normalize_batch(dense.to(torch.float32))  # (B, M)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(
                f"Split '{split_name}' produced non-finite values while decoding "
                f"pixels {chunk[0]}..{chunk[-1]}."
            )
        target_batch = dataset.get_target_batch(chunk)
        spectra_chunks.append(normalized.cpu())
        target_chunks.append(np.asarray(target_batch.values[target_field]))
        mask_chunks.append(np.asarray(target_batch.masks[target_field]))

    spectra = torch.cat(spectra_chunks, dim=0)  # (N, M)
    targets = np.concatenate(target_chunks, axis=0)  # (N, C)
    mask = np.concatenate(mask_chunks, axis=0)  # (N, C)
    logger.info(
        "Materialized split '%s': %s/%s pixel(s) decoded in %s batch(es) of %s.",
        split_name,
        sample_size,
        total,
        (indices.size + decode_batch_size - 1) // decode_batch_size,
        decode_batch_size,
    )

    if array_path is not None:
        np.savez(
            array_path,
            spectra=spectra.numpy(),
            targets=targets,
            mask=mask,
            indices=indices,
        )
        sidecar_path.write_text(json.dumps(metadata, sort_keys=True, default=str, indent=2))
        logger.info("Cached decoded split '%s' at %s.", split_name, array_path)

    return MaterializedSplit(
        name=split_name,
        spectra=spectra.float(),
        targets=targets,
        mask=mask,
        indices=indices,
        total=total,
        fraction=fraction,
        seed=seed,
    )


def fair_scope_mask(
    train_split: MaterializedSplit, test_split: MaterializedSplit
) -> np.ndarray:
    """Select classes that are observed positive in both the train and test samples.

    A class with no available positive in the training sample cannot have been
    learned, and one with none in the evaluation sample has an undefined average
    precision. Scoring over all classes therefore mixes a model-quality difference
    with a label-availability artifact; this mask removes the second.

    :param train_split: Materialized training split.
    :type train_split: MaterializedSplit
    :param test_split: Materialized evaluation split.
    :type test_split: MaterializedSplit
    :return: Boolean class mask, shape ``(C,)``.
    :rtype: numpy.ndarray
    """
    train_active = (train_split.targets * train_split.mask).sum(axis=0) > 0  # (C,)
    test_active = (test_split.targets * test_split.mask).sum(axis=0) > 0  # (C,)
    mask = train_active & test_active
    logger.info(
        "Fair scope: %s/%s class(es) active in train, %s also active in test.",
        int(train_active.sum()),
        train_active.size,
        int(mask.sum()),
    )
    return mask


def head_logits(
    model: Any,
    spectra: torch.Tensor,
    head_name: str,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Run one model's classification head over already-decoded spectra.

    :param model: Trained model in evaluation mode.
    :type model: torch.nn.Module
    :param spectra: Decoded input spectra, shape ``(N, M)``.
    :type spectra: torch.Tensor
    :param head_name: Head key, without the ``head_`` output prefix.
    :type head_name: str
    :param batch_size: Forward-pass batch size.
    :type batch_size: int
    :return: Unnormalized head outputs, shape ``(N, C)``.
    :rtype: numpy.ndarray
    """
    device = next(model.parameters()).device
    batches = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch_size):
            outputs = model(spectra[start : start + batch_size].to(device))
            batches.append(outputs[f"head_{head_name}"].cpu().numpy())  # (B, C)
    return np.concatenate(batches, axis=0)  # (N, C)


def canonical_latent(model: Any, spectra: torch.Tensor, *, batch_size: int = 256) -> np.ndarray:
    """Encode spectra and canonicalize the codes by the encoder's LayerNorm affine.

    :param model: Trained model in evaluation mode.
    :type model: torch.nn.Module
    :param spectra: Decoded input spectra, shape ``(N, M)``.
    :type spectra: torch.Tensor
    :param batch_size: Forward-pass batch size.
    :type batch_size: int
    :return: Canonicalized codes ``u = (z - beta) / gamma``, shape ``(N, D)``.
    :rtype: numpy.ndarray
    """
    device = next(model.parameters()).device
    gamma, beta = geometry.encoder_layer_norm_parameters(model)
    batches = []
    with torch.no_grad():
        for start in range(0, len(spectra), batch_size):
            latent = model(spectra[start : start + batch_size].to(device))["latent_space"]
            batches.append(latent.cpu().numpy())  # (B, D)
    return geometry.canonicalize(np.concatenate(batches, axis=0), gamma, beta)  # (N, D)


def _identity_fields(
    cell: PenaltySweepCell, model_name: str, campaign: str, repetition: Optional[int]
) -> Dict[str, Any]:
    """Build the grid-identity columns shared by every long-form output row."""
    return {
        "campaign": campaign,
        "model_name": model_name,
        "cell_label": cell.label,
        "penalty_metric": cell.penalty_metric,
        "input_geometry": cell.input_geometry,
        "weight": cell.weight,
        "repetition": repetition,
    }


def prediction_metrics_frame(
    logits: np.ndarray,
    split: MaterializedSplit,
    cell: PenaltySweepCell,
    *,
    model_name: str,
    campaign: str,
    repetition: Optional[int],
    scopes: Optional[Dict[str, Optional[np.ndarray]]] = None,
    threshold: float = 0.5,
    device: Optional[Any] = None,
) -> list[Dict[str, Any]]:
    """Evaluate one model's head into long-form records, one row per metric.

    Every metric :func:`~..heads.metrics.evaluate_head` returns is emitted, not a
    selected subset: threshold-free ranking metrics (``average_precision``,
    ``roc_auc``) and threshold-dependent ones (F1, precision, recall, Hamming)
    respond differently to a penalty that rescales the latent, and separating those
    two responses is only possible if both are recorded.

    :param logits: Head outputs for ``split``, shape ``(N, C)``.
    :type logits: numpy.ndarray
    :param split: The materialized split the logits were produced from.
    :type split: MaterializedSplit
    :param cell: Grid-cell identity of the evaluated model.
    :type cell: PenaltySweepCell
    :param model_name: Stored model identifier.
    :type model_name: str
    :param campaign: Campaign identifier.
    :type campaign: str
    :param repetition: Repetition index within the grid cell.
    :type repetition: int | None
    :param scopes: Mapping from scope name to a class mask (``None`` selects every
        class). Defaults to a single all-classes scope.
    :type scopes: Dict[str, numpy.ndarray | None] | None
    :param threshold: Probability threshold for the threshold-dependent metrics.
    :type threshold: float
    :param device: Device the metrics are evaluated on; defaults to CUDA when
        available. Every metric here is a sort followed by cumulative sums over the
        class axis, so evaluating a whole sweep on the accelerator rather than one
        class at a time on a CPU core is what keeps this tractable.
    :type device: str | torch.device | None
    :return: Records with the grid-identity columns plus ``split``, ``scope``,
        ``metric``, ``value`` and ``class_count``.
    :rtype: list[Dict[str, Any]]
    """
    scope_map = scopes if scopes is not None else {ALL_CLASSES_SCOPE: None}
    identity = _identity_fields(cell, model_name, campaign, repetition)

    rows: list[Dict[str, Any]] = []
    for scope_name, class_mask in scope_map.items():
        selector = slice(None) if class_mask is None else class_mask
        scored = evaluate_head_batch(
            logits[:, selector],
            split.targets[:, selector],
            split.mask[:, selector],
            threshold,
            device=device,
        )
        class_count = (
            logits.shape[1] if class_mask is None else int(np.asarray(class_mask).sum())
        )
        for metric, value in scored.items():
            rows.append(
                {
                    **identity,
                    "split": split.name,
                    "scope": scope_name,
                    "metric": metric,
                    "value": None if value is None else float(value),
                    "class_count": class_count,
                }
            )
    return rows


def per_class_metrics_frame(
    logits: np.ndarray,
    split: MaterializedSplit,
    cell: PenaltySweepCell,
    *,
    model_name: str,
    campaign: str,
    repetition: Optional[int],
    class_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    device: Optional[Any] = None,
) -> list[Dict[str, Any]]:
    """Evaluate one model's head per output class into long-form records.

    A macro average over several hundred mostly-rare classes hides whether a change
    acts on the rare or the frequent ones; this keeps the per-class resolution needed
    to tell those apart.

    :param logits: Head outputs for ``split``, shape ``(N, C)``.
    :type logits: numpy.ndarray
    :param split: The materialized split the logits were produced from.
    :type split: MaterializedSplit
    :param cell: Grid-cell identity of the evaluated model.
    :type cell: PenaltySweepCell
    :param model_name: Stored model identifier.
    :type model_name: str
    :param campaign: Campaign identifier.
    :type campaign: str
    :param repetition: Repetition index within the grid cell.
    :type repetition: int | None
    :param class_mask: Optional class selection; class indices in the output refer to
        positions in the *original* class axis, not in the masked subset.
    :type class_mask: numpy.ndarray | None
    :param threshold: Probability threshold for the threshold-dependent metrics.
    :type threshold: float
    :return: Records with the grid-identity columns plus ``split``, ``class_index``,
        ``positive_samples``, and one column per per-class metric.
    :rtype: list[Dict[str, Any]]
    """
    selector = slice(None) if class_mask is None else class_mask
    records = per_class_metrics_batch(
        logits[:, selector],
        split.targets[:, selector],
        split.mask[:, selector],
        threshold,
        device=device,
    )

    # Map masked positions back onto original class indices so per-class rows stay
    # joinable across scopes and campaigns.
    original_indices = (
        np.arange(logits.shape[1])
        if class_mask is None
        else np.flatnonzero(np.asarray(class_mask))
    )
    identity = _identity_fields(cell, model_name, campaign, repetition)
    rows = []
    for record in records:
        masked_index = int(record["class_index"])
        rows.append(
            {
                **identity,
                "split": split.name,
                **record,
                "class_index": int(original_indices[masked_index]),
            }
        )
    return rows


def latent_geometry_frame(
    latent: np.ndarray,
    cell: PenaltySweepCell,
    *,
    split_name: str,
    model_name: str,
    campaign: str,
    repetition: Optional[int],
    device: Optional[Any] = None,
) -> list[Dict[str, Any]]:
    """Summarize one model's canonicalized latent cloud into long-form records.

    :param latent: Canonicalized codes, shape ``(N, D)``.
    :type latent: numpy.ndarray
    :param cell: Grid-cell identity of the evaluated model.
    :type cell: PenaltySweepCell
    :param split_name: Split the codes were encoded from.
    :type split_name: str
    :param model_name: Stored model identifier.
    :type model_name: str
    :param campaign: Campaign identifier.
    :type campaign: str
    :param repetition: Repetition index within the grid cell.
    :type repetition: int | None
    :return: Records with the grid-identity columns plus ``split``, ``metric`` and
        ``value``, covering ``cloud_asymmetry``, ``trace``, ``effective_rank``,
        ``participation_ratio`` and ``two_nn_intrinsic_dimension``. The last is
        ``None`` when its estimate is not admissible (see below).
    :rtype: list[Dict[str, Any]]
    """
    usage = geometry.dimension_usage(latent)
    ambient_dimension = latent.shape[1]

    ## TwoNN admissibility
    ### REMARK: TwoNN assumes locally uniform density. Under a strong contractive
    ### penalty the cloud collapses (trace(Cov) -> 0), nearest and second-nearest
    ### distances become nearly equal, and the estimator diverges: a collapsed model
    ### in this campaign returned 127.7 on a 10-dimensional latent. An estimate above
    ### the ambient dimension is impossible rather than merely implausible, so it is
    ### reported as missing instead of being plotted next to valid values. The
    ### collapse itself stays visible through `trace`.
    two_nn = float(two_nn_intrinsic_dimension_batch(latent, device=device))
    if not np.isfinite(two_nn) or two_nn > ambient_dimension:
        logger.warning(
            "Discarding inadmissible TwoNN estimate %.3f for model '%s' (%s): it "
            "exceeds the ambient latent dimension %s, which indicates a collapsed "
            "cloud (trace(Cov)=%.4f), not an intrinsic dimension.",
            two_nn,
            model_name,
            cell.label,
            ambient_dimension,
            usage["trace"],
        )
        two_nn = None

    statistics = {
        "cloud_asymmetry": geometry.cloud_asymmetry(latent),
        "trace": usage["trace"],
        "effective_rank": usage["effective_rank"],
        "participation_ratio": usage["participation_ratio"],
        "two_nn_intrinsic_dimension": two_nn,
    }
    identity = _identity_fields(cell, model_name, campaign, repetition)
    return [
        {
            **identity,
            "split": split_name,
            "metric": metric,
            "value": None if value is None else float(value),
        }
        for metric, value in statistics.items()
    ]


def representation_stability_frame(
    latent_by_repetition: Dict[int, np.ndarray],
    cell: PenaltySweepCell,
    *,
    split_name: str,
    campaign: str,
    knn_neighbors: int = 10,
    device: Optional[Any] = None,
) -> list[Dict[str, Any]]:
    """Compare one grid cell's repetitions pairwise, as a seed-stability measure.

    A penalty that makes the representation reproducible across initializations is
    doing something a single-run geometry statistic cannot show. All three measures
    are invariant to the rotations that the latent is only defined up to.

    :param latent_by_repetition: Canonicalized codes per repetition, each ``(N, D)``
        and row-aligned across repetitions (the same materialized split).
    :type latent_by_repetition: Dict[int, numpy.ndarray]
    :param cell: Grid-cell identity shared by the repetitions.
    :type cell: PenaltySweepCell
    :param split_name: Split the codes were encoded from.
    :type split_name: str
    :param campaign: Campaign identifier.
    :type campaign: str
    :param knn_neighbors: Neighborhood size for ``knn_overlap``.
    :type knn_neighbors: int
    :return: One record per unordered repetition pair and measure, with the
        grid-identity columns plus ``repetition_a``, ``repetition_b``, ``metric``
        and ``value``.
    :rtype: list[Dict[str, Any]]
    """
    repetitions = sorted(latent_by_repetition)
    rows: list[Dict[str, Any]] = []
    for position, left in enumerate(repetitions):
        for right in repetitions[position + 1 :]:
            first, second = latent_by_repetition[left], latent_by_repetition[right]
            measures = {
                "procrustes_distance": geometry.procrustes_distance(first, second),
                "linear_cka": geometry.linear_cka(first, second),
                "knn_overlap": knn_overlap_batch(first, second, k=knn_neighbors, device=device),
            }
            for metric, value in measures.items():
                rows.append(
                    {
                        "campaign": campaign,
                        "cell_label": cell.label,
                        "penalty_metric": cell.penalty_metric,
                        "input_geometry": cell.input_geometry,
                        "weight": cell.weight,
                        "split": split_name,
                        "repetition_a": left,
                        "repetition_b": right,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows


def evaluate_campaign_models(
    model_records: Sequence[Dict[str, Any]],
    splits: Dict[str, MaterializedSplit],
    load_model: Callable[[str], Any],
    *,
    head_name: str,
    scopes: Optional[Dict[str, Optional[np.ndarray]]] = None,
    threshold: float = 0.5,
    batch_size: int = 256,
    collect_per_class: bool = True,
    per_class_scope_mask: Optional[np.ndarray] = None,
    collect_geometry: bool = True,
    geometry_sample_size: Optional[int] = None,
    geometry_seed: int = 42,
    metric_device: Optional[Any] = None,
) -> Dict[str, list[Dict[str, Any]]]:
    """Evaluate every model of a sweep on every materialized split, in one pass.

    Each model is loaded once and both its head outputs and its latent codes are
    taken from the same forward passes' worth of work, so adding the geometry
    analysis does not double the model-loading cost.

    :param model_records: Output rows of
        :func:`~.penalty_sweep.sweep_grid_frame`, filtered to the runs to evaluate.
        Each must carry ``model_name``, ``campaign``, ``repetition`` and the
        penalty-cell columns.
    :type model_records: Sequence[Dict[str, Any]]
    :param splits: Materialized splits to evaluate, keyed by split name.
    :type splits: Dict[str, MaterializedSplit]
    :param load_model: Callable mapping a stored model name to a loaded, ready model.
    :type load_model: Callable[[str], Any]
    :param head_name: Classification head key to evaluate.
    :type head_name: str
    :param scopes: Class scopes passed to :func:`prediction_metrics_frame`.
    :type scopes: Dict[str, numpy.ndarray | None] | None
    :param threshold: Probability threshold for threshold-dependent metrics.
    :type threshold: float
    :param batch_size: Forward-pass batch size.
    :type batch_size: int
    :param collect_per_class: Also produce per-class prediction records.
    :type collect_per_class: bool
    :param per_class_scope_mask: Class selection for the per-class records.
    :type per_class_scope_mask: numpy.ndarray | None
    :param collect_geometry: Also encode each model and summarize its latent cloud.
        Disable in an analysis that only needs head outputs; it saves one forward pass
        per model and split.
    :type collect_geometry: bool
    :param geometry_sample_size: Deterministically subsample this many rows before
        computing latent statistics. ``two_nn_intrinsic_dimension`` and
        ``knn_overlap`` are quadratic in the row count, so a split of several thousand
        pixels evaluated over tens of models is not tractable without a cap. The same
        row selection is used for every model and split, so the resulting statistics
        stay comparable. ``None`` uses every row.
    :type geometry_sample_size: int | None
    :param geometry_seed: Seed of the geometry subsample.
    :type geometry_seed: int
    :param metric_device: Device the head metrics are evaluated on; defaults to CUDA
        when available.
    :type metric_device: str | torch.device | None
    :return: Mapping with ``prediction``, ``per_class`` and ``geometry`` long-form
        record lists, plus ``latent_by_cell`` mapping ``(cell_label, split)`` to a
        per-repetition latent dictionary for
        :func:`representation_stability_frame`.
    :rtype: Dict[str, list[Dict[str, Any]]]
    """
    prediction_rows: list[Dict[str, Any]] = []
    per_class_rows: list[Dict[str, Any]] = []
    geometry_rows: list[Dict[str, Any]] = []
    latent_by_cell: Dict[tuple[str, str], Dict[int, np.ndarray]] = {}

    # Geometry row selection, drawn once so every model and split is summarized over
    # the same pixels and the statistics remain comparable across the grid.
    geometry_selection: Dict[str, Optional[np.ndarray]] = {}
    if collect_geometry and geometry_sample_size is not None:
        generator = np.random.default_rng(geometry_seed)
        for split_name, split in splits.items():
            available = split.spectra.shape[0]
            if available <= geometry_sample_size:
                geometry_selection[split_name] = None
                continue
            geometry_selection[split_name] = np.sort(
                generator.choice(available, size=geometry_sample_size, replace=False)
            )
            logger.info(
                "Latent geometry on split '%s' uses %s of %s row(s).",
                split_name,
                geometry_sample_size,
                available,
            )

    # Per-model evaluation
    for position, record in enumerate(model_records, start=1):
        model_name = record["model_name"]
        cell = PenaltySweepCell(
            penalty_metric=record["penalty_metric"],
            input_geometry=record["input_geometry"],
            weight=record["weight"],
            hinge_threshold=record.get("hinge_threshold"),
            hinge_alpha=record.get("hinge_alpha", 1.0),
            penalized_space=record.get("penalized_space"),
            calculation_method=record.get("calculation_method"),
        )
        model = load_model(model_name)
        model.eval()

        ## Every requested split, reusing the one loaded model
        for split_name, split in splits.items():
            logits = head_logits(model, split.spectra, head_name, batch_size=batch_size)
            prediction_rows.extend(
                prediction_metrics_frame(
                    logits,
                    split,
                    cell,
                    model_name=model_name,
                    campaign=record["campaign"],
                    repetition=record["repetition"],
                    scopes=scopes,
                    threshold=threshold,
                    device=metric_device,
                )
            )
            if collect_per_class:
                per_class_rows.extend(
                    per_class_metrics_frame(
                        logits,
                        split,
                        cell,
                        model_name=model_name,
                        campaign=record["campaign"],
                        repetition=record["repetition"],
                        class_mask=per_class_scope_mask,
                        threshold=threshold,
                        device=metric_device,
                    )
                )

            ## Latent geometry, on the shared row selection
            if not collect_geometry:
                continue
            latent = canonical_latent(model, split.spectra, batch_size=batch_size)
            selection = geometry_selection.get(split_name)
            if selection is not None:
                latent = latent[selection]  # (S, D)
            geometry_rows.extend(
                latent_geometry_frame(
                    latent,
                    cell,
                    split_name=split_name,
                    model_name=model_name,
                    campaign=record["campaign"],
                    repetition=record["repetition"],
                    device=metric_device,
                )
            )
            if record["repetition"] is not None:
                latent_by_cell.setdefault((cell.label, split_name), {})[
                    int(record["repetition"])
                ] = latent

        if position % 10 == 0 or position == len(model_records):
            logger.info("Evaluated %s/%s model(s).", position, len(model_records))

    return {
        "prediction": prediction_rows,
        "per_class": per_class_rows,
        "geometry": geometry_rows,
        "latent_by_cell": latent_by_cell,
    }
