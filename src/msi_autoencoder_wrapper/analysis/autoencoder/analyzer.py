"""Stateful, single-pass analysis facade for trained autoencoders."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from ...readers.spatial import Aggregation, SpatialImage, aggregate_window
from ...utils.exceptions import raise_validation_error
from ...utils.logger import get_custom_logger
from .metrics import evaluate_head, reconstruction_metrics, summarize
from .results import PreparationEstimate, PreparedAnalysis

logger = get_custom_logger(__name__)

_RETAINABLE = {"inputs", "reconstructions", "latents", "head_outputs", "targets"}


class AutoencoderAnalysis:
    """Analyze one trained autoencoder and its active single-image dataset.

    Initialization validates the wrapper state but performs no full-dataset
    inference. Call :meth:`estimate_prepare_size` before the explicit
    :meth:`prepare` pass when memory use matters.

    :param wrapper: Ready wrapper with a trained autoencoder and active dataset.
    :type wrapper: Any
    :raises ValidationError: If required model, dataset, reader, or binner state
        is unavailable.
    """

    def __init__(self, wrapper: Any) -> None:
        self.wrapper = wrapper
        self.dataset = getattr(wrapper, "active_dataset", None)
        self.context = getattr(wrapper, "active_context", None)
        manager = getattr(wrapper, "models_manager", None)
        self.interface = getattr(manager, "autoencoder", None)
        self.model = getattr(wrapper, "active_model", None)
        self.reader = getattr(self.context, "reader", None)
        self.binner = getattr(self.context, "binner", None)
        self.prepared: Optional[PreparedAnalysis] = None
        self._validate_runtime()

    def estimate_prepare_size(
        self,
        retain: Iterable[str] = _RETAINABLE,
    ) -> PreparationEstimate:
        """Estimate retained array memory without traversing the full dataset.

        One sample is passed through the model to resolve dynamic output shapes.

        :param retain: Names of large result groups to retain.
        :type retain: Iterable[str]
        :return: Estimated memory grouped by retained result name.
        :rtype: PreparationEstimate
        :raises ValidationError: If a retention name is unsupported.
        """
        retained = self._validate_retain(retain)
        sample = self.dataset[0]
        spectrum = sample[1].unsqueeze(0).to(self._device()).float()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(spectrum)
        sample_count = len(self.dataset)
        sizes: Dict[str, int] = {
            "spectrum_ids": sample_count * np.dtype(np.int64).itemsize
        }
        sizes["pixel_metrics"] = sample_count * 7 * np.dtype(np.float64).itemsize
        sizes["feature_metrics"] = (
            spectrum.shape[-1] * 3 * np.dtype(np.float64).itemsize
        )
        if "inputs" in retained:
            sizes["inputs"] = (
                sample_count * spectrum[0].numel() * spectrum.element_size()
            )
        if "reconstructions" in retained and "reconstruction" in outputs:
            value = outputs["reconstruction"]
            sizes["reconstructions"] = (
                sample_count * value[0].numel() * value.element_size()
            )
        if "latents" in retained and "latent_space" in outputs:
            value = outputs["latent_space"]
            sizes["latents"] = sample_count * value[0].numel() * value.element_size()
        if "head_outputs" in retained:
            sizes["head_outputs"] = sample_count * sum(
                value[0].numel() * value.element_size()
                for name, value in outputs.items()
                if name.startswith("head_")
            )
        if "targets" in retained and len(sample) >= 4:
            sizes["targets"] = sample_count * sum(
                value.numel() * value.element_size() for value in sample[2].values()
            )
            sizes["target_masks"] = sample_count * len(sample[3])
        return PreparationEstimate(sample_count=sample_count, retained_bytes=sizes)

    def prepare(
        self,
        retain: Iterable[str] = _RETAINABLE,
        batch_size: Optional[int] = None,
        loader_options: Optional[Mapping[str, Any]] = None,
    ) -> PreparedAnalysis:
        """Run one explicit inference pass and cache reusable analysis data.

        Reconstruction scalar and feature metrics are always calculated once.
        Large inputs, reconstructions, latents, head outputs, and targets are
        stored only when named in ``retain``.

        :param retain: Names of large result groups to retain.
        :type retain: Iterable[str]
        :param batch_size: Optional inference batch size.
        :type batch_size: int | None
        :param loader_options: Additional ``DataLoader`` keyword arguments.
        :type loader_options: Mapping[str, Any] | None
        :return: Prepared in-memory result cache.
        :rtype: PreparedAnalysis
        :raises ValidationError: If reconstruction output is unavailable.
        """
        retained = self._validate_retain(retain)
        options: Dict[str, Any] = {
            "batch_size": batch_size
            or getattr(self.wrapper.models_manager, "batch_size", 256),
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
        }
        if loader_options:
            options.update(loader_options)
        if options.get("shuffle"):
            raise_validation_error(
                "AutoencoderAnalysis",
                "prepare requires shuffle=False to preserve row order.",
            )
        buckets: Dict[str, list[np.ndarray]] = defaultdict(list)
        head_buckets: Dict[str, list[np.ndarray]] = defaultdict(list)
        target_buckets: Dict[str, list[np.ndarray]] = defaultdict(list)
        mask_buckets: Dict[str, list[np.ndarray]] = defaultdict(list)
        ids: list[np.ndarray] = []
        metric_buckets: Dict[str, list[np.ndarray]] = defaultdict(list)
        feature_sums: Dict[str, Optional[np.ndarray]] = {
            "feature_mse": None,
            "feature_mae": None,
            "feature_bias": None,
        }
        processed = 0

        logger.info(
            "Preparing autoencoder analysis over %s spectra.", len(self.dataset)
        )
        self.model.eval()
        with torch.no_grad():
            for batch in DataLoader(self.dataset, **options):
                batch_ids, spectra = batch[0], batch[1]
                device_spectra = spectra.to(
                    device=self._device(),
                    dtype=torch.float32,
                )
                outputs = self.model(device_spectra)
                reconstruction = outputs.get("reconstruction")
                if reconstruction is None:
                    raise_validation_error(
                        "AutoencoderAnalysis",
                        "The active model does not expose a reconstruction output.",
                    )
                input_array = spectra.detach().cpu().numpy()
                reconstruction_array = reconstruction.detach().cpu().numpy()
                batch_metrics = reconstruction_metrics(
                    input_array, reconstruction_array
                )
                ids.append(np.asarray(batch_ids, dtype=np.int64))
                batch_count = len(input_array)
                processed += batch_count
                for name in (
                    "mse",
                    "mae",
                    "cosine_similarity",
                    "spectral_angle",
                    "tic_input",
                    "tic_reconstruction",
                    "tic_error",
                ):
                    metric_buckets[name].append(batch_metrics[name])
                for name in feature_sums:
                    contribution = batch_metrics[name] * batch_count
                    feature_sums[name] = (
                        contribution
                        if feature_sums[name] is None
                        else feature_sums[name] + contribution
                    )
                if "inputs" in retained:
                    buckets["inputs"].append(input_array)
                if "reconstructions" in retained:
                    buckets["reconstructions"].append(reconstruction_array)
                if "latents" in retained:
                    buckets["latents"].append(
                        outputs["latent_space"].detach().cpu().numpy()
                    )
                if "head_outputs" in retained:
                    for output_name, value in outputs.items():
                        if output_name.startswith("head_"):
                            head_buckets[output_name.removeprefix("head_")].append(
                                value.detach().cpu().numpy()
                            )
                if "targets" in retained and len(batch) >= 4:
                    for field, value in batch[2].items():
                        target_buckets[field].append(value.detach().cpu().numpy())
                    for field, value in batch[3].items():
                        mask_buckets[field].append(value.detach().cpu().numpy())
                del reconstruction, outputs, device_spectra

        spectrum_ids = np.concatenate(ids)
        flat_metrics = {
            name: np.concatenate(parts) for name, parts in metric_buckets.items()
        }
        pixel_metrics = {
            int(spectrum_id): {
                name: float(values[row]) for name, values in flat_metrics.items()
            }
            for row, spectrum_id in enumerate(spectrum_ids)
        }
        self.prepared = PreparedAnalysis(
            spectrum_ids=spectrum_ids,
            pixel_metrics=pixel_metrics,
            feature_metrics={
                name: np.asarray(values) / processed
                for name, values in feature_sums.items()
                if values is not None
            },
            arrays={name: np.concatenate(parts) for name, parts in buckets.items()},
            head_outputs={
                name: np.concatenate(parts) for name, parts in head_buckets.items()
            },
            targets={
                name: np.concatenate(parts) for name, parts in target_buckets.items()
            },
            target_masks={
                name: np.concatenate(parts) for name, parts in mask_buckets.items()
            },
        )
        logger.info("Autoencoder analysis preparation completed.")
        return self.prepared

    def reconstruction_summary(self, metric: str = "mse") -> Mapping[str, float]:
        """Return descriptive statistics for one cached reconstruction metric.

        :param metric: Per-spectrum metric name.
        :type metric: str
        :return: Distribution summary.
        :rtype: Mapping[str, float]
        """
        return summarize(self._require_prepared().metric_array(metric))

    def reconstruction_error_image(self, metric: str = "mse") -> SpatialImage:
        """Map a cached reconstruction metric to the native image grid.

        :param metric: Per-spectrum metric name.
        :type metric: str
        :return: Spatial metric image.
        :rtype: SpatialImage
        """
        return self._map_rows(self._require_prepared().metric_array(metric))

    def select_spectra(
        self,
        metric: str = "mse",
        selection: str = "worst",
        count: int = 10,
    ) -> np.ndarray:
        """Return stable identifiers for best, worst, or median spectra.

        :param metric: Per-spectrum metric name.
        :type metric: str
        :param selection: ``best``, ``worst``, or ``median``.
        :type selection: str
        :param count: Maximum identifiers to return.
        :type count: int
        :return: Selected stable spectrum identifiers.
        :rtype: numpy.ndarray
        :raises ValidationError: If selection settings are invalid.
        """
        if count < 1 or selection not in {"best", "worst", "median"}:
            raise_validation_error(
                "AutoencoderAnalysis",
                "selection must be best, worst, or median and count must be positive.",
            )
        prepared = self._require_prepared()
        values = prepared.metric_array(metric)
        if selection == "best":
            rows = np.argsort(values)[:count]
        elif selection == "worst":
            rows = np.argsort(values)[-count:][::-1]
        else:
            rows = np.argsort(np.abs(values - np.median(values)))[:count]
        return prepared.spectrum_ids[rows]

    def latent_projection(
        self,
        method: str = "pca",
        components: int = 2,
        random_seed: int = 0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Project cached latent vectors with PCA or t-SNE.

        :param method: ``pca`` or ``tsne``.
        :type method: str
        :param components: Output dimension count.
        :type components: int
        :param random_seed: Reproducible estimator seed.
        :type random_seed: int
        :param kwargs: Additional estimator parameters.
        :return: Projected latent matrix.
        :rtype: numpy.ndarray
        :raises ValidationError: If latents were not retained or method is unknown.
        """
        latents = self._require_array("latents")
        if method == "pca":
            return PCA(
                n_components=components, random_state=random_seed, **kwargs
            ).fit_transform(latents)
        if method == "tsne":
            return TSNE(
                n_components=components, random_state=random_seed, **kwargs
            ).fit_transform(latents)
        raise_validation_error(
            "AutoencoderAnalysis", "Projection method must be 'pca' or 'tsne'."
        )

    def latent_statistics(self) -> Dict[str, np.ndarray]:
        """Return reusable distribution statistics for each latent component.

        :return: Per-component moments, extrema, and the correlation matrix.
        :rtype: Dict[str, numpy.ndarray]
        """
        latents = self._require_array("latents")
        return {
            "mean": np.mean(latents, axis=0),
            "std": np.std(latents, axis=0),
            "min": np.min(latents, axis=0),
            "max": np.max(latents, axis=0),
            "correlation": np.corrcoef(latents, rowvar=False),
            "norm": np.linalg.norm(latents, axis=1),
        }

    def latent_image(self, component: int) -> SpatialImage:
        """Map one retained latent component to the spatial grid.

        :param component: Zero-based latent component index.
        :type component: int
        :return: Spatial latent-component image.
        :rtype: SpatialImage
        """
        latents = self._require_array("latents")
        return self._map_rows(latents[:, component])

    def ion_image_comparison(
        self,
        mz: float,
        tolerance: float = 0.0,
        aggregation: Aggregation = "mean",
    ) -> Dict[str, SpatialImage]:
        """Compare cached binned input and reconstruction at one m/z window.

        This method reports the model-grid comparison. Raw reader extraction is
        available independently through :meth:`raw_ion_image`.

        :param mz: Center mass value on the binner axis.
        :type mz: float
        :param tolerance: Non-negative absolute m/z tolerance.
        :type tolerance: float
        :param aggregation: Aggregation across selected bins.
        :type aggregation: str | Callable[[numpy.ndarray], float]
        :return: Input, reconstruction, and residual spatial images.
        :rtype: Dict[str, SpatialImage]
        :raises ValidationError: If required arrays or bins are unavailable.
        """
        inputs = self._require_array("inputs")
        reconstructions = self._require_array("reconstructions")
        indices = self.binner.GetBinIndices(mz, tolerance)
        if len(indices) == 0:
            raise_validation_error(
                "AutoencoderAnalysis", "The requested m/z window contains no bins."
            )
        input_values = np.asarray(
            [aggregate_window(row[indices], aggregation) for row in inputs]
        )
        reconstruction_values = np.asarray(
            [aggregate_window(row[indices], aggregation) for row in reconstructions]
        )
        return {
            "input": self._map_rows(input_values),
            "reconstruction": self._map_rows(reconstruction_values),
            "residual": self._map_rows(input_values - reconstruction_values),
        }

    def raw_ion_image(
        self,
        mz: float,
        tolerance: float,
        aggregation: Aggregation = "mean",
    ) -> SpatialImage:
        """Extract one ion image directly from the active raw reader.

        :param mz: Center mass value.
        :type mz: float
        :param tolerance: Non-negative absolute m/z tolerance.
        :type tolerance: float
        :param aggregation: Raw window aggregation.
        :type aggregation: str | Callable[[numpy.ndarray], float]
        :return: Native raw ion image.
        :rtype: SpatialImage
        """
        return self.reader.GetIonImage(mz, tolerance, aggregation=aggregation)

    def binner_report(
        self,
        spectrum_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        """Report basic forward-binner behavior on selected raw spectra.

        The report is intentionally separate from reconstruction metrics: it
        measures the preprocessing transform, not the autoencoder.

        :param spectrum_ids: Stable identifiers to inspect, or all identifiers.
        :type spectrum_ids: Sequence[int] | None
        :return: Aggregate finite-value and TIC-preservation diagnostics.
        :rtype: Dict[str, float]
        """
        selected = self._selected_ids(spectrum_ids)
        finite_values: list[float] = []
        tic_ratios: list[float] = []
        for spectrum_id in selected:
            mz_axis, intensities = self.reader.GetSpectrum(int(spectrum_id))
            binned = np.asarray(self.binner(xs=mz_axis, ys=intensities))
            finite_values.append(float(np.mean(np.isfinite(binned))))
            raw_tic = float(np.sum(intensities))
            binned_tic = float(np.sum(binned))
            tic_ratios.append(binned_tic / raw_tic if raw_tic != 0 else np.nan)
        return {
            "spectrum_count": float(len(selected)),
            "finite_fraction": float(np.mean(finite_values)),
            "mean_tic_ratio": float(np.nanmean(tic_ratios)),
            "min_tic_ratio": float(np.nanmin(tic_ratios)),
            "max_tic_ratio": float(np.nanmax(tic_ratios)),
        }

    def inverse_binner_report(
        self,
        spectrum_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        """Report inverse-binner loss through an inverse/forward round trip.

        :param spectrum_ids: Stable identifiers to inspect, or all identifiers.
        :type spectrum_ids: Sequence[int] | None
        :return: Aggregate round-trip MSE, MAE, and TIC diagnostics.
        :rtype: Dict[str, float]
        :raises ValidationError: If no inverse binner is configured.
        """
        inverse_binner = getattr(self.context, "inverse_binner", None)
        if inverse_binner is None:
            raise_validation_error(
                "AutoencoderAnalysis",
                "The active context has no inverse binner to evaluate.",
            )
        selected = self._selected_ids(spectrum_ids)
        mse_values: list[float] = []
        mae_values: list[float] = []
        tic_ratios: list[float] = []
        for spectrum_id in selected:
            sample = self.dataset[int(spectrum_id)]
            binned = np.asarray(sample[1], dtype=np.float64)
            mz_axis, intensities = inverse_binner(binned)
            round_trip = np.asarray(
                self.binner(xs=mz_axis, ys=intensities), dtype=np.float64
            )
            residual = binned - round_trip
            mse_values.append(float(np.mean(residual**2)))
            mae_values.append(float(np.mean(np.abs(residual))))
            original_tic = float(np.sum(binned))
            round_trip_tic = float(np.sum(round_trip))
            tic_ratios.append(
                round_trip_tic / original_tic if original_tic != 0 else np.nan
            )
        return {
            "spectrum_count": float(len(selected)),
            "mean_mse": float(np.mean(mse_values)),
            "mean_mae": float(np.mean(mae_values)),
            "mean_tic_ratio": float(np.nanmean(tic_ratios)),
        }

    def evaluate_heads(self, threshold: float = 0.5) -> Dict[str, Dict[str, Any]]:
        """Evaluate every retained head through its configured target binding.

        :param threshold: Multi-label probability threshold.
        :type threshold: float
        :return: Metrics grouped by head identifier.
        :rtype: Dict[str, Dict[str, Any]]
        :raises ValidationError: If head outputs or bound targets were not retained.
        """
        prepared = self._require_prepared()
        head_specs = getattr(self.model, "head_specs", {})
        target_specs = getattr(self.dataset, "target_specs", {})
        results: Dict[str, Dict[str, Any]] = {}
        for head_name, logits in prepared.head_outputs.items():
            target_field = head_specs.get(head_name, {}).get("target_field")
            if not target_field or target_field not in prepared.targets:
                raise_validation_error(
                    "AutoencoderAnalysis",
                    f"Head '{head_name}' has no retained bound target.",
                )
            target_type = target_specs.get(target_field, {}).get("type")
            results[head_name] = evaluate_head(
                logits,
                prepared.targets[target_field],
                target_type,
                prepared.target_masks.get(target_field),
                threshold,
            )
        return results

    def head_probability_image(
        self,
        head_name: str,
        class_index: int,
    ) -> SpatialImage:
        """Map one head class probability to the spatial grid.

        :param head_name: Configured head identifier.
        :type head_name: str
        :param class_index: Zero-based output class index.
        :type class_index: int
        :return: Spatial probability image.
        :rtype: SpatialImage
        :raises ValidationError: If the head output is unavailable.
        """
        prepared = self._require_prepared()
        if head_name not in prepared.head_outputs:
            raise_validation_error(
                "AutoencoderAnalysis", f"Head '{head_name}' was not retained."
            )
        logits = prepared.head_outputs[head_name]
        target_field = (
            getattr(self.model, "head_specs", {}).get(head_name, {}).get("target_field")
        )
        target_type = (
            getattr(self.dataset, "target_specs", {}).get(target_field, {}).get("type")
        )
        if target_type == "multi_label":
            probabilities = 1.0 / (1.0 + np.exp(-logits))
        else:
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        return self._map_rows(probabilities[:, class_index])

    def _validate_runtime(self) -> None:
        """Validate all single-image analysis dependencies."""
        missing = [
            name
            for name, value in (
                ("active dataset", self.dataset),
                ("active context", self.context),
                ("trained autoencoder interface", self.interface),
                ("active model", self.model),
                ("reader", self.reader),
                ("binner", self.binner),
            )
            if value is None
        ]
        if missing:
            raise_validation_error(
                "AutoencoderAnalysis",
                "Missing required wrapper state: " + ", ".join(missing) + ".",
            )
        if not self.interface.is_trained:
            raise_validation_error(
                "AutoencoderAnalysis",
                "The active autoencoder is not marked as trained.",
            )
        if len(self.dataset) < 1:
            raise_validation_error(
                "AutoencoderAnalysis", "The active dataset contains no spectra."
            )

    def _device(self) -> Any:
        return getattr(self.wrapper, "device", "cpu")

    def _validate_retain(self, retain: Iterable[str]) -> set[str]:
        retained = set(retain)
        unknown = retained - _RETAINABLE
        if unknown:
            raise_validation_error(
                "AutoencoderAnalysis",
                "Unsupported retained results: " + ", ".join(sorted(unknown)) + ".",
            )
        return retained

    def _require_prepared(self) -> PreparedAnalysis:
        if self.prepared is None:
            raise_validation_error(
                "AutoencoderAnalysis",
                "Call prepare() before requesting analysis results.",
            )
        return self.prepared

    def _require_array(self, name: str) -> np.ndarray:
        prepared = self._require_prepared()
        if name not in prepared.arrays:
            raise_validation_error(
                "AutoencoderAnalysis",
                f"'{name}' was not retained; include it in prepare(retain=...).",
            )
        return prepared.arrays[name]

    def _map_rows(self, values: Sequence[float] | np.ndarray) -> SpatialImage:
        prepared = self._require_prepared()
        full_values = np.full(
            self.reader.GetNumberOfSpectra(), np.nan, dtype=np.float64
        )
        full_values[prepared.spectrum_ids] = np.asarray(values)
        return self.reader.MapSpectrumValuesToImage(full_values)

    def _selected_ids(self, spectrum_ids: Optional[Sequence[int]]) -> np.ndarray:
        selected = (
            np.arange(len(self.dataset), dtype=np.int64)
            if spectrum_ids is None
            else np.asarray(spectrum_ids, dtype=np.int64)
        )
        if selected.ndim != 1 or len(selected) == 0:
            raise_validation_error(
                "AutoencoderAnalysis", "At least one spectrum identifier is required."
            )
        if np.any(selected < 0) or np.any(selected >= len(self.dataset)):
            raise_validation_error(
                "AutoencoderAnalysis", "A spectrum identifier is outside the dataset."
            )
        return selected
