"""Shared runtime and cache orchestration for autoencoder analyses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...readers.spatial import SpatialImage
from ...utils.exceptions import raise_validation_error
from ...utils.logger import get_custom_logger
from ...visualization import VisualizationTheme, resolve_theme
from .reconstruction.metrics import reconstruction_metrics
from .results import PreparationEstimate, PreparedAnalysis

logger = get_custom_logger(__name__)

RETAINABLE_RESULTS = {
    "inputs",
    "reconstructions",
    "latents",
    "head_outputs",
    "targets",
}


@dataclass(frozen=True)
class ModelAnalysisContext:
    """Runtime state captured for one named model.

    :param name: Stable model identifier.
    :type name: str
    :param model: Ready PyTorch architecture.
    :type model: torch.nn.Module
    :param dataset: Dataset configured with the saved model.
    :type dataset: Any
    """

    name: str
    model: torch.nn.Module
    dataset: Any


class BaseAutoencoderAnalysis:
    """Own common image context, theme, model runtimes, and prepared caches.

    Domain objects attached by concrete analyzers consume this interface and do
    not own model loading or full-dataset inference.

    :param wrapper: Wrapper with a default image or active model context.
    :type wrapper: Any
    :param theme: Global visualization strategy or preset name.
    :type theme: VisualizationTheme | str | None
    """

    def __init__(
        self,
        wrapper: Any,
        theme: VisualizationTheme | str | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.theme = resolve_theme(theme)
        self.context = getattr(wrapper, "active_context", None)
        self.reader = getattr(self.context, "reader", None)
        self.binner = getattr(self.context, "binner", None)
        self.models: Dict[str, ModelAnalysisContext] = {}
        self.prepared_models: Dict[str, PreparedAnalysis] = {}

    @property
    def model_names(self) -> tuple[str, ...]:
        """Return analyzed model names in deterministic display order.

        :return: Ordered model identifiers.
        :rtype: tuple[str, ...]
        """
        return tuple(self.models)

    @property
    def is_multi(self) -> bool:
        """Return whether more than one model belongs to this analysis.

        :return: Multi-model flag.
        :rtype: bool
        """
        return len(self.models) > 1

    @property
    def default_model_name(self) -> str:
        """Return the first configured model name.

        :return: Default model identifier.
        :rtype: str
        :raises ValidationError: If no models were registered.
        """
        if not self.models:
            raise_validation_error("AutoencoderAnalysis", "No models are configured.")
        return next(iter(self.models))

    def prepared_for(self, model_name: Optional[str] = None) -> PreparedAnalysis:
        """Return a prepared result for one model.

        :param model_name: Model identifier or the default model when omitted.
        :type model_name: str | None
        :return: Prepared model result.
        :rtype: PreparedAnalysis
        :raises ValidationError: If preparation has not run for the model.
        """
        resolved = model_name or self.default_model_name
        if resolved not in self.prepared_models:
            raise_validation_error(
                "AutoencoderAnalysis",
                f"Call prepare() before requesting results for '{resolved}'.",
            )
        return self.prepared_models[resolved]

    def iter_prepared(self) -> Iterable[tuple[str, PreparedAnalysis]]:
        """Yield named prepared results in model display order.

        :return: Iterator of model names and prepared results.
        :rtype: Iterable[tuple[str, PreparedAnalysis]]
        """
        for model_name in self.model_names:
            yield model_name, self.prepared_for(model_name)

    def estimate_model_size(
        self,
        model_name: str,
        retain: Iterable[str],
    ) -> PreparationEstimate:
        """Estimate retained memory for one configured model.

        :param model_name: Configured model identifier.
        :type model_name: str
        :param retain: Large result groups to retain.
        :type retain: Iterable[str]
        :return: Estimated retained memory.
        :rtype: PreparationEstimate
        """
        retained = self._validate_retain(retain)
        runtime = self.models[model_name]
        sample = runtime.dataset[0]
        spectrum = sample[1].unsqueeze(0).to(self._device()).float()
        runtime.model.eval()
        with torch.no_grad():
            outputs = runtime.model(spectrum)
        count = len(runtime.dataset)
        sizes: Dict[str, int] = {
            "spectrum_ids": count * np.dtype(np.int64).itemsize,
            "pixel_metrics": count * 7 * np.dtype(np.float64).itemsize,
            "feature_metrics": spectrum.shape[-1]
            * 3
            * np.dtype(np.float64).itemsize,
        }
        output_names = {
            "inputs": spectrum,
            "reconstructions": outputs.get("reconstruction"),
            "latents": outputs.get("latent_space"),
        }
        for result_name, value in output_names.items():
            if result_name in retained and value is not None:
                sizes[result_name] = count * value[0].numel() * value.element_size()
        if "head_outputs" in retained:
            sizes["head_outputs"] = count * sum(
                value[0].numel() * value.element_size()
                for name, value in outputs.items()
                if name.startswith("head_")
            )
        if "targets" in retained and len(sample) >= 4:
            sizes["targets"] = count * sum(
                value.numel() * value.element_size() for value in sample[2].values()
            )
            sizes["target_masks"] = count * len(sample[3])
        return PreparationEstimate(sample_count=count, retained_bytes=sizes)

    def _prepare_model(
        self,
        runtime: ModelAnalysisContext,
        retain: Iterable[str],
        batch_size: Optional[int],
        loader_options: Optional[Mapping[str, Any]],
    ) -> PreparedAnalysis:
        """Execute one model traversal and construct its reusable cache."""
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
                "prepare requires shuffle=False to preserve spectrum identifiers.",
            )

        # The following buckets are concatenated once after inference. This
        # avoids repeated array resizing while preserving dataset row order.
        arrays: Dict[str, list[np.ndarray]] = defaultdict(list)
        heads: Dict[str, list[np.ndarray]] = defaultdict(list)
        targets: Dict[str, list[np.ndarray]] = defaultdict(list)
        masks: Dict[str, list[np.ndarray]] = defaultdict(list)
        ids: list[np.ndarray] = []
        pixel_metrics: Dict[str, list[np.ndarray]] = defaultdict(list)
        feature_sums: Dict[str, Optional[np.ndarray]] = {
            "feature_mse": None,
            "feature_mae": None,
            "feature_bias": None,
        }
        processed = 0
        runtime.model.eval()
        logger.info(
            "Preparing analysis for model '%s' over %s spectra.",
            runtime.name,
            len(runtime.dataset),
        )
        with torch.no_grad():
            for batch in DataLoader(runtime.dataset, **options):
                batch_ids, spectra = batch[0], batch[1]
                device_spectra = spectra.to(self._device(), dtype=torch.float32)
                outputs = runtime.model(device_spectra)
                reconstruction = outputs.get("reconstruction")
                if reconstruction is None:
                    raise_validation_error(
                        "AutoencoderAnalysis",
                        f"Model '{runtime.name}' has no reconstruction output.",
                    )
                input_array = spectra.detach().cpu().numpy()
                reconstruction_array = reconstruction.detach().cpu().numpy()
                metrics = reconstruction_metrics(input_array, reconstruction_array)
                ids.append(np.asarray(batch_ids, dtype=np.int64))
                batch_count = len(input_array)
                processed += batch_count
                for metric_name in (
                    "mse",
                    "mae",
                    "cosine_similarity",
                    "spectral_angle",
                    "tic_input",
                    "tic_reconstruction",
                    "tic_error",
                ):
                    pixel_metrics[metric_name].append(metrics[metric_name])
                for metric_name in feature_sums:
                    contribution = metrics[metric_name] * batch_count
                    feature_sums[metric_name] = (
                        contribution
                        if feature_sums[metric_name] is None
                        else feature_sums[metric_name] + contribution
                    )
                if "inputs" in retained:
                    arrays["inputs"].append(input_array)
                if "reconstructions" in retained:
                    arrays["reconstructions"].append(reconstruction_array)
                if "latents" in retained:
                    latent = outputs.get("latent_space")
                    if latent is None:
                        raise_validation_error(
                            "AutoencoderAnalysis",
                            f"Model '{runtime.name}' has no latent output.",
                        )
                    arrays["latents"].append(latent.detach().cpu().numpy())
                if "head_outputs" in retained:
                    for output_name, value in outputs.items():
                        if output_name.startswith("head_"):
                            heads[output_name.removeprefix("head_")].append(
                                value.detach().cpu().numpy()
                            )
                if "targets" in retained and len(batch) >= 4:
                    for field, value in batch[2].items():
                        targets[field].append(value.detach().cpu().numpy())
                    for field, value in batch[3].items():
                        masks[field].append(value.detach().cpu().numpy())
                del outputs, reconstruction, device_spectra

        spectrum_ids = np.concatenate(ids)
        flattened = {
            name: np.concatenate(parts) for name, parts in pixel_metrics.items()
        }
        indexed_metrics = {
            int(spectrum_id): {
                name: float(values[row]) for name, values in flattened.items()
            }
            for row, spectrum_id in enumerate(spectrum_ids)
        }
        return PreparedAnalysis(
            spectrum_ids=spectrum_ids,
            pixel_metrics=indexed_metrics,
            feature_metrics={
                name: np.asarray(values) / processed
                for name, values in feature_sums.items()
                if values is not None
            },
            arrays={name: np.concatenate(parts) for name, parts in arrays.items()},
            head_outputs={name: np.concatenate(parts) for name, parts in heads.items()},
            targets={name: np.concatenate(parts) for name, parts in targets.items()},
            target_masks={name: np.concatenate(parts) for name, parts in masks.items()},
        )

    def map_rows(self, values: Sequence[float] | np.ndarray) -> SpatialImage:
        """Map row-aligned values through stable spectrum identifiers.

        :param values: Values aligned with a prepared cache.
        :type values: Sequence[float] | numpy.ndarray
        :return: Native spatial image.
        :rtype: SpatialImage
        """
        prepared = self.prepared_for()
        return self.map_prepared_rows(prepared, values)

    def map_prepared_rows(
        self,
        prepared: PreparedAnalysis,
        values: Sequence[float] | np.ndarray,
    ) -> SpatialImage:
        """Map values aligned with an explicit prepared result.

        :param prepared: Prepared result defining spectrum identifiers.
        :type prepared: PreparedAnalysis
        :param values: Row-aligned scalar values.
        :type values: Sequence[float] | numpy.ndarray
        :return: Native spatial image.
        :rtype: SpatialImage
        """
        full = np.full(self.reader.GetNumberOfSpectra(), np.nan, dtype=np.float64)
        full[prepared.spectrum_ids] = np.asarray(values)
        return self.reader.MapSpectrumValuesToImage(full)

    def annotation_mask(
        self,
        prepared: PreparedAnalysis,
        target_field: Optional[str] = None,
    ) -> np.ndarray:
        """Return row-level annotation availability from retained targets.

        :param prepared: Prepared model result.
        :type prepared: PreparedAnalysis
        :param target_field: Specific target field or all retained fields.
        :type target_field: str | None
        :return: Boolean mask aligned with prepared rows.
        :rtype: numpy.ndarray
        """
        fields = [target_field] if target_field else list(prepared.targets)
        if not fields:
            raise_validation_error(
                "AutoencoderAnalysis",
                "Targets were not retained; annotation filtering is unavailable.",
            )
        combined = np.zeros(len(prepared.spectrum_ids), dtype=bool)
        for field in fields:
            if field not in prepared.targets:
                raise_validation_error(
                    "AutoencoderAnalysis", f"Target field '{field}' was not retained."
                )
            target = np.asarray(prepared.targets[field])
            field_mask = (
                np.any(target > 0, axis=1) if target.ndim > 1 else np.ones(len(target), dtype=bool)
            )
            availability = prepared.target_masks.get(field)
            if availability is not None:
                field_mask &= np.asarray(availability, dtype=bool).reshape(-1)
            combined |= field_mask
        return combined

    def require_array(
        self,
        name: str,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Return one retained model array with a contextual validation error."""
        prepared = self.prepared_for(model_name)
        if name not in prepared.arrays:
            raise_validation_error(
                "AutoencoderAnalysis",
                f"'{name}' was not retained; include it in prepare(retain=...).",
            )
        return prepared.arrays[name]

    def selected_ids(
        self,
        spectrum_ids: Optional[Sequence[int]],
        dataset: Optional[Any] = None,
    ) -> np.ndarray:
        """Validate dataset indices used by diagnostic reports."""
        active_dataset = dataset or self.models[self.default_model_name].dataset
        selected = (
            np.arange(len(active_dataset), dtype=np.int64)
            if spectrum_ids is None
            else np.asarray(spectrum_ids, dtype=np.int64)
        )
        if selected.ndim != 1 or len(selected) == 0:
            raise_validation_error(
                "AutoencoderAnalysis", "At least one spectrum identifier is required."
            )
        if np.any(selected < 0) or np.any(selected >= len(active_dataset)):
            raise_validation_error(
                "AutoencoderAnalysis", "A spectrum identifier is outside the dataset."
            )
        return selected

    def _device(self) -> Any:
        return getattr(self.wrapper, "device", "cpu")

    @staticmethod
    def _validate_retain(retain: Iterable[str]) -> set[str]:
        retained = set(retain)
        unknown = retained - RETAINABLE_RESULTS
        if unknown:
            raise_validation_error(
                "AutoencoderAnalysis",
                "Unsupported retained results: " + ", ".join(sorted(unknown)) + ".",
            )
        return retained

    def _refresh_image_context(self) -> None:
        """Refresh reader and binner references after configuration loading."""
        self.context = getattr(self.wrapper, "active_context", None)
        self.reader = getattr(self.context, "reader", None)
        self.binner = getattr(self.context, "binner", None)
        if self.context is None or self.reader is None or self.binner is None:
            raise_validation_error(
                "AutoencoderAnalysis",
                "A configured active image reader and binner are required.",
            )
