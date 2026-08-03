"""Multi-model autoencoder analysis facade."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ...utils.exceptions import raise_validation_error
from ...visualization import VisualizationTheme
from .base import BaseAutoencoderAnalysis, ModelAnalysisContext, RETAINABLE_RESULTS
from .binning import BinningAnalysis
from .heads import HeadAnalysis
from .latent import LatentAnalysis
from .reconstruction import ReconstructionAnalysis
from .results import MultiPreparedAnalysis, PreparationEstimate


class AutoencoderMultiAnalysis(BaseAutoencoderAnalysis):
    """Analyze and compare multiple named autoencoders on one image.

    Models are restored sequentially from the wrapper's default image context,
    captured as independent runtime objects, and evaluated with analogous domain
    APIs to :class:`AutoencoderAnalysis`.

    :param wrapper: Wrapper with a configured default image.
    :type wrapper: Any
    :param model_names: Two or more saved autoencoder names.
    :type model_names: Sequence[str]
    :param theme: Global visualization strategy or preset name.
    :type theme: VisualizationTheme | str | None
    :raises ValidationError: If models do not share compatible dataset inputs.
    """

    def __init__(
        self,
        wrapper: Any,
        model_names: Sequence[str],
        theme: VisualizationTheme | str | None = None,
    ) -> None:
        super().__init__(wrapper, theme)
        names = list(dict.fromkeys(model_names))
        if len(names) < 2:
            raise_validation_error(
                "AutoencoderMultiAnalysis",
                "At least two distinct model names are required.",
            )
        reference_shape: Optional[tuple[int, ...]] = None
        reference_count: Optional[int] = None
        for model_name in names:
            wrapper.load_configuration(model_name=model_name)
            model = getattr(wrapper, "active_model", None)
            dataset = getattr(wrapper, "active_dataset", None)
            interface = getattr(wrapper.models_manager, "autoencoder", None)
            if model is None or dataset is None or interface is None or not interface.is_trained:
                raise_validation_error(
                    "AutoencoderMultiAnalysis",
                    f"Model '{model_name}' did not restore as a trained autoencoder.",
                )
            sample_shape = tuple(dataset[0][1].shape)
            if reference_shape is None:
                reference_shape = sample_shape
                reference_count = len(dataset)
            elif sample_shape != reference_shape or len(dataset) != reference_count:
                raise_validation_error(
                    "AutoencoderMultiAnalysis",
                    "All models must use datasets with equal samples and input shape.",
                )
            self.models[model_name] = ModelAnalysisContext(model_name, model, dataset)
        self._refresh_image_context()
        self.prepared: Optional[MultiPreparedAnalysis] = None
        self.reconstruction = ReconstructionAnalysis(self)
        self.latent = LatentAnalysis(self)
        self.heads = HeadAnalysis(self)
        self.binning = BinningAnalysis(self)

    def estimate_prepare_size(
        self,
        retain: Iterable[str] = RETAINABLE_RESULTS,
    ) -> PreparationEstimate:
        """Estimate multi-model memory with shared input and target storage."""
        retained = self._validate_retain(retain)
        combined: dict[str, int] = {}
        for index, model_name in enumerate(self.model_names):
            model_retain = retained if index == 0 else retained - {"inputs", "targets"}
            estimate = self.estimate_model_size(model_name, model_retain)
            for result_name, size in estimate.retained_bytes.items():
                key = (
                    result_name
                    if result_name in {"inputs", "targets", "target_masks"}
                    else f"{model_name}:{result_name}"
                )
                combined[key] = size
        sample_count = len(self.models[self.default_model_name].dataset)
        return PreparationEstimate(sample_count, combined)

    def prepare(
        self,
        retain: Iterable[str] = RETAINABLE_RESULTS,
        batch_size: Optional[int] = None,
        loader_options: Optional[Mapping[str, Any]] = None,
    ) -> MultiPreparedAnalysis:
        """Prepare every model while storing immutable inputs and targets once."""
        retained = self._validate_retain(retain)
        shared_inputs = None
        shared_targets = {}
        shared_masks = {}
        reference_ids = None
        for index, model_name in enumerate(self.model_names):
            model_retain = retained if index == 0 else retained - {"inputs", "targets"}
            prepared = self._prepare_model(
                self.models[model_name],
                model_retain,
                batch_size,
                loader_options,
            )
            if reference_ids is None:
                reference_ids = prepared.spectrum_ids
                shared_inputs = prepared.arrays.get("inputs")
                shared_targets = prepared.targets
                shared_masks = prepared.target_masks
            elif not np.array_equal(reference_ids, prepared.spectrum_ids):
                raise_validation_error(
                    "AutoencoderMultiAnalysis",
                    "Prepared spectrum identifiers differ between models.",
                )
            # Shared arrays are referenced, not copied. Domain methods can use
            # identical access paths for single- and multi-model analyses.
            if shared_inputs is not None:
                prepared.arrays["inputs"] = shared_inputs
            if shared_targets:
                prepared.targets = shared_targets
                prepared.target_masks = shared_masks
            self.prepared_models[model_name] = prepared
        self.prepared = MultiPreparedAnalysis(
            models=dict(self.prepared_models),
            shared_inputs=shared_inputs,
            shared_targets=shared_targets,
            shared_target_masks=shared_masks,
        )
        return self.prepared
