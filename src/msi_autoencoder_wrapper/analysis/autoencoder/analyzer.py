"""Single-model autoencoder analysis facade."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

from ...utils.exceptions import raise_validation_error
from ...visualization import VisualizationTheme
from .base import BaseAutoencoderAnalysis, ModelAnalysisContext, RETAINABLE_RESULTS
from .binning import BinningAnalysis
from .heads import HeadAnalysis
from .latent import LatentAnalysis
from .reconstruction import ReconstructionAnalysis
from .results import PreparationEstimate, PreparedAnalysis


class AutoencoderAnalysis(BaseAutoencoderAnalysis):
    """Analyze one named or currently active trained autoencoder.

    The facade owns model preparation while domain objects expose grouped APIs:
    ``analysis.reconstruction``, ``analysis.latent``, ``analysis.heads``, and
    ``analysis.binning``.

    :param wrapper: Ready wrapper or wrapper with a configured default image.
    :type wrapper: Any
    :param model_name: Saved model to load, or the active model when omitted.
    :type model_name: str | None
    :param theme: Global visualization strategy or preset name.
    :type theme: VisualizationTheme | str | None
    :raises ValidationError: If no trained model and dataset can be resolved.
    """

    def __init__(
        self,
        wrapper: Any,
        model_name: Optional[str] = None,
        theme: VisualizationTheme | str | None = None,
    ) -> None:
        super().__init__(wrapper, theme)
        if model_name is not None:
            wrapper.load_configuration(model_name=model_name)
        resolved_name = (
            model_name
            or getattr(wrapper.models_manager, "active_model_name", None)
            or getattr(wrapper.models_manager, "_active_model_name", None)
            or "active_model"
        )
        model = getattr(wrapper, "active_model", None)
        dataset = getattr(wrapper, "active_dataset", None)
        interface = getattr(wrapper.models_manager, "autoencoder", None)
        if model is None or dataset is None or interface is None:
            raise_validation_error(
                "AutoencoderAnalysis",
                "A named or active autoencoder with its dataset is required.",
            )
        if not interface.is_trained:
            raise_validation_error(
                "AutoencoderAnalysis", "The active autoencoder is not marked as trained."
            )
        self._refresh_image_context()
        self.models[resolved_name] = ModelAnalysisContext(
            resolved_name,
            model,
            dataset,
        )
        self.prepared: Optional[PreparedAnalysis] = None

        # Domain objects keep the public API navigable and share this owner's
        # model cache, theme, reader, binner, and spectrum-id mapping.
        self.reconstruction = ReconstructionAnalysis(self)
        self.latent = LatentAnalysis(self)
        self.heads = HeadAnalysis(self)
        self.binning = BinningAnalysis(self)

    @property
    def model(self) -> Any:
        """Return the analyzed PyTorch model for compatibility."""
        return self.models[self.default_model_name].model

    @property
    def dataset(self) -> Any:
        """Return the analyzed dataset for compatibility."""
        return self.models[self.default_model_name].dataset

    def estimate_prepare_size(
        self,
        retain: Iterable[str] = RETAINABLE_RESULTS,
    ) -> PreparationEstimate:
        """Estimate retained memory for this model's prepared analysis."""
        return self.estimate_model_size(self.default_model_name, retain)

    def prepare(
        self,
        retain: Iterable[str] = RETAINABLE_RESULTS,
        batch_size: Optional[int] = None,
        loader_options: Optional[Mapping[str, Any]] = None,
    ) -> PreparedAnalysis:
        """Execute one explicit full-dataset inference pass."""
        result = self._prepare_model(
            self.models[self.default_model_name],
            retain,
            batch_size,
            loader_options,
        )
        self.prepared_models[self.default_model_name] = result
        self.prepared = result
        return result

    # Compatibility layer
    ## Existing notebooks can migrate one method at a time to grouped domains.
    def reconstruction_summary(self, metric: str = "mse"):
        return self.reconstruction.summary(metric)

    def reconstruction_error_image(self, metric: str = "mse"):
        return self.reconstruction.metric_image(metric)

    def select_spectra(
        self,
        metric: str = "mse",
        selection: str = "worst",
        count: int = 10,
    ):
        return self.reconstruction.select_spectra(metric, selection, count)

    def ion_image_comparison(self, mz: float, tolerance: float = 0.0, aggregation="mean"):
        images = self.reconstruction.ion_images(mz, tolerance, aggregation)
        model_name = self.default_model_name
        return {
            "input": images["input"],
            "reconstruction": images[f"{model_name}: reconstruction"],
            "residual": images[f"{model_name}: residual"],
        }

    def raw_ion_image(self, mz: float, tolerance: float, aggregation="mean"):
        return self.reconstruction.raw_ion_image(mz, tolerance, aggregation)

    def latent_projection(self, method="pca", components=2, random_seed=0, **kwargs):
        return self.latent.project(method, components, random_seed, **kwargs)

    def latent_statistics(self):
        return self.latent.statistics()

    def latent_image(self, component: int):
        return self.latent.component_image(component)

    def evaluate_heads(self, threshold: float = 0.5):
        return self.heads.evaluate(threshold)

    def head_probability_image(self, head_name: str, class_index: int):
        return self.heads.probability_image(head_name, class_index)

    def binner_report(self, spectrum_ids: Optional[Sequence[int]] = None):
        return self.binning.forward_report(spectrum_ids)

    def inverse_binner_report(self, spectrum_ids: Optional[Sequence[int]] = None):
        return self.binning.inverse_report(spectrum_ids)
