"""Public reconstruction-analysis group attached to an analyzer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import numpy as np

from ....readers.spatial import Aggregation, SpatialImage, aggregate_window
from ....utils.exceptions import raise_validation_error
from ....visualization.interactive import IonImageViewer
from .metrics import feature_error_distribution, rank_models, summarize
from .overviews import plot_selected_spectra as render_selected_spectra
from .views import (
    plot_error_images,
    plot_feature_profiles,
    plot_metric_distributions,
)


class ReconstructionAnalysis:
    """Expose reconstruction calculations and visualizations for an analyzer.

    :param owner: Single- or multi-model autoencoder analyzer.
    :type owner: Any
    """

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def summary(
        self,
        metric: str = "mse",
    ) -> Mapping[str, float] | Mapping[str, Mapping[str, float]]:
        """Summarize one metric for one model or every compared model.

        :param metric: Per-spectrum reconstruction metric.
        :type metric: str
        :return: One summary or model-keyed summaries.
        :rtype: Mapping[str, float] | Mapping[str, Mapping[str, float]]
        """
        summaries = {
            model_name: summarize(prepared.metric_array(metric))
            for model_name, prepared in self.owner.iter_prepared()
        }
        return summaries if self.owner.is_multi else summaries[self.owner.default_model_name]

    def compare_metric(self, metric: str = "mse") -> list[Dict[str, Any]]:
        """Return ranked model summaries for one metric.

        :param metric: Per-spectrum reconstruction metric.
        :type metric: str
        :return: Best-to-worst summary records.
        :rtype: list[Dict[str, Any]]
        """
        summaries = {
            model_name: summarize(prepared.metric_array(metric))
            for model_name, prepared in self.owner.iter_prepared()
        }
        ranking = rank_models(summaries, metric)
        return [
            {"model": name, "rank": index + 1, **summaries[name]}
            for index, name in enumerate(ranking)
        ]

    def metric_image(
        self,
        metric: str = "mse",
        annotation_filter: str = "all",
        target_field: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> SpatialImage | Mapping[str, SpatialImage]:
        """Map a metric to space with optional annotation availability filter.

        :param metric: Per-spectrum reconstruction metric.
        :type metric: str
        :param annotation_filter: ``all``, ``annotated``, or ``unannotated``.
        :type annotation_filter: str
        :param target_field: Optional annotation target used by the filter.
        :type target_field: str | None
        :param model_name: Optional model selection.
        :type model_name: str | None
        :return: One spatial image or model-keyed images.
        :rtype: SpatialImage | Mapping[str, SpatialImage]
        """
        if annotation_filter not in {"all", "annotated", "unannotated"}:
            raise_validation_error(
                "ReconstructionAnalysis",
                "annotation_filter must be all, annotated, or unannotated.",
            )
        selected_models = [model_name] if model_name else list(self.owner.model_names)
        images: Dict[str, SpatialImage] = {}
        for name in selected_models:
            prepared = self.owner.prepared_for(name)
            values = prepared.metric_array(metric).copy()
            if annotation_filter != "all":
                annotated = self.owner.annotation_mask(prepared, target_field)
                keep = annotated if annotation_filter == "annotated" else ~annotated
                values[~keep] = np.nan
            images[name] = self.owner.map_prepared_rows(prepared, values)
        if model_name or not self.owner.is_multi:
            return images[selected_models[0]]
        return images

    def select_spectra(
        self,
        metric: str = "mse",
        selection: str = "worst",
        count: int = 10,
        selection_model: Optional[str] = None,
    ) -> np.ndarray:
        """Select best, median, or worst identifiers using one model's score.

        :param metric: Per-spectrum metric.
        :type metric: str
        :param selection: ``best``, ``median``, or ``worst``.
        :type selection: str
        :param count: Maximum selected identifiers.
        :type count: int
        :param selection_model: Model defining selection in a multi-analysis.
        :type selection_model: str | None
        :return: Stable spectrum identifiers.
        :rtype: numpy.ndarray
        """
        if selection not in {"best", "median", "worst"} or count < 1:
            raise_validation_error(
                "ReconstructionAnalysis",
                "selection must be best, median, or worst and count must be positive.",
            )
        prepared = self.owner.prepared_for(selection_model)
        values = prepared.metric_array(metric)
        if selection == "best":
            rows = np.argsort(values)[:count]
        elif selection == "worst":
            rows = np.argsort(values)[-count:][::-1]
        else:
            rows = np.argsort(np.abs(values - np.median(values)))[:count]
        return prepared.spectrum_ids[rows]

    def ion_images(
        self,
        mz: float,
        tolerance: float = 0.0,
        aggregation: Aggregation = "mean",
    ) -> Mapping[str, SpatialImage]:
        """Return shared input plus each model reconstruction and residual image.

        :param mz: Center mass on the binner grid.
        :type mz: float
        :param tolerance: Absolute grid tolerance.
        :type tolerance: float
        :param aggregation: Bin aggregation strategy.
        :type aggregation: str | Callable[[numpy.ndarray], float]
        :return: Named spatial image collection.
        :rtype: Mapping[str, SpatialImage]
        """
        indices = self.owner.binner.GetBinIndices(mz, tolerance)
        if len(indices) == 0:
            raise_validation_error(
                "ReconstructionAnalysis", "The requested m/z window contains no bins."
            )
        first = self.owner.prepared_for()
        inputs = self.owner.require_array("inputs")
        input_values = _aggregate_rows(inputs[:, indices], aggregation)
        images: Dict[str, SpatialImage] = {
            "input": self.owner.map_prepared_rows(first, input_values)
        }
        for model_name, prepared in self.owner.iter_prepared():
            reconstruction = self.owner.require_array("reconstructions", model_name)
            reconstructed_values = _aggregate_rows(
                reconstruction[:, indices], aggregation
            )
            images[f"{model_name}: reconstruction"] = self.owner.map_prepared_rows(
                prepared, reconstructed_values
            )
            images[f"{model_name}: residual"] = self.owner.map_prepared_rows(
                prepared, input_values - reconstructed_values
            )
        return images

    def raw_ion_image(
        self,
        mz: float,
        tolerance: float,
        aggregation: Aggregation = "mean",
    ) -> SpatialImage:
        """Extract one ion image directly from the raw reader."""
        return self.owner.reader.GetIonImage(mz, tolerance, aggregation=aggregation)

    def feature_distribution(
        self,
        metric: str = "mse",
        quantiles: tuple[float, float] = (0.05, 0.95),
        chunk_size: int = 2048,
    ) -> Mapping[str, Mapping[str, np.ndarray]]:
        """Calculate feature-error distribution profiles for every model."""
        inputs = self.owner.require_array("inputs")
        return {
            model_name: feature_error_distribution(
                inputs,
                self.owner.require_array("reconstructions", model_name),
                metric=metric,
                quantiles=quantiles,
                chunk_size=chunk_size,
            )
            for model_name in self.owner.model_names
        }

    def plot_distribution(
        self,
        metric: str = "mse",
        bins: int = 60,
        ax: Any = None,
    ) -> Any:
        """Plot one overlaid metric distribution for all models."""
        return plot_metric_distributions(
            {
                name: prepared.metric_array(metric)
                for name, prepared in self.owner.iter_prepared()
            },
            metric,
            bins=bins,
            ax=ax,
            theme=self.owner.theme,
        )

    def plot_error_images(
        self,
        metric: str = "mse",
        annotation_filter: str = "all",
        target_field: Optional[str] = None,
    ) -> Any:
        """Plot filtered spatial error maps on a common scale."""
        images = self.metric_image(metric, annotation_filter, target_field)
        mapping = (
            images
            if isinstance(images, Mapping)
            else {self.owner.default_model_name: images}
        )
        return plot_error_images(
            {name: image.values for name, image in mapping.items()},
            metric,
            theme=self.owner.theme,
        )

    def plot_selected_spectra(
        self,
        metric: str = "mse",
        selection: str = "worst",
        count: int = 5,
        selection_model: Optional[str] = None,
        clip_reconstruction: bool = True,
    ) -> Any:
        """Plot selected inputs, all reconstructions, and signed residuals."""
        spectrum_ids = self.select_spectra(
            metric,
            selection,
            count,
            selection_model,
        )
        first = self.owner.prepared_for()
        input_array = self.owner.require_array("inputs")
        row_maps = {
            name: {int(identifier): row for row, identifier in enumerate(prepared.spectrum_ids)}
            for name, prepared in self.owner.iter_prepared()
        }
        first_rows = row_maps[self.owner.default_model_name]
        input_spectra = {
            int(identifier): input_array[first_rows[int(identifier)]]
            for identifier in spectrum_ids
        }
        reconstructions = {
            model_name: {
                int(identifier): self.owner.require_array(
                    "reconstructions", model_name
                )[row_maps[model_name][int(identifier)]]
                for identifier in spectrum_ids
            }
            for model_name in self.owner.model_names
        }
        selection_prepared = self.owner.prepared_for(selection_model)
        scores = {
            int(identifier): selection_prepared.pixel_metrics[int(identifier)][metric]
            for identifier in spectrum_ids
        }
        return render_selected_spectra(
            self.owner.binner.GetXAxis(),
            spectrum_ids,
            input_spectra,
            reconstructions,
            scores,
            selection,
            clip_reconstruction,
            self.owner.theme,
        )

    def plot_feature_error_overview(
        self,
        metric: str = "mse",
        quantiles: tuple[float, float] = (0.05, 0.95),
        top_n: int = 10,
        chunk_size: int = 2048,
        ax: Any = None,
    ) -> Any:
        """Plot feature means, distribution bands, and worst m/z positions."""
        profiles = self.feature_distribution(metric, quantiles, chunk_size)
        return plot_feature_profiles(
            self.owner.binner.GetXAxis(),
            profiles,
            metric,
            top_n=top_n,
            ax=ax,
            theme=self.owner.theme,
        )

    def ion_viewer(
        self,
        tolerance: float = 0.0,
        aggregation: Aggregation = "mean",
    ) -> IonImageViewer:
        """Return an interactive viewer over the active binner axis."""
        return IonImageViewer(
            self.owner.binner.GetXAxis(),
            lambda mass: {
                name: image.values
                for name, image in self.ion_images(
                    mass, tolerance=tolerance, aggregation=aggregation
                ).items()
            },
            theme=self.owner.theme,
        )

    def overview(
        self,
        metric: str = "mse",
        annotation_filter: str = "all",
        selected_count: int = 5,
        top_features: int = 10,
    ) -> Mapping[str, Any]:
        """Build the standard reconstruction visualization collection."""
        return {
            "comparison": self.compare_metric(metric),
            "distribution": self.plot_distribution(metric),
            "error_images": self.plot_error_images(metric, annotation_filter),
            "selected_spectra": self.plot_selected_spectra(
                metric, "worst", selected_count
            ),
            "feature_errors": self.plot_feature_error_overview(
                metric, top_n=top_features
            ),
        }


def _aggregate_rows(values: np.ndarray, aggregation: Aggregation) -> np.ndarray:
    """Apply reader-compatible aggregation independently to each row."""
    return np.asarray([aggregate_window(row, aggregation) for row in values])
