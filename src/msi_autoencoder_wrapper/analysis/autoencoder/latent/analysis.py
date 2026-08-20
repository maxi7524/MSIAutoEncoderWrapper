"""Public latent-analysis group attached to an analyzer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA

from ..heads.metrics import probabilities_from_logits
from .metrics import latent_statistics, project_latents
from .overviews import plot_component_grid
from .views import plot_multilabel_projection, plot_projection_grid


class LatentAnalysis:
    """Expose latent calculations and visualizations for one analyzer."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def statistics(self) -> Mapping[str, Any]:
        """Return one statistics mapping or mappings keyed by model."""
        values = {
            name: latent_statistics(self.owner.require_array("latents", name))
            for name in self.owner.model_names
        }
        return values if self.owner.is_multi else values[self.owner.default_model_name]

    def project(
        self,
        method: str = "pca",
        components: int = 2,
        random_seed: int = 0,
        **kwargs: Any,
    ) -> Mapping[str, np.ndarray] | np.ndarray:
        """Project every model independently using equivalent parameters."""
        projections = {
            name: project_latents(
                self.owner.require_array("latents", name),
                method,
                components,
                random_seed,
                **kwargs,
            )
            for name in self.owner.model_names
        }
        return (
            projections
            if self.owner.is_multi
            else projections[self.owner.default_model_name]
        )

    def component_image(
        self,
        component: int,
        model_name: Optional[str] = None,
    ):
        """Map one raw latent component to space."""
        names = [model_name] if model_name else list(self.owner.model_names)
        images = {
            name: self.owner.map_prepared_rows(
                self.owner.prepared_for(name),
                self.owner.require_array("latents", name)[:, component],
            )
            for name in names
        }
        return images[names[0]] if model_name or not self.owner.is_multi else images

    def pca_component_image(
        self,
        component: int,
        model_name: Optional[str] = None,
        random_seed: int = 0,
    ):
        """Map one independently fitted PCA score component to space."""
        names = [model_name] if model_name else list(self.owner.model_names)
        images = {}
        for name in names:
            latents = self.owner.require_array("latents", name)
            scores = PCA(
                n_components=component + 1,
                random_state=random_seed,
            ).fit_transform(latents)
            images[name] = self.owner.map_prepared_rows(
                self.owner.prepared_for(name), scores[:, component]
            )
        return images[names[0]] if model_name or not self.owner.is_multi else images

    def plot_projection(
        self,
        method: str = "pca",
        labels: Optional[np.ndarray] = None,
        random_seed: int = 0,
        **kwargs: Any,
    ):
        """Plot every model projection in a separate aligned panel."""
        projected = self.project(method, 2, random_seed, **kwargs)
        mapping = (
            projected
            if isinstance(projected, Mapping)
            else {self.owner.default_model_name: projected}
        )
        return plot_projection_grid(
            mapping,
            labels,
            method.upper(),
            self.owner.theme,
        )

    def plot_components(
        self,
        component_indices: Sequence[int],
        source: str = "latent",
        page_size: int = 16,
    ):
        """Plot raw latent or independently fitted PCA spatial components."""
        components = list(component_indices)
        if page_size < 1 or not components:
            raise ValueError("page_size and component_indices must be non-empty.")
        if len(components) > page_size:
            return {
                page: self.plot_components(
                    components[start : start + page_size],
                    source,
                    page_size,
                )
                for page, start in enumerate(range(0, len(components), page_size), 1)
            }
        images = {}
        explained_variance = {}
        for model_name in self.owner.model_names:
            images[model_name] = {}
            if source == "latent":
                for component in components:
                    images[model_name][component] = self.component_image(
                        component, model_name
                    ).values
            elif source == "pca":
                # One PCA fit per model/page
                ## Fitting once avoids repeating an expensive decomposition for
                ## every displayed component and exposes explained variance.
                latents = self.owner.require_array("latents", model_name)
                decomposition = PCA(n_components=max(components) + 1).fit(latents)
                scores = decomposition.transform(latents)
                explained_variance[model_name] = {
                    component: float(decomposition.explained_variance_ratio_[component])
                    for component in components
                }
                prepared = self.owner.prepared_for(model_name)
                for component in components:
                    images[model_name][component] = self.owner.map_prepared_rows(
                        prepared, scores[:, component]
                    ).values
            else:
                raise ValueError("source must be 'latent' or 'pca'.")
        return plot_component_grid(
            images,
            components,
            source.upper(),
            self.owner.theme,
            explained_variance or None,
        )

    def plot_target_projection(
        self,
        target_field: str,
        method: str = "pca",
        class_indices: Optional[Sequence[int]] = None,
        mode: str = "overlay",
        head_name: Optional[str] = None,
        random_seed: int = 0,
        page_size: int = 6,
        **kwargs: Any,
    ):
        """Plot every requested multi-label class in latent projection space.

        :param target_field: Retained multi-label target field.
        :type target_field: str
        :param method: Projection method, ``pca`` or ``tsne``.
        :type method: str
        :param class_indices: Selected classes, or every mapped class.
        :type class_indices: Sequence[int] | None
        :param mode: Combined ``overlay`` or per-class ``panels``.
        :type mode: str
        :param head_name: Optional head providing probability intensities.
        :type head_name: str | None
        :param random_seed: Projection random seed.
        :type random_seed: int
        :return: Figure and axes.
        :rtype: Any
        """
        first = self.owner.prepared_for()
        if target_field not in first.targets:
            raise ValueError(f"Target '{target_field}' was not retained.")
        targets = np.asarray(first.targets[target_field])
        if targets.ndim != 2:
            raise ValueError("Target projection requires a multi-label matrix.")
        mapping = self.owner.models[
            self.owner.default_model_name
        ].dataset.get_class_mappings()[target_field]
        inverse_mapping = {index: label for label, index in mapping.items()}
        selected = list(class_indices or sorted(inverse_mapping))
        labels = {index: inverse_mapping[index] for index in selected}
        target_mask = np.asarray(
            first.target_masks.get(
                target_field,
                np.ones(len(targets), dtype=bool),
            ),
            dtype=bool,
        )

        # Optional head confidence
        ## Each model supplies its own probability intensity while ground truth
        ## and spectrum positions remain shared between comparisons.
        probabilities = None
        if head_name is not None:
            probabilities = {}
            for model_name, prepared in self.owner.iter_prepared():
                runtime = self.owner.models[model_name]
                bound_field = (
                    getattr(runtime.model, "head_specs", {})
                    .get(head_name, {})
                    .get("target_field")
                )
                if bound_field != target_field:
                    raise ValueError(
                        f"Head '{head_name}' is not bound to '{target_field}'."
                    )
                target_type = runtime.dataset.target_specs[target_field]["type"]
                probabilities[model_name] = probabilities_from_logits(
                    prepared.head_outputs[head_name], target_type
                )
        projected = self.project(method, 2, random_seed, **kwargs)
        projections = (
            projected
            if isinstance(projected, Mapping)
            else {self.owner.default_model_name: projected}
        )
        if mode == "panels" and len(selected) > page_size:
            return {
                page: plot_multilabel_projection(
                    projections,
                    targets,
                    target_mask,
                    {
                        index: labels[index]
                        for index in selected[start : start + page_size]
                    },
                    probabilities,
                    mode,
                    method.upper(),
                    self.owner.theme,
                )
                for page, start in enumerate(range(0, len(selected), page_size), 1)
            }
        return plot_multilabel_projection(
            projections,
            targets,
            target_mask,
            labels,
            probabilities,
            mode,
            method.upper(),
            self.owner.theme,
        )

    def overview(
        self,
        projection: str = "pca",
        component_indices: Sequence[int] = range(4),
        labels: Optional[np.ndarray] = None,
    ) -> Mapping[str, Any]:
        """Build the standard latent visualization collection."""
        return {
            "statistics": self.statistics(),
            "projection": self.plot_projection(projection, labels=labels),
            "latent_components": self.plot_components(component_indices, "latent"),
            "pca_components": self.plot_components(component_indices, "pca"),
        }
