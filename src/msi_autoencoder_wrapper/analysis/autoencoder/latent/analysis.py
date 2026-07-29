"""Public latent-analysis group attached to an analyzer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
from sklearn.decomposition import PCA

from .metrics import latent_statistics, project_latents
from .overviews import plot_component_grid
from .views import plot_projection_grid


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
        return projections if self.owner.is_multi else projections[self.owner.default_model_name]

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
    ):
        """Plot raw latent or independently fitted PCA spatial components."""
        components = list(component_indices)
        images = {}
        for model_name in self.owner.model_names:
            images[model_name] = {}
            for component in components:
                image = (
                    self.component_image(component, model_name)
                    if source == "latent"
                    else self.pca_component_image(component, model_name)
                )
                images[model_name][component] = image.values
        return plot_component_grid(
            images,
            components,
            source.upper(),
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
