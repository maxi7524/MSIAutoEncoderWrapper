"""Public annotation-head analysis group."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import numpy as np

from ....utils.exceptions import raise_validation_error
from .metrics import evaluate_head, per_class_metrics, probabilities_from_logits
from .overviews import plot_class_overviews


class HeadAnalysis:
    """Expose target-bound head metrics and spatial class views."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def evaluate(self, threshold: float = 0.5) -> Mapping[str, Any]:
        """Evaluate every retained head for every analyzed model."""
        model_results: Dict[str, Dict[str, Any]] = {}
        for model_name, prepared in self.owner.iter_prepared():
            runtime = self.owner.models[model_name]
            head_specs = getattr(runtime.model, "head_specs", {})
            target_specs = getattr(runtime.dataset, "target_specs", {})
            model_results[model_name] = {}
            for head_name, logits in prepared.head_outputs.items():
                target_field = head_specs.get(head_name, {}).get("target_field")
                if not target_field or target_field not in prepared.targets:
                    raise_validation_error(
                        "HeadAnalysis",
                        f"Head '{head_name}' has no retained bound target.",
                    )
                model_results[model_name][head_name] = evaluate_head(
                    logits,
                    prepared.targets[target_field],
                    target_specs[target_field]["type"],
                    prepared.target_masks.get(target_field),
                    threshold,
                )
        return (
            model_results
            if self.owner.is_multi
            else model_results[self.owner.default_model_name]
        )

    def class_metrics(
        self,
        head_name: str,
        threshold: float = 0.5,
    ) -> Mapping[str, list[Dict[str, float]]] | list[Dict[str, float]]:
        """Return multi-label metrics separately for every class."""
        results = {}
        for model_name, prepared in self.owner.iter_prepared():
            target_field, target_type = self._target_binding(model_name, head_name)
            if target_type != "multi_label":
                raise_validation_error(
                    "HeadAnalysis", "Per-class maps currently require multi_label targets."
                )
            probabilities = probabilities_from_logits(
                prepared.head_outputs[head_name], target_type
            )
            results[model_name] = per_class_metrics(
                probabilities,
                prepared.targets[target_field],
                threshold,
            )
        return results if self.owner.is_multi else results[self.owner.default_model_name]

    def class_maps(
        self,
        head_name: str,
        class_index: int,
        threshold: float = 0.5,
    ) -> Mapping[str, Mapping[str, Any]]:
        """Return ground truth, probability, and error maps per model."""
        maps: Dict[str, Dict[str, Any]] = {}
        for model_name, prepared in self.owner.iter_prepared():
            target_field, target_type = self._target_binding(model_name, head_name)
            probabilities = probabilities_from_logits(
                prepared.head_outputs[head_name], target_type
            )[:, class_index]
            targets = np.asarray(prepared.targets[target_field])
            truth = (
                targets[:, class_index].astype(bool)
                if targets.ndim > 1
                else targets == class_index
            )
            predicted = probabilities >= threshold
            signed = np.where(truth, probabilities, -probabilities)
            maps[model_name] = {
                "ground_truth": self.owner.map_prepared_rows(
                    prepared, truth.astype(float)
                ).values,
                "probability": self.owner.map_prepared_rows(
                    prepared, probabilities
                ).values,
                "signed_assignment": self.owner.map_prepared_rows(
                    prepared, signed
                ).values,
                "true_positive": self.owner.map_prepared_rows(
                    prepared, (truth & predicted).astype(float)
                ).values,
                "false_positive": self.owner.map_prepared_rows(
                    prepared, (~truth & predicted).astype(float)
                ).values,
                "false_negative": self.owner.map_prepared_rows(
                    prepared, (truth & ~predicted).astype(float)
                ).values,
            }
        return maps

    def probability_image(
        self,
        head_name: str,
        class_index: int,
        model_name: Optional[str] = None,
    ):
        """Map one class probability for one model or every model."""
        names = [model_name] if model_name else list(self.owner.model_names)
        images = {}
        for name in names:
            prepared = self.owner.prepared_for(name)
            _, target_type = self._target_binding(name, head_name)
            probabilities = probabilities_from_logits(
                prepared.head_outputs[head_name], target_type
            )[:, class_index]
            images[name] = self.owner.map_prepared_rows(prepared, probabilities)
        return images[names[0]] if model_name or not self.owner.is_multi else images

    def plot_class_overview(
        self,
        head_name: str,
        class_indices: Sequence[int],
        threshold: float = 0.5,
    ) -> Mapping[int, Any]:
        """Plot ground truth, probability, and signed assignment per class."""
        first_runtime = self.owner.models[self.owner.default_model_name]
        target_field, _ = self._target_binding(self.owner.default_model_name, head_name)
        mapping = first_runtime.dataset.get_class_mappings()[target_field]
        labels = {index: label for label, index in mapping.items()}
        maps = {
            int(class_index): self.class_maps(head_name, int(class_index), threshold)
            for class_index in class_indices
        }
        return plot_class_overviews(
            maps,
            labels,
            class_indices,
            self.owner.theme,
        )

    def overview(
        self,
        head_name: str,
        class_indices: Sequence[int],
        threshold: float = 0.5,
    ) -> Mapping[str, Any]:
        """Build standard head metrics and class visualization collection."""
        return {
            "metrics": self.evaluate(threshold),
            "class_metrics": self.class_metrics(head_name, threshold),
            "class_views": self.plot_class_overview(
                head_name, class_indices, threshold
            ),
        }

    def _target_binding(self, model_name: str, head_name: str) -> tuple[str, str]:
        runtime = self.owner.models[model_name]
        target_field = getattr(runtime.model, "head_specs", {}).get(
            head_name, {}
        ).get("target_field")
        target_spec = getattr(runtime.dataset, "target_specs", {}).get(
            target_field, {}
        )
        if not target_field or "type" not in target_spec:
            raise_validation_error(
                "HeadAnalysis", f"Head '{head_name}' has no valid target binding."
            )
        return target_field, target_spec["type"]
