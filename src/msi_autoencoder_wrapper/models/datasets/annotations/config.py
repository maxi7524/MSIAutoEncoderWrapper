"""Validated dataset-level annotation mapping configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ....utils.exceptions import raise_validation_error


@dataclass(frozen=True)
class AnnotationTargetSettings:
    """Define dataset behaviour for one annotation-derived target."""

    empty_spectrum_policy: str = "predictive"
    unobserved_label_policy: str = "negative"
    add_annotation_presence_target: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "AnnotationTargetSettings":
        """Validate one target policy mapping.

        :param config: Optional target policy mapping.
        :type config: Mapping[str, Any] | None
        :return: Immutable target policy.
        :rtype: AnnotationTargetSettings
        """
        values = dict(config or {})
        empty_policy = str(values.get("empty_spectrum_policy", "predictive"))
        if empty_policy not in {"exclude", "reconstruction_only", "predictive"}:
            raise_validation_error(
                "AnnotationSettings",
                "empty_spectrum_policy must be 'exclude', 'reconstruction_only', or 'predictive'.",
            )
        unobserved_policy = str(values.get("unobserved_label_policy", "negative"))
        if unobserved_policy not in {"negative", "unlabelled", "masked"}:
            raise_validation_error(
                "AnnotationSettings",
                "unobserved_label_policy must be 'negative', 'unlabelled', or 'masked'.",
            )
        presence = values.get("add_annotation_presence_target", False)
        if not isinstance(presence, bool):
            raise_validation_error(
                "AnnotationSettings", "add_annotation_presence_target must be Boolean."
            )
        return cls(
            empty_spectrum_policy=empty_policy,
            unobserved_label_policy=unobserved_policy,
            add_annotation_presence_target=presence,
        )

    def get_config(self) -> dict[str, Any]:
        """Return a portable configuration mapping."""
        return {
            "empty_spectrum_policy": self.empty_spectrum_policy,
            "unobserved_label_policy": self.unobserved_label_policy,
            "add_annotation_presence_target": self.add_annotation_presence_target,
        }


@dataclass(frozen=True)
class AnnotationSettings:
    """Configure coordinate mapping, source selection, and target policies."""

    x_mapping: str = "binner"
    annotated_fraction: float = 1.0
    unannotated_fraction: float = 1.0
    max_unannotated_to_annotated_ratio: float | None = None
    seed: int = 0
    targets: Mapping[str, AnnotationTargetSettings] | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "AnnotationSettings":
        """Validate the declarative dataset annotation settings.

        :param config: Optional configuration mapping.
        :type config: Mapping[str, Any] | None
        :return: Immutable annotation settings.
        :rtype: AnnotationSettings
        """
        values = dict(config or {})
        mapping = values.get("mapping", {})
        if not isinstance(mapping, Mapping):
            raise_validation_error("AnnotationSettings", "mapping must be a mapping.")
        x_mapping = str(mapping.get("x_mapping", "binner"))
        if x_mapping not in {"binner", "annotation"}:
            raise_validation_error(
                "AnnotationSettings", "mapping.x_mapping must be 'binner' or 'annotation'."
            )
        selection = values.get("selection", {})
        if not isinstance(selection, Mapping):
            raise_validation_error("AnnotationSettings", "selection must be a mapping.")
        annotated_fraction = _fraction(selection.get("annotated_fraction", 1.0), "annotated_fraction")
        unannotated_fraction = _fraction(
            selection.get("unannotated_fraction", 1.0), "unannotated_fraction"
        )
        ratio = selection.get("max_unannotated_to_annotated_ratio")
        if ratio is not None and (isinstance(ratio, bool) or float(ratio) < 0):
            raise_validation_error(
                "AnnotationSettings",
                "max_unannotated_to_annotated_ratio must be non-negative or null.",
            )
        seed = selection.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise_validation_error("AnnotationSettings", "selection.seed must be an integer.")
        raw_targets = values.get("targets", {})
        if not isinstance(raw_targets, Mapping):
            raise_validation_error("AnnotationSettings", "targets must be a mapping.")
        targets = {
            str(name): AnnotationTargetSettings.from_config(target)
            for name, target in raw_targets.items()
        }
        return cls(
            x_mapping=x_mapping,
            annotated_fraction=annotated_fraction,
            unannotated_fraction=unannotated_fraction,
            max_unannotated_to_annotated_ratio=(None if ratio is None else float(ratio)),
            seed=seed,
            targets=targets,
        )

    def target(self, name: str) -> AnnotationTargetSettings:
        """Return settings for one target, using the neutral default when absent."""
        return dict(self.targets or {}).get(name, AnnotationTargetSettings())

    def get_config(self) -> dict[str, Any]:
        """Return a portable configuration mapping."""
        return {
            "mapping": {"x_mapping": self.x_mapping},
            "selection": {
                "annotated_fraction": self.annotated_fraction,
                "unannotated_fraction": self.unannotated_fraction,
                "max_unannotated_to_annotated_ratio": self.max_unannotated_to_annotated_ratio,
                "seed": self.seed,
            },
            "targets": {
                name: target.get_config() for name, target in dict(self.targets or {}).items()
            },
        }


def _fraction(value: Any, name: str) -> float:
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        raise_validation_error("AnnotationSettings", f"selection.{name} must be in [0, 1].")
    return fraction
