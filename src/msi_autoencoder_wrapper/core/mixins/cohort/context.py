"""Immutable runtime values for one cohort of MSI images."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from ....utils.configuration import get_component_config


AutoencoderPolicy = Literal["common", "per_member"]


@dataclass(frozen=True)
class ModelReference:
    """Identify a model folder without loading it into ``ModelsManager``."""

    image_name: Optional[str] = None
    model_name: Optional[str] = None
    path: Optional[str] = None
    fingerprint: Optional[str] = None

    @classmethod
    def parse(cls, value: str | Path | Mapping[str, Any]) -> "ModelReference":
        """Parse ``image/model``, a path, or a portable reference mapping."""
        if isinstance(value, Mapping):
            return cls(
                image_name=value.get("image_name"),
                model_name=value.get("model_name"),
                path=value.get("path"),
                fingerprint=value.get("fingerprint"),
            )
        raw = str(value)
        candidate = Path(raw)
        if not candidate.is_absolute() and len(candidate.parts) == 2:
            return cls(image_name=candidate.parts[0], model_name=candidate.parts[1])
        return cls(path=raw)

    def get_config(self) -> Dict[str, Any]:
        """Return non-empty portable reference fields."""
        return {
            name: value
            for name, value in {
                "image_name": self.image_name,
                "model_name": self.model_name,
                "path": self.path,
                "fingerprint": self.fingerprint,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class CohortMember:
    """Bind one stable image key to its resolved local pipeline."""

    image_key: str
    reader: Any
    binner: Any = None
    annotation_reader: Any = None
    latent_reader: Any = None
    context_config: Mapping[str, Any] = field(default_factory=dict)
    autoencoder_reference: Optional[ModelReference] = None

    def get_reader(self, source: str) -> Any:
        """Return the materialized image or latent reader."""
        if source == "image":
            return self.reader
        if source == "latent" and self.latent_reader is not None:
            return self.latent_reader
        raise ValueError(
            f"Cohort member '{self.image_key}' has no materialized {source} reader."
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the reproducible member definition."""
        result: Dict[str, Any] = {
            "image_key": self.image_key,
            "context": dict(self.context_config),
        }
        if self.autoencoder_reference is not None:
            result["autoencoder"] = self.autoencoder_reference.get_config()
        if self.latent_reader is not None:
            result["latent"] = get_component_config(self.latent_reader)
        return result


@dataclass(frozen=True)
class CohortContext:
    """Represent one named set of independent local image contexts."""

    name: str
    members: Tuple[CohortMember, ...] = ()
    autoencoder_policy: AutoencoderPolicy = "common"
    common_autoencoder: Optional[ModelReference] = None

    @property
    def key(self) -> str:
        """Return the workspace model-context key."""
        return f"cohort_{self.name}"

    def with_members(self, members: Tuple[CohortMember, ...]) -> "CohortContext":
        """Return a replaced context with unique members."""
        keys = [member.image_key for member in members]
        if len(keys) != len(set(keys)):
            raise ValueError("Cohort image keys must be unique.")
        return replace(self, members=members)

    def get_config(self) -> Dict[str, Any]:
        """Return the cohort file and model-snapshot representation."""
        autoencoder: Dict[str, Any] = {"policy": self.autoencoder_policy}
        if self.common_autoencoder is not None:
            autoencoder["model"] = self.common_autoencoder.get_config()
        return {
            "schema_version": 1,
            "name": self.name,
            "key": self.key,
            "members": [member.get_config() for member in self.members],
            "autoencoder": autoencoder,
        }
