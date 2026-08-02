"""Reconstruct configured model architectures from portable JSON documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json
import hashlib
import torch
import torch.nn as nn

from .architectures.architectures_manager import ArchitecturesManager
from ..utils.exceptions import raise_model_initialization_error
from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ModelLoader:
    """Build model instances from the loaded-model section of saved configuration."""

    @classmethod
    def build(cls, config: Dict[str, Any]) -> tuple[nn.Module, str, str | None]:
        """Build an uninitialized-weight model from a portable configuration.

        :param config: Complete saved configuration or loaded-model subsection.
        :type config: Dict[str, Any]
        :return: Model instance, model family, and optional model name.
        :rtype: tuple[torch.nn.Module, str, Optional[str]]
        :raises ModelNotInitializedError: If the architecture configuration is incomplete.
        """
        model_config = config.get("model")
        if not isinstance(model_config, dict):
            raise_model_initialization_error(
                model_name="LoadedModel",
                message="Saved configuration does not contain a loaded-model definition.",
            )

        model_type = model_config.get("type")
        if not isinstance(model_type, str) or not model_type:
            raise_model_initialization_error(
                model_name="LoadedModel",
                message="Saved model family is missing.",
            )

        component_config = model_config.get("components") or cls._runtime_components(
            model_config
        )
        if not isinstance(component_config, dict) or not component_config:
            raise_model_initialization_error(
                model_name=model_type,
                message="Saved model does not contain reconstructable components.",
            )

        ArchitecturesManager.discover_architectures()
        components_setup = {
            category: cls._component_setup(descriptor)
            for category, descriptor in component_config.items()
        }
        model_parameters = model_config.get("parameters", {})
        if not isinstance(model_parameters, dict):
            raise_model_initialization_error(
                model_name=model_type,
                message="Saved model parameters must be a dictionary.",
            )

        logger.info("Reconstructing loaded model family '%s'.", model_type)
        model = ArchitecturesManager.build_model(
            model_type=model_type,
            components_setup=components_setup,
            **model_parameters,
        )
        return model, model_type, model_config.get("name")

    @classmethod
    def load_artifact(
        cls,
        reference: Path | str,
        *,
        workspace_root: Optional[Path | str] = None,
        strict: bool = True,
    ) -> tuple[nn.Module, Dict[str, Any], Path]:
        """Load a model folder without attaching it to wrapper runtime state."""
        model_dir = cls.resolve_artifact_dir(reference, workspace_root=workspace_root)
        config_path = model_dir / "config" / "config.json"
        weights_path = model_dir / "config" / "weights.pt"
        if not config_path.is_file() or not weights_path.is_file():
            raise_model_initialization_error(
                model_name=model_dir.name,
                message="Model artifact requires config/config.json and config/weights.pt.",
            )
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        model, _, _ = cls.build(config)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=strict)
        return model, config, model_dir

    @staticmethod
    def resolve_artifact_dir(
        reference: Path | str, *, workspace_root: Optional[Path | str] = None
    ) -> Path:
        """Resolve workspace ``image/model`` first and current-directory paths second."""
        raw = Path(reference)
        candidates = []
        if workspace_root is not None:
            candidates.append(Path(workspace_root) / "models" / raw)
        candidates.extend((raw, Path.cwd() / raw))
        model_dir = next((path.resolve() for path in candidates if path.is_dir()), None)
        if model_dir is None:
            raise_model_initialization_error(
                model_name=str(reference), message="Model artifact folder does not exist."
            )
        return model_dir

    @staticmethod
    def artifact_fingerprint(model_dir: Path | str) -> str:
        """Hash the exact saved configuration and weight bytes."""
        directory = Path(model_dir)
        digest = hashlib.sha256()
        for path in (
            directory / "config" / "config.json",
            directory / "config" / "weights.pt",
        ):
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _runtime_components(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract component descriptors from the runtime fallback configuration."""
        runtime = model_config.get("runtime", {})
        runtime_parameters = runtime.get("parameters", {}) if isinstance(runtime, dict) else {}
        components = runtime_parameters.get("components", {})
        return components if isinstance(components, dict) else {}

    @classmethod
    def _component_setup(cls, descriptor: Any) -> Dict[str, Any]:
        """Convert a portable descriptor into an architecture-manager setup node."""
        if not isinstance(descriptor, dict):
            raise_model_initialization_error(
                model_name="LoadedModel",
                message="A saved component descriptor must be a dictionary.",
            )
        if "type" in descriptor:
            parameters = descriptor.get("parameters", {})
            if not isinstance(parameters, dict):
                raise_model_initialization_error(
                    model_name=str(descriptor.get("type")),
                    message="Saved component parameters must be a dictionary.",
                )
            return {"strategy": descriptor["type"], "params": parameters}
        return {
            child_name: cls._component_setup(child_descriptor)
            for child_name, child_descriptor in descriptor.items()
        }
