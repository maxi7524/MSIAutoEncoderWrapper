"""Runtime coordination for the single currently loaded model."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from .base_models_manager_proxy import BaseModelsManagerProxy
from ...active_context.autoencoder_context_manager import AutoencoderContextInterface
from .....models.architectures.architectures_manager import ArchitecturesManager
from .....models.model_loader import ModelLoader
from .....utils.exceptions import raise_model_initialization_error, raise_validation_error
from .....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ModelRuntimeProxy(BaseModelsManagerProxy):
    """Own the runtime interface for exactly one currently loaded model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize empty runtime model state."""
        super().__init__(*args, **kwargs)
        self._autoencoder_interface: Optional[AutoencoderContextInterface] = None

    @property
    def autoencoder(self) -> Optional[AutoencoderContextInterface]:
        """Return the autoencoder interface for the currently loaded model.

        :return: Active autoencoder interface or ``None`` for another model family.
        :rtype: Optional[AutoencoderContextInterface]
        """
        return self._autoencoder_interface

    @property
    def model_functionality(self) -> Optional[Any]:
        """Return the family interface for the one currently loaded model.

        :return: Loaded-model functionality or ``None`` for an unsupported family.
        :rtype: Optional[Any]
        """
        return self._autoencoder_interface

    @property
    def loaded_model(self) -> Optional[nn.Module]:
        """Return the raw currently loaded PyTorch model.

        :return: Loaded model object or ``None``.
        :rtype: Optional[torch.nn.Module]
        """
        return getattr(self._wrapper, "active_model", None)

    def attach_model(
        self,
        torch_model: nn.Module,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        trained: bool = False,
        bind_to_local_context: bool = False,
    ) -> nn.Module:
        """Attach a ready model and automatically expose its family interface.

        :param torch_model: Ready PyTorch model instance.
        :type torch_model: torch.nn.Module
        :param model_type: Optional registered family override. Detected when omitted.
        :type model_type: Optional[str]
        :param model_name: Optional loaded-model name.
        :type model_name: Optional[str]
        :param trained: Whether weights are ready for inference.
        :type trained: bool
        :param bind_to_local_context: Preserve this interface in the active image ledger.
        :type bind_to_local_context: bool
        :return: The same attached model instance.
        :rtype: torch.nn.Module
        :raises ModelNotInitializedError: If no registered family matches the model.
        """
        ArchitecturesManager.discover_architectures()
        resolved_type = model_type or self._detect_model_type(torch_model)
        architecture_class = ArchitecturesManager._MODEL_REGISTRY.get(resolved_type or "")
        if architecture_class is None or not isinstance(torch_model, architecture_class):
            raise_model_initialization_error(
                model_name=type(torch_model).__name__,
                message=(
                    f"The model does not match registered architecture family "
                    f"'{resolved_type}'."
                ),
            )

        target_device = getattr(self._wrapper, "device", "cpu")
        target_dtype = getattr(self._wrapper, "dtype", None)
        torch_model.to(device=target_device, dtype=target_dtype)
        self._wrapper.active_model = torch_model
        self._training_transient_cache.clear()
        self.active_model_type = resolved_type
        if model_name is not None:
            self._active_model_name = model_name

        self._autoencoder_interface = None
        if resolved_type == "autoencoder":
            self._autoencoder_interface = AutoencoderContextInterface(
                torch_model=torch_model,
                active_context=self._wrapper.active_context,
                trained=trained,
            )
        if bind_to_local_context:
            self.bind_loaded_model_to_local_context()
        logger.info(
            "Attached loaded model '%s' as family '%s'.",
            self._active_model_name or type(torch_model).__name__,
            resolved_type,
        )
        return torch_model

    def load_model(
        self,
        img_name: str,
        model_name: str,
        strict: bool = True,
        bind_to_local_context: bool = False,
    ) -> nn.Module:
        """Load configuration and weights, reconstruct the model, and attach it.

        Loading the model does not require loading its original image.

        :param img_name: Saved image key or ``global`` storage context.
        :type img_name: str
        :param model_name: Saved model name.
        :type model_name: str
        :param strict: Enforce exact state-dictionary key matching.
        :type strict: bool
        :param bind_to_local_context: Bind the loaded interface to the active image.
        :type bind_to_local_context: bool
        :return: Reconstructed and attached model.
        :rtype: torch.nn.Module
        """
        config, state_dict = self._wrapper.workspace.load_model_artifacts(
            img_name=img_name,
            model_name=model_name,
        )
        model, model_type, configured_name = ModelLoader.build(config)
        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as error:
            raise_model_initialization_error(
                model_name=configured_name or model_name,
                message=f"Saved weights are incompatible with the configured model: {error}",
            )
        return self.attach_model(
            torch_model=model,
            model_type=model_type,
            model_name=configured_name or model_name,
            trained=True,
            bind_to_local_context=bind_to_local_context,
        )

    def bind_loaded_model_to_local_context(self) -> Any:
        """Bind current model functionality to the selected image context.

        The binding stores an independent interface reference in the image
        ledger. Loading another model later replaces only the loaded-model
        context and leaves this local binding available.

        :return: Functionality bound to the active image.
        :rtype: Any
        :raises ValidationError: If no supported model or image context exists.
        """
        functionality = self.model_functionality
        if functionality is None:
            raise_validation_error(
                context_name="ModelsManager",
                message="The loaded model does not expose registered functionality.",
            )
        active_context = self._wrapper.active_context
        image_key = (
            getattr(active_context, "_instantiated_image_key", None)
            or getattr(self._wrapper.workspace, "active_img_name", None)
        )
        ledger = self._wrapper.context_manager.config_ledger
        if not image_key or image_key not in ledger:
            raise_validation_error(
                context_name="ModelsManager",
                message=(
                    "A configured active image context is required before binding "
                    "loaded model functionality locally."
                ),
            )
        ledger[image_key]["model_functionality"] = functionality
        active_context._cached_model_functionality = functionality
        logger.info(
            "Bound loaded model functionality to image context '%s'.",
            image_key,
        )
        return functionality

    def mark_model_trained(self, trained: bool = True) -> None:
        """Update inference readiness for the currently loaded model.

        :param trained: New inference readiness state.
        :type trained: bool
        :raises ValidationError: If no autoencoder is currently attached.
        """
        if self._autoencoder_interface is None:
            raise_validation_error(
                context_name="ModelsManager",
                message="No autoencoder is currently attached.",
            )
        self._autoencoder_interface.is_trained = trained

    def unload_model(self) -> None:
        """Clear only the loaded-model context, leaving image contexts untouched."""
        self._wrapper.active_model = None
        self._wrapper.active_dataset = None
        self._autoencoder_interface = None
        self.active_model_type = None
        self._active_model_name = None
        self._training_transient_cache.clear()

    @staticmethod
    def _detect_model_type(torch_model: nn.Module) -> Optional[str]:
        """Return the registered architecture family matching a model instance."""
        ArchitecturesManager.discover_architectures()
        for model_type, architecture_class in ArchitecturesManager._MODEL_REGISTRY.items():
            if isinstance(torch_model, architecture_class):
                return model_type
        return None
