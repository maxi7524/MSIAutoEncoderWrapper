# Heading 1 (MSI AutoEncoder Wrapper Facade)
## Facade interface consolidating workspace operations, contexts registries, and PyTorch models manager

from __future__ import annotations
from typing import Any, Dict, Optional

# Core mixins imports targeting the new modular structures
from .mixins.workspace.workspace_manager_mixin import WorkspaceMixin
from .mixins.context_manager.context_manager_mixin import ContextManagerMixin
from .mixins.active_context.active_context_mixin import ActiveContextMixin
from .mixins.models_manager.models_manager_mixin import ModelsManagerMixin
from .mixins.spatial_context_mixin import SpatialContextMixin, CoordinateOrder

# external libraries
import torch

# Centralized utilities imports
from ..utils.logger import get_custom_logger
from ..configuration import ConfigurationOrchestrator

# Logger initialization
logger = get_custom_logger(__name__)


class MSIAutoEncoderWrapper(
    SpatialContextMixin,  # Global XY versus matrix row-column convention
    WorkspaceMixin,  # Flat filesystem IO proxy interface
    ContextManagerMixin,  # Multi-image metadata configuration ledger database
    ActiveContextMixin,  # Dynamic transparent routing command proxy for the active target file
    ModelsManagerMixin,  # Core PyTorch model builders, datasets, and training loop proxy
):
    """
    Core Facade (Wrapper) orchestrating functional mixins of the MSI AutoEncoder Library.
    Exposes a unified, clean API for downstream pipelines execution.
    """

    def __init__(
        self,
        project_path: str,
        device: str = None,
        auto_create_dirs: bool = True,
        layout: Optional[Dict[str, str]] = None,
        coordinate_order: CoordinateOrder = "xy",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the cohesive facade wrapper, configuring MRO cooperatively.

        :param project_path: Resolved path to the project root directory.
        :type project_path: str
        :param device: Default hardware training device target ('cpu', 'cuda', 'mps'). Defaults to 'None', which sets local available.
        :type device: str
        :param auto_create_dirs: Toggles automatic directory structure layout creation. Defaults to True.
        :type auto_create_dirs: bool
        :param layout: Custom dictionary layout definition. Defaults to None.
        :type layout: Optional[Dict[str, str]]
        :param coordinate_order: Spatial API convention, ``xy`` or ``matrix``.
        :type coordinate_order: Literal["xy", "matrix"]
        """
        # Pre-initialization state binding
        ## Set executing hardware device reference so cooperative mixins have instant access
        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device

        logger.info(
            "MSIAutoEncoderWrapper: Anchoring processing device state: %s", device
        )

        # Cooperative MRO initialization chain
        ## Execute mixin constructor chains passing arguments sequentially
        super().__init__(
            project_path=project_path,
            auto_create_dirs=auto_create_dirs,
            layout=layout,
            coordinate_order=coordinate_order,
            *args,
            **kwargs,
        )

        logger.info("MSIAutoEncoderWrapper facade successfully initialized and bound.")

    def load_configuration(
        self,
        model_name: str,
        image_name: Optional[str] = None,
        apply: bool = True,
        load_model: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load and optionally apply one saved experiment configuration.

        The default operation restores the image pipeline, annotation reader,
        dataset, model architecture, and model weights, then returns the source
        configuration dictionary.

        :param model_name: Saved model identifier.
        :type model_name: str
        :param image_name: Optional image context override. Uses the workspace
            default image when omitted.
        :type image_name: str | None
        :param apply: Apply runtime configuration when true.
        :type apply: bool
        :param load_model: Load model weights when applying runtime configuration.
        :type load_model: bool
        :param strict: Enforce exact model weight compatibility.
        :type strict: bool
        :return: Loaded consolidated configuration dictionary.
        :rtype: Dict[str, Any]
        """
        return ConfigurationOrchestrator(self).load(
            model_name=model_name,
            image_name=image_name,
            apply=apply,
            load_model=load_model,
            strict=strict,
        )
