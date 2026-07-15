# Heading 1 (MSI AutoEncoder Wrapper Facade)
## Facade interface consolidating workspace operations, contexts registries, and PyTorch models manager

from __future__ import annotations
from typing import Any, Dict, Optional

# Core mixins imports targeting the new modular structures
from .mixins.workspace.workspace_manager_mixin import WorkspaceMixin
from .mixins.context_manager.context_manager_mixin import ContextManagerMixin
from .mixins.active_context.active_context_mixin import ActiveContextMixin
from .mixins.models_manager.models_manager_mixin import ModelsManagerMixin

# Centralized utilities imports
from ..utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


class MSIAutoEncoderWrapper(
    WorkspaceMixin,        # Flat filesystem IO proxy interface
    ContextManagerMixin,   # Multi-image metadata configuration ledger database
    ActiveContextMixin,    # Dynamic transparent routing command proxy for the active target file
    ModelsManagerMixin     # Core PyTorch model builders, datasets, and training loop proxy
):
    """
    Core Facade (Wrapper) orchestrating functional mixins of the MSI AutoEncoder Library.
    Exposes a unified, clean API for downstream pipelines execution.
    """

    def __init__(
        self,
        project_path: str,
        device: str = "cpu",
        auto_create_dirs: bool = True,
        layout: Optional[Dict[str, str]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initializes the cohesive facade wrapper, configuring MRO cooperatively.

        :param project_path: Resolved path to the project root directory.
        :type project_path: str
        :param device: Default hardware training device target ('cpu', 'cuda', 'mps'). Defaults to 'cpu'.
        :type device: str
        :param auto_create_dirs: Toggles automatic directory structure layout creation. Defaults to True.
        :type auto_create_dirs: bool
        :param layout: Custom dictionary layout definition. Defaults to None.
        :type layout: Optional[Dict[str, str]]
        """
        # Pre-initialization state binding
        ## Set executing hardware device reference so cooperative mixins have instant access
        self.device = device
        logger.info("MSIAutoEncoderWrapper: Anchoring processing device state: %s", device)

        # Cooperative MRO initialization chain
        ## Execute mixin constructor chains passing arguments sequentially
        super().__init__(
            project_path=project_path,
            auto_create_dirs=auto_create_dirs,
            layout=layout,
            *args,
            **kwargs
        )

        logger.info("MSIAutoEncoderWrapper facade successfully initialized and bound.")