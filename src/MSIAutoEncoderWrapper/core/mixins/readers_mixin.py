"""
Mixin module extending the main MSI wrapper with safe dataset reader and binning instance registration features.
"""

from typing import Any, Optional
from pathlib import Path
from ...utils.logger import get_logger
from ...readers.manager import ReaderManager as ReaderManager
from ...binners.manager import BinningManager
from ..utils.validators import resolve_component

logger = get_logger(__name__)


class ReadersMixin:
    """
    Mixin class designed to provide image data parsing setups and dynamic alignment mappings for active binners.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the abstract component reference handles.
        """
        self.reader: Optional[Any] = None
        self.binner: Optional[Any] = None
        self.inverse_binner: Optional[Any] = None
        super().__init__(*args, **kwargs)

    def set_reader(self, target: Any, img_name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Loads a mass spectrometry image reader strategy into the execution context session.

        If target is a string name matching a driver registry key, it resolves via path lookups.
        If img_name is None, it falls back onto the workspace's active image definition state.

        :param target: Implemented concrete reader instance or string registry lookup identifier key.
        :type target: Any
        :param img_name: Optional target image file tracker reference name key, defaults to None.
        :type img_name: Optional[str]
        :raises ProjectConfigError: If string lookup values fail registration matches.
        """
        active_img = img_name if img_name else getattr(self.workspace, "active_img_name", None)
        
        # Resolve target storage location if parameter data paths are missing from initialization keys
        if "file_path" not in kwargs:
            if not active_img:
                from ..utils.exceptions import ProjectConfigError
                raise ProjectConfigError("Target configuration aborted: No active image identifier bound in workspace context.")
            
            # Request file paths directly from the workspace layer mixin interfaces
            kwargs["file_path"] = str(self.workspace.get_raw_img_path(active_img))

        # Instantiate or map concrete strategy driver execution engine hooks
        self.reader = resolve_component(
            target=target,
            registry=ReaderManager.REGISTRY,
            component_type="Reader",
            **kwargs
        )
        
        # Synchronize linked state mappings forward onto the updated tracking context node
        if hasattr(self.reader, "attach_binners"):
            self.reader.attach_binners(binner=self.binner, inverse_binner=self.inverse_binner)
            
        logger.info(f"MSI Reader successfully assigned and mounted to active workflow pipeline targets.")

    def set_binner(self, target: Any, **kwargs: Any) -> None:
        """
        Configures the primary forward mass spectrometry profiling spectrum compression binning engine.

        :param target: Concrete binning strategy object or string registration identifier key matching registries.
        :type target: Any
        """
        # Dynamically evaluate the target registration maps
        self.binner = resolve_component(
            target=target,
            registry=BinningManager.REGISTRY,
            component_type="Binner",
            **kwargs
        )
        
        # Synchronize linked state handles backward onto the active image context reader object
        if self.reader and hasattr(self.reader, "attach_binners"):
            self.reader.attach_binners(binner=self.binner)
            
        logger.info("Primary compression forward spectrum binner registered into execution context successfully.")

    def set_inverse_binner(self, target: Any, **kwargs: Any) -> None:
        """
        Configures the utility spatial reverse mapping reconstruction un-binning strategy.

        :param target: Concrete reconstruction strategy instance or lookup token key string mapping.
        :type target: Any
        """
        # Resolve reference definitions tracking registries
        self.inverse_binner = resolve_component(
            target=target,
            registry=BinningManager.INVERSE_REGISTRY,
            component_type="InverseBinner",
            **kwargs
        )
        
        # Mount the utility reference handles onto the functional parsing reader engine bounds
        if self.reader and hasattr(self.reader, "attach_binners"):
            self.reader.attach_binners(inverse_binner=self.inverse_binner)
            
        logger.info("Spatial reconstruction reverse tracking un-binner mounted inside project session successfully.")