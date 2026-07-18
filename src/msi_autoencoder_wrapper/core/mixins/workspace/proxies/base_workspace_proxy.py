# Heading 1 (Base Workspace Proxy Definition)
## Shared base class managing common directories and context parameters for workspace proxies

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, List
from ...base_wrapper_proxy import BaseWrapperProxy


class BaseWorkspaceProxy(BaseWrapperProxy):
    """
    Common base class for Workspace sub-proxies, sharing state and layout metadata.
    """

    def __init__(
        self, 
        wrapper_ref: Any, 
        workspace_root: Path, 
        layout: Dict[str, str], 
        auto_create_dirs: bool
    ) -> None:
        """
        Initializes the shared workspace proxy environment parameters.

        :param wrapper_ref: Loose reference back to the coordinating master wrapper facade.
        :type wrapper_ref: Any
        :param workspace_root: Absolute resolved path to the project root directory.
        :type workspace_root: Path
        :param layout: Dictionary mapping folder keys to their relative structural names.
        :type layout: Dict[str, str]
        :param auto_create_dirs: Boolean flag controlling automatic filesystem creation.
        :type auto_create_dirs: bool
        """
        # Base initialization
        ## Bind wrapper reference via super constructor
        super().__init__(wrapper_ref=wrapper_ref)
        
        # Workspace state configuration
        self.project_path_resolved: Path = workspace_root
        self._layout: Dict[str, str] = layout
        self.auto_create_dirs: bool = auto_create_dirs

        # Stateful session caches
        self.active_model_name: Optional[str] = None
        self.active_img_name: Optional[str] = None
        self.active_img_names: Optional[List[str]] = None
        self._active_img_custom_path: Optional[Path] = None

        self.default_img_name: Optional[str] = None
        self._default_img_custom_path: Optional[Path] = None