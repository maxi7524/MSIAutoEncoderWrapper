# Heading 1 (Workspace Manager Mixin with Clean Multiple Inheritance)
## Flatten the API dynamically via cooperative MRO inheritance trees

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from .proxies.getters_and_setters_proxy import GettersAndSettersProxy
from .proxies.helpers_proxy import HelpersProxy
from .proxies.io_models_proxy import IoModelsProxy
from ...utils.validators import validate_workspace_path

# Default directory structure definition
DEFAULT_LAYOUT: Dict[str, str] = {
    "imgs": "imgs",
    "models": "models",
    "config": "config",
    "latent": "latent"
}


class WorkspaceProxy(GettersAndSettersProxy, HelpersProxy, IoModelsProxy):
    """
    Unified workspace controller. Aggregates all proxy behaviors through 
    multiple inheritance, exposing a clean, zero-boilerplate flat API.
    """

    def __init__(
        self, 
        wrapper_ref: Any, 
        workspace_root: Path, 
        layout: Dict[str, str], 
        auto_create_dirs: bool = True
    ) -> None:
        """
        Initializes all inherited proxy domains using cooperative inheritance.
        """
        # Execute cooperative inheritance chain initialization
        ## Arguments are consumed by BaseWorkspaceProxy at the end of the MRO path
        super().__init__(
            wrapper_ref=wrapper_ref,
            workspace_root=workspace_root,
            layout=layout,
            auto_create_dirs=auto_create_dirs
        )


class WorkspaceMixin:
    """
    Mixin class injected into the MSIAutoEncoderWrapper to expose flat Workspace API.
    """

    def __init__(
        self, 
        project_path: str, 
        auto_create_dirs: bool = True, 
        layout: Optional[Dict[str, str]] = None,
        *args: Any, 
        **kwargs: Any
    ) -> None:
        # Configuration setup
        ## Resolve layout dictionary parameters
        resolved_layout = layout if layout is not None else DEFAULT_LAYOUT
        
        ## Validate project path and convert to resolved Path object
        resolved_path = validate_workspace_path(project_path)
        
        # Proxy assignment
        ## Instantiate the workspace proxy which inherits all interfaces
        self.workspace = WorkspaceProxy(
            wrapper_ref=self, 
            workspace_root=resolved_path, 
            layout=resolved_layout, 
            auto_create_dirs=auto_create_dirs
        )
        
        # Continue MRO chain initialization
        super().__init__(*args, **kwargs)