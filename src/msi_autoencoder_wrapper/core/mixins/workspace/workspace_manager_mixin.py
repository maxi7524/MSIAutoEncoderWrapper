# Heading 1 (Workspace Manager Mixin Initialization Update)
## Ensure parent and root project directory paths exist before validation executes

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from .proxies.getters_and_setters_proxy import GettersAndSettersProxy
from .proxies.helpers_proxy import HelpersProxy
from .proxies.io_models_proxy import IoModelsProxy
from ...utils.validators import validate_workspace_path
from ....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)

# Legacy layout structure definition restored for full backward compatibility
DEFAULT_LAYOUT: Dict[str, str] = {
    "imgs_dir": "imgs",
    "models_root": "models",
    "model_config_subdir": "config",
    "model_latent_subdir": "latent"
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
        ## Resolve layout dictionary parameters using legacy layout keys
        resolved_layout = {**DEFAULT_LAYOUT, **(layout or {})}
        
        ## Automatically create project path folder if flagged and missing
        if auto_create_dirs:
            target_path = Path(project_path).resolve()
            if not target_path.exists():
                logger.info("Auto-creating project root directory: %s", target_path)
                target_path.mkdir(parents=True, exist_ok=True)

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
