"""
Workspace Management Mixin and Proxy for the MSI Library Facade.
Handles robust path configurations, custom layouts, and automated directory provisioning.

Default Directory Layout Structure:
===================================
project_path/
├── imgs/                          # Raw input image datasets (.imzML / .ibd)
└── models/
    └── <model_name>/              # Primary isolated machine learning model folder
        └── <img_name>/            # Nested subdirectory assigned to a target image context
            ├── config/            # Segregated operational configurations and weights storage
            │   ├── config.json    # Binner, model parameters, and execution blueprints
            │   ├── weights.pt     # Compiled PyTorch graph binary checkpoint weights
            │   └── history.json   # Training performance history trace metric logs
            └── latent/            # Generated spatial compressed latent space target representations
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any, List
from ...utils.logger import get_custom_logger
from ...utils.exceptions import WorkspaceConfigError

logger = get_custom_logger(__name__)


# --------------------------------------------------
# Section: WorkspaceProxy Context Management
# --------------------------------------------------

class WorkspaceProxy:
    """
    Proxy class exposed via WrapperInstance.workspace to manage project directories.
    Provides explicit get_<config> and set_<config> accessors with context resetting.
    """
    def __init__(self, project_path: str, auto_create_dirs: bool = True, custom_layout: Optional[Dict[str, str]] = None):
        """
        Initialize the Workspace configuration state.

        :param project_path: Absolute or relative root directory path for the project.
        :type project_path: str
        :param auto_create_dirs: If True, directory tree is built on context modifications. Defaults to True.
        :type auto_create_dirs: bool
        :param custom_layout: Dictionary containing customized subfolder templates. Defaults to None.
        :type custom_layout: Optional[Dict[str, str]]
        """
        self.project_path = Path(project_path)
        self.auto_create_dirs = auto_create_dirs
        
        # Temporary working context states (Cleared after method execution)
        self.active_model_name: Optional[str] = None
        self.active_img_name: Optional[str] = None
        self.active_img_names: Optional[List[str]] = None
        self._active_img_custom_path: Optional[Path] = None

        # Global persistent fallbacks (Default values - can be standalone tokens or paths)
        self.default_img_name: Optional[str] = None
        self.default_model_name: Optional[str] = None

        default_layout = {
            "imgs_dir": "imgs",
            "models_root": "models",
            "model_config_subdir": "config",
            "model_latent_subdir": "latent"
        }
        if custom_layout:
            default_layout.update(custom_layout)
        self._layout = default_layout

    # --------------------------------------------------
    # Subsection: System Path Resolution & Validation Helpers
    # --------------------------------------------------

    def is_path_in_workspace(self, path: Path) -> bool:
        """
        Checks if the given path is located within the project workspace directory.
        Handles both absolute paths and relative configurations dynamically.

        :param path: Path to inspect.
        :type path: Path
        :return: True if path belongs inside project_path, False otherwise.
        :rtype: bool
        """
        try:
            resolved_project = self.project_path.resolve()
            resolved_path = Path(path).resolve()
            return resolved_project in resolved_path.parents or resolved_path == resolved_project
        except Exception:
            # Fallback in case of strict environment resolution limitations
            try:
                abs_project = self.project_path.absolute()
                abs_path = Path(path).absolute()
                return abs_project in abs_path.parents or abs_path == abs_project
            except Exception:
                return False

    def validate_path(self, path: Path, check_writable: bool = False, should_exist: bool = True) -> bool:
        """
        Validates system paths for existence, accessibility, or write permissions.

        :param path: Path target to check.
        :type path: Path
        :param check_writable: If True, ensures destination has OS write permissions. Defaults to False.
        :type check_writable: bool
        :param should_exist: If True, asserts path availability on disk. Defaults to True.
        :type should_exist: bool
        :return: True if validation succeeds.
        :rtype: bool
        :raises WorkspaceConfigError: If path checks fail structural constraints.
        """
        if should_exist and not path.exists():
            raise WorkspaceConfigError(f"System path validation failed. Destination does not exist: {path}")
            
        if check_writable:
            target_dir = path if path.is_dir() else path.parent
            if target_dir.exists() and not os.access(target_dir, os.W_OK):
                raise WorkspaceConfigError(f"System path validation failed. Target directory is not writable: {target_dir}")
        return True

    def validate_active_context_paths(self, check_weights: bool = False) -> None:
        """
        Validates the correctness and physical presence of critical files for the active context.
        Ensures the raw imagery files exist, and optionally validates compiled model weights.

        :param check_weights: If True, asserts .pt checkpoint presence in model layout. Defaults to False.
        :type check_weights: bool
        """
        if not self.active_img_name:
            raise WorkspaceConfigError("Active validation context blocked: No active image context has been set.")

        # Check raw input (.imzML)
        img_path = self.get_active_image_file_path(extension=".imzML")
        if img_path and not img_path.exists():
            raise WorkspaceConfigError(f"Active image context mapping failed: Missing raw dataset file: {img_path}")

        # Check model requirements
        if self.active_model_name:
            self.validate_path(self.get_active_model_dir(), should_exist=True)
            if check_weights:
                weights_file = self.get_config_dir() / "weights.pt"
                if not weights_file.exists():
                    raise WorkspaceConfigError(f"Execution context fault: Required model checkpoint weights missing: {weights_file}")

    # --------------------------------------------------
    # Subsection: single image setup
    # --------------------------------------------------

    def set_active_image(self, image_name: Optional[str] = None) -> None:
        """
        Sets the active image context, resolving raw paths, workspace images, or defaults.

        :param image_name: Name of the image or direct file path. Falls back to default if None.
        :type image_name: Optional[str]
        """
        if image_name is None:
            if self.default_img_name:
                logger.info("No image name provided. Falling back to default workspace image configuration.")
                image_name = self.default_img_name
            else:
                logger.info("Active image configuration cleared. No default context available.")
                self.active_img_name = None
                self._active_img_custom_path = None
                return

        image_path = Path(image_name)

        ## Discriminate between full disk path triggers and standalone keys
        if image_path.is_absolute() or len(image_path.parts) > 1 or image_path.exists():
            ### Resolve direct raw filesystem path input
            logger.info("Resolving direct filesystem path context for image: %s", str(image_name))
            self.active_img_name = image_path.stem
            
            ### Manage system mapping based on layout association
            if self.is_path_in_workspace(image_path):
                self._active_img_custom_path = None
                logger.info("Image path successfully validated inside active workspace root.")
            else:
                # Save the base path without extensions for external system paths
                self._active_img_custom_path = image_path.parent / image_path.stem
                logger.info("External filesystem path detected. Image context mapped outside of workspace.")
        else:
            ### Resolve relative key mapping inside standard workspace image repository
            logger.info("Mapping relative image handle against workspace repository path: %s", str(image_name))
            self.active_img_name = image_path.stem
            self._active_img_custom_path = None

        if self.auto_create_dirs:
            self.create_required_directories()

    def set_default_image(self, img_name_or_path: str) -> None:
        """
        Establish a global fallback default image key or file path when parameters are omitted.
        Supports both standalone identifiers (within workspace) and full system paths (outside workspace).

        :param img_name_or_path: Unique identifier or absolute/relative path of the fallback image target.
        :type img_name_or_path: str
        """
        self.default_img_name = img_name_or_path


    # --------------------------------------------------
    # Subsection: multiple image setup
    # --------------------------------------------------

    def set_active_images(self, img_names: List[str]) -> None:
        """
        Set the temporary working multi-image context for multi-dataset models.

        :param img_names: List of unique identifiers of the target image files.
        :type img_names: List[str]
        :raises WorkspaceConfigError: If the image names list is empty or invalid.
        """
        if not img_names:
            raise WorkspaceConfigError("Image names list cannot be empty.")
        self.active_img_names = img_names
        self.active_img_name = None  # Clear single-image context to avoid conflicts
        self._active_img_custom_path = None
        if self.auto_create_dirs:
            self.create_required_directories()


    # --------------------------------------------------
    # Subsection: model setup
    # --------------------------------------------------

    def set_active_model(self, model_name: str) -> None:
        """
        Set the temporary working machine learning model context.

        :param model_name: Unique identifier of the model architecture configuration.
        :type model_name: str
        :raises WorkspaceConfigError: If the model name string is empty or invalid.
        """
        if not model_name or not model_name.strip():
            raise WorkspaceConfigError("Model name identifier cannot be empty.")
        self.active_model_name = model_name
        if self.auto_create_dirs:
            self.create_required_directories()

    def set_default_model(self, model_name: str) -> None:
        """
        Establish a global fallback default model key when parameters are omitted.

        :param model_name: Unique identifier of the fallback model architecture.
        :type model_name: str
        """
        self.default_model_name = model_name

    # --------------------------------------------------
    # Subsection: clear context
    # --------------------------------------------------

    def clear_active_context(self) -> None:
        """
        Resets the temporary working context attributes back to None after method execution.
        """
        self.active_model_name = None
        self.active_img_name = None
        self.active_img_names = None
        self._active_img_custom_path = None


# --------------------------------------------------
# Section: Path and Directory Accessors
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Project folder
    # --------------------------------------------------

    def get_project_path(self) -> Path:
        """
        Retrieve absolute root workspace reference directory.

        :return: Path to project root folder.
        :rtype: Path
        """
        return self.project_path

    def get_imgs_dir(self) -> Path:
        """
        Retrieve absolute path locating raw input imaging targets.

        :return: Path to images directory.
        :rtype: Path
        """
        return self.project_path / self._layout["imgs_dir"]

    def get_models_root(self) -> Path:
        """
        Retrieve absolute path to structural base directory housing checkpoints.

        :return: Path to models directory root.
        :rtype: Path
        """
        return self.project_path / self._layout["models_root"]
    
    # --------------------------------------------------
    # Subsection: Default getters
    # --------------------------------------------------

    #TODO - those are not implemented (i got bullshit last time )

    def get_default_image(self) -> Path:
        """
        Calculate default image path reference.

        :return: Path to the default image file inside the images directory.
        :rtype: Path
        :raises WorkspaceConfigError: If default image context is uninitialized.
        """
        if not self.default_img_name:
            raise WorkspaceConfigError("Default image name context is uninitialized.")
        return self.get_imgs_root() / self.default_img_name
    
    def get_default_model(self) -> Path:
        """
        Calculate default model directory path reference.

        :return: Path to specific model directory.
        :rtype: Path
        :raises WorkspaceConfigError: If active model context is uninitialized.
        """
        if not self.active_model_name:
            raise WorkspaceConfigError("Active model context is uninitialized.")
        return self.get_models_root() / self.active_model_name

    # --------------------------------------------------
    # Subsection: Active getters workspace (model + img + latent) 
    # --------------------------------------------------

    def get_active_model_dir(self) -> Path:
        """
        Calculate active model path reference hierarchy.

        :return: Path to specific model directory.
        :rtype: Path
        :raises WorkspaceConfigError: If active model context is uninitialized.
        """
        if not self.active_model_name:
            raise WorkspaceConfigError("Active model context is uninitialized.")
        return self.get_models_root() / self.active_model_name

    def get_active_model_image_dir(self, img_name: Optional[str] = None) -> Path:
        """
        Calculate nested operational structure matching specific model coupled with a dataset target.

        :param img_name: Optional explicit image name to bypass current context tracking. Defaults to None.
        :type img_name: Optional[str]
        :return: Path to image context folder within the active model hierarchy.
        :rtype: Path
        :raises WorkspaceConfigError: If active image context is uninitialized.
        """
        target_img = img_name or self.active_img_name
        if not target_img:
            raise WorkspaceConfigError("Target image context is uninitialized.")
        return self.get_active_model_dir() / target_img

    def get_config_dir(self, img_name: Optional[str] = None) -> Path:
        """
        Locate directory dedicated to structural configuration blueprints and binary weights.

        :param img_name: Optional explicit image name to locate directory. Defaults to None.
        :type img_name: Optional[str]
        :return: Path to configuration folder.
        :rtype: Path
        """
        return self.get_active_model_image_dir(img_name=img_name) / self._layout["model_config_subdir"]

    def get_latent_dir(self, img_name: Optional[str] = None) -> Path:
        """
        Locate directory storing exported spatial compressed representations.

        :param img_name: Optional explicit image name to locate directory. Defaults to None.
        :type img_name: Optional[str]
        :return: Path to latent output folder.
        :rtype: Path
        """
        return self.get_active_model_image_dir(img_name=img_name) / self._layout["model_latent_subdir"]


# --------------------------------------------------
# Section: Active getters imzML paths & names 
# --------------------------------------------------

    def get_active_image_name(self) -> Optional[str]:
        """
        Retrieves the raw string identifier of the currently active dataset image context.

        :return: The active image name string, or None if no image context is currently set.
        :rtype: Optional[str]
        """
        return self.active_img_name

    def get_active_image_file_path(self, extension: str = ".imzML") -> Optional[Path]:
        """
        Calculate the full file path pointing directly to the active raw MSI dataset file.
        Returns None safely if the context has not been explicitly initialized.

        :param extension: Target file format string specifier (e.g., '.imzML' or '.ibd'). Defaults to '.imzML'.
        :type extension: str
        :return: Complete Path target reference mapping to raw input dataset, or None if context is empty.
        :rtype: Optional[Path]
        """
        if not self.active_img_name:
            return None
            
        ext = extension if extension.startswith(".") else f".{extension}"
        
        # If an external filesystem path was provided, use it directly
        if self._active_img_custom_path is not None:
            return self._active_img_custom_path.with_suffix(ext)
            
        return self.get_imgs_dir() / f"{self.active_img_name}{ext}"

    def get_active_model_name(self) -> Optional[str]:
        """
        Retrieves the raw string identifier of the currently active machine learning model context.

        :return: The active model name string, or None if no model context is currently set.
        :rtype: Optional[str]
        """
        return self.active_model_name


# --------------------------------------------------
# Section: OS Directory Provisioning Execution
# --------------------------------------------------

    def create_required_directories(self) -> None:
        """
        Safely instantiates all structural subdirectories required by the active execution context.
        Uses exist_ok=True to protect existing runtime binary artifacts and files from corruption.

        :raises WorkspaceConfigError: If disk provisioning fails due to OS or permission faults.
        """
        try:
            self.get_imgs_dir().mkdir(parents=True, exist_ok=True)
            self.get_models_root().mkdir(parents=True, exist_ok=True)
            
            if self.active_model_name:
                self.get_active_model_dir().mkdir(parents=True, exist_ok=True)
                
                # Provision directory layout for single image context
                if self.active_img_name:
                    self.get_config_dir().mkdir(parents=True, exist_ok=True)
                    self.get_latent_dir().mkdir(parents=True, exist_ok=True)
                
                # Provision directory layout for multi image context
                if self.active_img_names:
                    for name in self.active_img_names:
                        self.get_config_dir(img_name=name).mkdir(parents=True, exist_ok=True)
                        self.get_latent_dir(img_name=name).mkdir(parents=True, exist_ok=True)
                    
        except Exception as e:
            logger.error("OS directory provisioning interrupted: %s", str(e))
            raise WorkspaceConfigError(f"Failed to securely provision workspace directories: {e}")


# --------------------------------------------------
# Section: Structural Visualization Utilities
# --------------------------------------------------

    def __str__(self) -> str:
        """
        Defines the native string representation of the WorkspaceProxy configuration.
        Constructs the current filesystem schema block for terminal logging and print output.
        """
        model_part = self.active_model_name or "<model_name>"
        img_part = self.active_img_name or "<img_name>"
        
        lines = [
            f"{self.project_path}/",
            f"├── {self._layout['imgs_dir']}/",
            f"│   └── {img_part}.imzML / {img_part}.ibd",
            f"└── {self._layout['models_root']}/",
            f"    └── {model_part}/",
            f"        └── {img_part}/",
            f"            ├── {self._layout['model_config_subdir']}/",
            f"            │   ├── config.json",
            f"            │   ├── weights.pt",
            f"            │   └── history.json",
            f"            └── {self._layout['model_latent_subdir']}/"
        ]
        return "\n".join(lines)

    def print_workspace_layout(self) -> None:
        """
        Dynamically prints the project workspace file layout tree structure to stdout based on the current
        active model and image context variables. Useful for verification inside interactive notebooks.
        """
        print(self)

# --------------------------------------------------
# Section: WorkspaceMixin Injection Hook
# --------------------------------------------------

class WorkspaceMixin:
    """
    Mixin class designed to inject project structural workspace features into the main MSI wrapper.
    """
    def __init__(self, project_path: str, auto_create_dirs: bool = True, custom_layout: Optional[Dict[str, str]] = None, *args, **kwargs):
        """
        Initialize workspace engine instance attached directly to the hosting context object framework.

        :param project_path: Absolute or relative root directory path for the project.
        :type project_path: str
        :param auto_create_dirs: If True, directory tree is built on context modifications. Defaults to True.
        :type auto_create_dirs: bool
        :param custom_layout: Dictionary containing customized subfolder templates. Defaults to None.
        :type custom_layout: Optional[Dict[str, str]]
        """
        self.workspace = WorkspaceProxy(
            project_path=project_path,
            auto_create_dirs=auto_create_dirs,
            custom_layout=custom_layout
        )
        super().__init__(*args, **kwargs)