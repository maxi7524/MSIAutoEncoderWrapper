"""
Workspace Management Mixin and Proxy for the MSI Library Facade.
Handles robust path configurations, custom layouts, and automated directory provisioning.
"""

from pathlib import Path
from typing import Dict, Optional, Any, List
from ...utils.logger import get_logger
from ..utils.exceptions import WorkspaceConfigError

logger = get_logger(__name__)


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

        # Global persistent fallbacks (Default values)
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

    def set_active_model(self, model_name: str) -> None:
        """
        Set the temporary working machine learning model context.

        :param model_name: Unique identifier of the model architecture configuration.
        :type model_name: str
        :return: None
        :rtype: None
        :raises WorkspaceConfigError: If the model name string is empty or invalid.
        """
        if not model_name or not model_name.strip():
            raise WorkspaceConfigError("Model name identifier cannot be empty.")
        self.active_model_name = model_name
        if self.auto_create_dirs:
            self.create_required_directories()

    def set_active_image(self, img_name: str) -> None:
        """
        Set the temporary working single mass spectrometry image context.

        :param img_name: Unique identifier of the image file target.
        :type img_name: str
        :return: None
        :rtype: None
        :raises WorkspaceConfigError: If the image name string is empty or invalid.
        """
        if not img_name or not img_name.strip():
            raise WorkspaceConfigError("Image name identifier cannot be empty.")
        self.active_img_name = img_name
        self.active_img_names = None  # Clear multi-image context to avoid conflicts
        if self.auto_create_dirs:
            self.create_required_directories()

    def set_active_images(self, img_names: List[str]) -> None:
        """
        Set the temporary working multi-image context for multi-dataset models.

        :param img_names: List of unique identifiers of the target image files.
        :type img_names: List[str]
        :return: None
        :rtype: None
        :raises WorkspaceConfigError: If the image names list is empty or invalid.
        """
        if not img_names:
            raise WorkspaceConfigError("Image names list cannot be empty.")
        self.active_img_names = img_names
        self.active_img_name = None  # Clear single-image context to avoid conflicts
        if self.auto_create_dirs:
            self.create_required_directories()

    def set_default_image(self, img_name: str) -> None:
        """
        Establish a global fallback default image key when parameters are omitted.

        :param img_name: Unique identifier of the fallback image target.
        :type img_name: str
        :return: None
        :rtype: None
        """
        self.default_img_name = img_name

    def set_default_model(self, model_name: str) -> None:
        """
        Establish a global fallback default model key when parameters are omitted.

        :param model_name: Unique identifier of the fallback model architecture.
        :type model_name: str
        :return: None
        :rtype: None
        """
        self.default_model_name = model_name

    def clear_active_context(self) -> None:
        """
        Resets the temporary working context attributes back to None after method execution.

        :return: None
        :rtype: None
        """
        self.active_model_name = None
        self.active_img_name = None
        self.active_img_names = None

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

    def get_active_image_dir(self, img_name: Optional[str] = None) -> Path:
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
        return self.get_active_image_dir(img_name=img_name) / self._layout["model_config_subdir"]

    def get_latent_dir(self, img_name: Optional[str] = None) -> Path:
        """
        Locate directory storing exported spatial compressed representations.

        :param img_name: Optional explicit image name to locate directory. Defaults to None.
        :type img_name: Optional[str]
        :return: Path to latent output folder.
        :rtype: Path
        """
        return self.get_active_image_dir(img_name=img_name) / self._layout["model_latent_subdir"]

    def create_required_directories(self) -> None:
        """
        Safely instantiates all structural subdirectories required by the active execution context.
        Uses exist_ok=True to protect existing runtime binary artifacts and files from corruption.

        :return: None
        :rtype: None
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