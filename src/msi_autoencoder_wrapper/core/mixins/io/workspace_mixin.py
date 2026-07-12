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

            #TODO - latent i inne foldery pomocnicze powinny zależec od moedelui, obraz z modelem powinny się zamienićbo logiak jest tak że rozdzielamy modele lokalneod modeli związanych z innymi obrazami - coś takiego - zatanawoc się  czy to ma sens , ta ststurktua też ma plusy bo pozwla ąłdowąc w jakim modelu co było, ale ciężęj wychwywic tą autoencododwanaprzesztrzeń 
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, Tuple
from ....utils.logger import get_custom_logger
from ....utils.exceptions import WorkspaceConfigError

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
# Section: Setters
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Project setters
    # --------------------------------------------------


    # --------------------------------------------------
    # Subsection: Single image setters 
    # --------------------------------------------------

    def set_active_image(self, img_name_or_path: Optional[str] = None) -> None:
        """
        Sets the active image context, resolving raw paths, workspace images, or defaults.

        If no parameter is supplied, it gracefully rolls back to the pre-configured default
        image settings (including its custom external path tracking if present).

        :param img_name_or_path: Name of the target image or its direct file path location.
        :type img_name_or_path: Optional[str]
        """
        # Context evaluation block
        ## Handle empty input arguments by attempting default fallback transitions
        if img_name_or_path is None:
            if getattr(self, "default_img_name", None):
                logger.info("No image name provided. Falling back to default workspace image configuration.")
                self.active_img_name = self.default_img_name
                self._active_img_custom_path = getattr(self, "_default_img_custom_path", None)
            else:
                logger.info("Active image configuration cleared. No default context available.")
                self.active_img_name = None
                self._active_img_custom_path = None
                return
        else:
            ## Resolve provided explicit path argument using central parsing helper
            name, custom_path = self._resolve_incoming_path(img_name_or_path)
            self.active_img_name = name
            self._active_img_custom_path = custom_path
            logger.info("Active image context successfully established: %s", name)

        # Trigger explicit session reset
        ## Force the active reader proxy to wipe cached file handles to synchronize with the new active context
        if hasattr(self, "_wrapper") and hasattr(self._wrapper, "active_context"):
            if hasattr(self._wrapper.active_context, "clear_active_context"):
                self._wrapper.active_context.clear_active_context()

        # Directory provisioning step
        ## Automatically construct required subfolders if workspace is in automated mode
        if getattr(self, "auto_create_dirs", True):
            self.create_required_directories()

    def set_default_image_path(self, img_name_or_path: Optional[str] = None) -> None:
        """
        Sets the default system image configuration context using the unified resolution system.

        :param img_name_or_path: Name of the default image or its direct file path location. Clears if None.
        :type img_name_or_path: Optional[str]
        """
        # Default assignment block
        ## Evaluate input context to determine state assignment or erasure
        if img_name_or_path is None:
            logger.info("Clearing default image configuration context.")
            self.default_img_name = None
            self._default_img_custom_path = None
        else:
            ## Parse incoming path using centralized discrimination helper
            name, custom_path = self._resolve_incoming_path(img_name_or_path)
            self.default_img_name = name
            self._default_img_custom_path = custom_path
            logger.info("Default image context successfully established: %s", name)


    # --------------------------------------------------
    # Subsection: Multiple images setters 
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
    # Subsection: Models
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
# Section: Getters
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Project getters
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
    # Subsection: Single image getters
    # --------------------------------------------------

    def get_active_image_name(self) -> Optional[str]:
        """
        Retrieves the raw string identifier of the currently active dataset image context.

        :return: The active image name string, or None if no image context is currently set.
        :rtype: Optional[str]
        """
        return self.active_img_name

    def get_active_image_file_path(self) -> Optional[Path]:
        """
        Returns the absolute filesystem path to the current active image file (.imzML).

        :return: Absolute Path object targeting the dataset file, or None if unassigned.
        :rtype: Optional[Path]
        """
        # Path generation pipeline
        ## Delegate absolute resolution to the unified building helper
        return self._build_absolute_path(self.active_img_name, self._active_img_custom_path)

    def get_default_image_path(self) -> Optional[Path]:
        """
        Returns the absolute filesystem path to the configured default image file (.imzML).

        :return: Absolute Path object targeting the default dataset file, or None if unassigned.
        :rtype: Optional[Path]
        """
        # Path generation pipeline
        ## Delegate absolute resolution to the unified building helper
        return self._build_absolute_path(self.default_img_name, getattr(self, "_default_img_custom_path", None))
    

    # --------------------------------------------------
    # Subsection: Multiple image getters
    # --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Models
    # --------------------------------------------------

    def get_default_model_path(self) -> Path:
        """
        Calculate default model directory path reference.

        :return: Path to specific model directory.
        :rtype: Path
        :raises WorkspaceConfigError: If active model context is uninitialized.
        """
        if not self.active_model_name:
            raise WorkspaceConfigError("Active model context is uninitialized.")
        return self.get_models_root() / self.active_model_name
    
    def get_active_model_name(self) -> Optional[str]:
        """
        Retrieves the raw string identifier of the currently active machine learning model context.

        :return: The active model name string, or None if no model context is currently set.
        :rtype: Optional[str]
        """
        return self.active_model_name


# --------------------------------------------------
# Section: IO models
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: saving
    # --------------------------------------------------

    def save_training_metrics(
        self, 
        image_key: str, 
        model_type: str, 
        phase_name: str, 
        metrics: dict
    ) -> None:
        """
        Appends epoch training metrics to a phase-specific CSV ledger file inside the workspace structure.

        :param image_key: Unique tracking key token assigned to the targeting image context.
        :type image_key: str
        :param model_type: Identity string defining the master model architecture category.
        :type model_type: str
        :param phase_name: Unique naming descriptor identifying the active optimization phase.
        :type phase_name: str
        :param metrics: Dictionary tracking calculated evaluation scores for the current epoch step.
        :type metrics: dict
        :raises RuntimeError: If file system append operations fail due to structural access restrictions.
        """
        # Directory resolution path block
        ## Resolve absolute directory target paths utilizing project layout paths tracking variables
        try:
            self.set_active_model(model_type)
            self.set_active_image(image_key)
            config_dir = self.get_config_dir()
            history_dir = config_dir / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            logger.error("Failed to resolve metric directory paths via workspace: %s", str(error), exc_info=True)
            raise RuntimeError(f"Workspace metrics directory configuration failure: {error}") from error
        
        csv_path = history_dir / f"{phase_name}_history.csv"
        csv_is_new = not csv_path.exists()
        
        # File transmission stream pass
        ## Execute stream line writer append operations encapsulated inside exception boundaries
        try:
            with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(metrics.keys()))
                if csv_is_new:
                    writer.writeheader()
                writer.writerow(metrics)
        except Exception as error:
            logger.error("Failed to append training phase metrics to the CSV file: %s", csv_path, exc_info=True)
            raise RuntimeError(f"Workspace metrics append failure: {error}") from error

    def save_model_weights(
        self, 
        image_key: str, 
        model_type: str, 
        state_dict: dict
    ) -> None:
        """
        Serializes PyTorch model parameters state dictionaries into a binary checkpoint file within the workspace layout.

        :param image_key: Unique tracking key token assigned to the targeting image context.
        :type image_key: str
        :param model_type: Identity string defining the master model architecture category.
        :type model_type: str
        :param state_dict: PyTorch weights parameters layout dictionary maps state blueprint.
        :type state_dict: dict
        :raises RuntimeError: If weight binary serialization fails due to access violations.
        """
        import torch
        
        # Checkpoint directory paths execution block
        ## Resolve absolute configuration directory path layers
        try:
            self.set_active_model(model_type)
            self.set_active_image(image_key)
            config_dir = self.get_config_dir()
        except Exception as error:
            logger.error("Failed to resolve weights directory paths via workspace: %s", str(error), exc_info=True)
            raise RuntimeError(f"Workspace weights directory configuration failure: {error}") from error
        
        weights_path = config_dir / "weights.pt"
        
        # Serialization dump operation
        ## Directly serialize tensors parameters onto the local storage structures using torch runtime hooks
        try:
            torch.save(state_dict, weights_path)
            logger.info("Model parameter checkpoints weights successfully saved via workspace to path: %s", weights_path)
        except Exception as error:
            logger.error("Failed to serialize model checkpoint weights parameters to disk path: %s", weights_path, exc_info=True)
            raise RuntimeError(f"Workspace parameter checkpoint write failure: {error}") from error

    def save_model_config(
        self, 
        image_key: str, 
        model_type: str, 
        config_dict: dict
    ) -> None:
        """
        Serializes high-level pipeline setup configurations into a structural JSON blueprint file on disk.

        :param image_key: Unique tracking key token assigned to the targeting image context.
        :type image_key: str
        :param model_type: Identity string defining the master model architecture category.
        :type model_type: str
        :param config_dict: Parameter blueprint tracking layouts collected from operational layer buffers.
        :type config_dict: dict
        :raises RuntimeError: If metadata writing operations are rejected by the file system.
        """
        # Paths resolution mapping step
        try:
            self.set_active_model(model_type)
            self.set_active_image(image_key)
            config_dir = self.get_config_dir()
        except Exception as error:
            logger.error("Failed to resolve configuration directory paths via workspace: %s", str(error), exc_info=True)
            raise RuntimeError(f"Workspace configuration directory framework failure: {error}") from error
        
        config_path = config_dir / "config.json"
        
        # JSON dump execution block
        try:
            with open(config_path, mode="w", encoding="utf-8") as json_file:
                json.dump(config_dict, json_file, indent=4, ensure_ascii=False)
            logger.info("Model pipeline structural blueprint metadata successfully written to path: %s", config_path)
        except Exception as error:
            logger.error("Failed to dump structural metadata configurations configuration ledger to path: %s", config_path, exc_info=True)
            raise RuntimeError(f"Workspace config ledger write failure: {error}") from error

    def save_model(
        self, 
        image_key: str, 
        model_type: str, 
        state_dict: dict, 
        config_dict: dict
    ) -> None:
        """
        Triggers synchronized compound serialization dumps for both weight binaries and structural setup Blueprints.

        :param image_key: Unique tracking key token assigned to the targeting image context.
        :type image_key: str
        :param model_type: Identity string defining the master model architecture category.
        :type model_type: str
        :param state_dict: PyTorch weights parameters layout dictionary maps state blueprint.
        :type state_dict: dict
        :param config_dict: Parameter blueprint tracking layouts collected from operational layer buffers.
        :type config_dict: dict
        """
        # Execute individual modular serialization operations tracking sequence checkpoints
        ## 1. Persist layer structure blueprints to disk maps
        self.save_model_config(image_key=image_key, model_type=model_type, config_dict=config_dict)
        
        ## 2. Persist real numerical parameters weights arrays onto disk blocks
        self.save_model_weights(image_key=image_key, model_type=model_type, state_dict=state_dict)
        
        logger.info("Unified model collective states serialization cleanly finalized for image context: %s", image_key)

    # --------------------------------------------------
    # Subsection: loading 
    # --------------------------------------------------

    def load_model_state(self, image_key: str, model_type: str) -> tuple:
        """
        Deserializes and reads structural configurations and binary parameter weight states from target workspace branches.

        :param image_key: Unique tracking key token assigned to the targeting image context.
        :type image_key: str
        :param model_type: Identity string defining the master model architecture category.
        :type model_type: str
        :return: A pair containing the structural layers metadata ledger dictionary and the loaded raw parameters weights map.
        :rtype: tuple(dict, dict)
        :raises FileNotFoundError: If the requested configurations file components are completely missing from the paths lookup.
        :raises RuntimeError: If binary data parse procedures fail.
        """
        import json
        import torch
        
        # File path extraction mapping phase
        try:
            self.set_active_model(model_type)
            self.set_active_image(image_key)
            config_dir = self.get_config_dir()
        except Exception as error:
            logger.error("Failed to resolve configuration recovery directory paths via workspace: %s", str(error), exc_info=True)
            raise RuntimeError(f"Workspace execution recovery configuration failure: {error}") from error
        
        config_path = config_dir / "config.json"
        weights_path = config_dir / "weights.pt"
        
        # Verification checking block
        if not config_path.exists() or not weights_path.exists():
            logger.error("State restoration pass rejected: Incomplete setup footprints found at folder: %s", config_dir)
            raise FileNotFoundError(f"Cannot restore model execution state: Checkpoint target blocks are missing in '{config_dir}'.")
            
        # Unpacking deserialization streaming data blocks
        try:
            ## 1. Load configuration metadata dictionary
            with open(config_path, mode="r", encoding="utf-8") as json_file:
                config_dict = json.load(json_file)
                
            ## 2. Load numerical array state tensors maps using standard torch loader mappings
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            
            logger.info("Structural config parameters map and binary weight maps recovered cleanly from disk nodes.")
            return config_dict, state_dict
            
        except Exception as error:
            logger.error("Critical failure during binary and JSON deserialization loops at folder target: %s", config_dir, exc_info=True)
            raise RuntimeError(f"Workspace state restoration operational failure: {error}") from error

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
# Section: Helpers
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: OS Directory Provisioning Execution
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

    # =====================================================================
    # Subsection: Path Orchestration Helpers
    # =====================================================================

    ## Helper 1: Resolve incoming path or identifier token
    def _resolve_incoming_path(self, img_name_or_path: Optional[Union[str, Path]]) -> Tuple[Optional[str], Optional[Path]]:
        """
        Analyzes an incoming image descriptor to extract its stem name and custom base path context.

        Discriminates between local workspace keys and absolute/external file system paths.
        If the path resides inside the workspace root, it strips the absolute context to maintain local layout integrity.

        :param img_name_or_path: Raw text key, relative path, or absolute file system path.
        :type img_name_or_path: Optional[Union[str, Path]]
        :return: A tuple containing the resolved image stem name and the custom base directory path (if external).
        :rtype: Tuple[Optional[str], Optional[Path]]
        """
        if img_name_or_path is None:
            return None, None

        resolved_path = Path(img_name_or_path)
        img_name = resolved_path.stem
        custom_path: Optional[Path] = None

        # Path discrimination logic
        ## Evaluate if the parameter represents a complex file system location
        if resolved_path.is_absolute() or len(resolved_path.parts) > 1 or resolved_path.exists():
            ### Determine layout association relative to the project root directory
            if hasattr(self, "is_path_in_workspace") and self.is_path_in_workspace(resolved_path):
                logger.debug("Resolved internal path for image token: %s", img_name)
                custom_path = None
            else:
                logger.debug("Resolved external path mapping for image token: %s", img_name)
                custom_path = resolved_path.parent / resolved_path.stem
        else:
            logger.debug("Resolved standalone workspace identifier key: %s", img_name)
            custom_path = None

        return img_name, custom_path

    ## Helper 2: Reconstruct absolute filesystem target path
    def _build_absolute_path(self, img_name: Optional[str], custom_path: Optional[Path], suffix: str = ".imzML") -> Optional[Path]:
        """
        Synthesizes a complete absolute filesystem path for an image using its tracked state parameters.

        If a custom external base path is recorded, it reconstructs the target using that context.
        Otherwise, it generates a standard workspace repository path inside the default 'imgs' directory.

        :param img_name: Clean identifier handle representing the target image stem.
        :type img_name: Optional[str]
        :param custom_path: Optional external path tracker matching the image target.
        :type custom_path: Optional[Path]
        :param suffix: File format extension to append onto the reconstructed layout. Defaults to ".imzML".
        :type suffix: str
        :return: Complete absolute system path pointing to the dataset file, or None if unassigned.
        :rtype: Optional[Path]
        """
        if not img_name:
            return None

        # Absolute path synthesis
        ## Check if the image utilizes an external filesystem tracking reference
        if custom_path:
            logger.debug("Synthesizing absolute external target location for image: %s", img_name)
            return custom_path.with_suffix(suffix)
        
        ## Fall back to standard internal workspace layout generation
        logger.debug("Synthesizing internal repository target location for image: %s", img_name)
        return Path(self.project_path) / "imgs" / f"{img_name}{suffix}"

    ## Helper 3: Clearing active files
    def clear_active_context(self) -> None:
        """
        Resets the temporary working context attributes back to None after method execution.
        """
        self.active_model_name = None
        self.active_img_name = None
        self.active_img_names = None
        self._active_img_custom_path = None
        
        # State signaling block
        ## Notify the coupled active reader proxy to close file handles and release RAM allocations
        if hasattr(self, "_wrapper") and hasattr(self._wrapper, "active_context"):
            if hasattr(self._wrapper.active_context, "clear_active_context"):
                logger.debug("Signaling active reader proxy to clear binary file handles via context discharge.")
                self._wrapper.active_context.clear_active_context()

# --------------------------------------------------
# Section: WorkspaceMixin Injection Hook
# --------------------------------------------------

class WorkspaceMixin:
    """
    Mixin class designed to inject project structural workspace features into the main MSI wrapper.
    """
    def __init__(
            self, 
            project_path: str, 
            auto_create_dirs: bool = True, 
            custom_layout: Optional[Dict[str, str]] = None, 
            *args, **kwargs):
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
        # Context binding hook
        ## Inject the master facade instance reference into the proxy container to enable cross-proxy signaling
        self.workspace._wrapper = self
        super().__init__(*args, **kwargs)