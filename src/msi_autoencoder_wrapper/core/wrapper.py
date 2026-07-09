"""
Core Fasade (Wrapper) module orchestrating the subcomponents of the MSI Library.
Provides a clean, high-level API for end-users to manage Mass Spectrometry Imaging pipelines.
"""

## Base 
import os
import json
import torch
from typing import Union, Dict, Any, Optional, List
from ..utils.logger import get_custom_logger

## mixins 
from .mixins.workspace_mixin import WorkspaceMixin
from .mixins.context_manager_mixin import ContextManagerMixin
from .mixins.active_context_mixin import ActiveContextMixin

## exceptions handling 
from ..utils.validators import validate_components
from ..utils.exceptions import (
    ProjectConfigError, 
    ModelNotInitializedError, 
    IncompatibleInterfaceError
)

## Other modules 
from ..readers.readers_manager import ReaderManager
from ..binners.binners_manager import BinnerManager
from ..models.architecture.architecture_manager import ArchitectureManager
from ..models.datasets.dataset_manager import DatasetManager
from ..training.manager import TrainingManager

logger = get_custom_logger(__name__)


class MSIAutoEncoderWrapper(
    ContextManagerMixin,  # Configuration and multi-image registries ledger state database
    ActiveContextMixin,    # Dynamic transparent routing command proxy for the active target file
    # Model setup mixin,        # mixin repsonsible for model instantiation
    # TrainingMixin,        # training module
    # InferenceMixin,       # inference module
    WorkspaceMixin        # Folder automation - `core/mixins/workspace_mixin` module
    ):
    """
    Main Facade class that integrates data loading, binning, modeling, and training into a single workspace session.
    
    This wrapper abstracts underlying manager complexities and automates structural directory setup,
    state tracking, configuration persistence, and checkpoint loading.

    Attributes:
    #TODO(documentation) - after library restructurization write all necessary infromation here
        project_path (str): Absolute path to the active project working directory.
        device (str): PyTorch compute device ('cuda', 'cpu', etc.).
        dirs (Dict[str, str]): Evaluated absolute paths for project subdirectories.
        config (Dict[str, Any]): Runtime parameters, tracking data components and architectural setups.
        model (Optional[torch.nn.Module]): The underlying PyTorch model compiled by the ArchitectureManager.
        history_tracker (Optional[Dict[str, Any]]): Training phase loss metrics history.
    """

    def __init__(
        self, 
        project_path: str, 
        auto_create_dirs: bool = True, 
        custom_layout: Optional[Dict[str, str]] = None,
        device: str = None,
        *args: Any,
        **kwargs: Any
    ):
        """
        Initialize the high-level MSI pipeline orchestration engine.

        :param project_path: Absolute or relative root reference directory path.
        :type project_path: str
        :param auto_create_dirs: If True, automatically creates folders on context initialization. Defaults to True.
        :type auto_create_dirs: bool
        :param custom_layout: Optional layout override templates dictionary. Defaults to None.
        :type custom_layout: Optional[Dict[str, str]]
        :param device: Hardware execution framework target ('cpu', 'cuda'). Defaults to "cpu".
        :type device: str
        """
        # set workspace and memory 
        ## Initialize the WorkspaceMixin layout engine
        super().__init__(
            project_path=project_path, 
            auto_create_dirs=auto_create_dirs, 
            custom_layout=custom_layout,
            *args,
            **kwargs
        )

        ## Core runtime component containers (All loaded objects stay alive in memory dictionaries)
        self.loaders: Dict[str, Any] = {}
        self.binners: Dict[str, Any] = {}
        
        # Session internal states
        ## set device
        self.device: str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        ## Other states
        self.model: Optional[torch.nn.Module] = None
        self.history_tracker: Optional[Dict[str, Any]] = None
        self.config: Dict[str, Any] = {}
        
        logger.info("MSI AutoEncoder Facade workspace initialized on device: %s", device)

    # DEPRACATED 
    # def define_model(
    #     self, 
    #     encoder_config: Dict[str, Any], 
    #     decoder_config: Dict[str, Any], 
    #     projector_config: Optional[Dict[str, Any]] = None
    # ) -> torch.nn.Module:
    #     """
    #     Compiles the full autoencoder architecture using textual configuration specifications.

    #     Args:
    #         encoder_config (Dict[str, Any]): Structural hyperparameters for the Encoder module.
    #         decoder_config (Dict[str, Any]): Structural hyperparameters for the Decoder module.
    #         projector_config (Optional[Dict[str, Any]]): Optional configuration for contrastive projector heads.

    #     Returns:
    #         torch.nn.Module: Compiled PyTorch model deployed to the configured device.
    #     """
    #     logger.info("Requesting structural model compilation from ArchitectureManager...")
        
    #     # Cache configuration schema within the runtime state
    #     self.config["model_spec"] = {
    #         "encoder": encoder_config,
    #         "decoder": decoder_config,
    #         "projector": projector_config
    #     }

    #     # Delegate execution down to the specialized architecture subsystem
    #     self.model = ArchitectureManager.compile_architecture(
    #         encoder_cfg=encoder_config,
    #         decoder_cfg=decoder_config,
    #         projector_cfg=projector_config
    #     )
    #     self.model.to(self.device)
        
    #     logger.info("Model compiled successfully and mounted onto device: %s", self.device)
    #     return self.model

    def fit(self, model_name: Optional[str] = None, img_name_or_path: Optional[Union[str, List[str]]] = None) -> None:
        """
        Executes low-level optimization loops. Resolves target contexts from defaults if parameters are omitted.

        :param model_name: Unique name of the working target model. If None, uses global default fallback.
        :type model_name: Optional[str]
        :param img_name: Single image key or list of image keys. If None, uses global default fallback.
        :type img_name: Optional[Union[str, List[str]]]
        :return: None
        :rtype: None
        :raises ValidationError: If resolved components or mandatory files do not exist.
        """
        # 1. Fallback evaluation logic from global defaults if inputs are None
        resolved_model = model_name or self.workspace.default_model_name
        resolved_img = img_name_or_path or self.workspace.default_img_name

        # 2. Collect targets for batch atomic validation check before execution block
        validate_components([
            (resolved_model, "model_name"),
            (resolved_img, "img_name")
        ])

        try:
            # 3. Mount temporary runtime environment context
            self.workspace.set_active_model(resolved_model)
            if isinstance(resolved_img, list):
                self.workspace.set_active_images(resolved_img)
            else:
                self.workspace.set_active_image(resolved_img)

            logger.info("Starting training pipeline for model '%s' using image context: %s", resolved_model, resolved_img)
            
            # TODO: Integrate with TrainingManager using the computed paths:
            # config_path = self.workspace.get_config_dir()
            
        finally:
            # 4. Critical Context Reset: Always wipe working variables to None, preserving defaults
            self.workspace.clear_active_context()
            logger.info("Temporary fit context flushed. Workspace active contexts returned to None.")

    # DEPRACATED 
    # def save(self, model_name: str) -> str:
    #     """
    #     Saves the complete configuration profile and structural model checkpoints separately.

    #     Ensures clean detachment of binary state_dicts (.pt) from textual metadata manifests (.json).

    #     Args:
    #         model_name (str): Identifier name to bind to file exports.

    #     Raises:
    #         ModelNotInitializedError: If the internal model instance is missing.

    #     Returns:
    #         str: Absolute file path location of the exported configuration manifest.
    #     """
    #     if self.model is None:
    #         raise ModelNotInitializedError("Persistence failed: No initialized model structure found to save.")

    #     model_io_path = os.path.join(self.dirs["models"], f"{model_name}.pt")
    #     config_io_path = os.path.join(self.dirs["configs"], f"{model_name}.json")

    #     # 1. Isolate and save binary tensors
    #     torch.save(self.model.state_dict(), model_io_path)
    #     logger.info("PyTorch weights exported successfully to: %s", model_io_path)

    #     # 2. Persist comprehensive metadata registry blueprint
    #     with open(config_io_path, "w", encoding="utf-8") as f:
    #         json.dump(self.config, f, indent=4, ensure_ascii=False)
    #     logger.info("Session configuration manifest saved to: %s", config_io_path)
        
    #     return config_io_path

    # def load(self, model_name: str) -> None:
    #     """
    #     Restores a historical project session environment dynamically from filesystem payloads.

    #     Reads the structural JSON blueprint to re-compile the neural network dynamically, 
    #     then maps the saved weights into the graph.

    #     Args:
    #         model_name (str): Core target identifier tag to parse out of folders.

    #     Raises:
    #         FileNotFoundError: If matching runtime payloads are missing on disk.
    #         ProjectConfigError: If the blueprint lacks structural definition entries.
    #     """
    #     model_io_path = os.path.join(self.dirs["models"], f"{model_name}.pt")
    #     config_io_path = os.path.join(self.dirs["configs"], f"{model_name}.json")

    #     if not os.path.exists(config_io_path) or not os.path.exists(model_io_path):
    #         raise FileNotFoundError(f"Project session reload failed. Missing elements for checkpoint target: '{model_name}'")

    #     # 1. Parse JSON project metadata manifest
    #     with open(config_io_path, "r", encoding="utf-8") as f:
    #         self.config = json.load(f)
        
    #     model_spec = self.config.get("model_spec")
    #     if not model_spec:
    #         raise ProjectConfigError(f"Corrupted configuration structure in {config_io_path}. Missing 'model_spec' entries.")

    #     # 2. Re-compile graph architecture via structural dictionaries
    #     self.define_model(
    #         encoder_config=model_spec["encoder"],
    #         decoder_config=model_spec["decoder"],
    #         projector_config=model_spec.get("projector")
    #     )

    #     # 3. Mount binary state maps back into runtime memory structures
    #     self.model.load_state_dict(torch.load(model_io_path, map_location=self.device))
    #     self.model.eval()
        
    #     # Synchronize analytics monitoring arrays if available
    #     if "history" in self.config:
    #         self.history_tracker = self.config["history"]

    #     logger.info("Project workspace '%s' re-activated successfully in evaluation mode.", model_name)

    # def transform(self, raw_spectrum_data: Any) -> torch.Tensor:
    #     """
    #     Performs direct forward-pass feature extraction (inference) into the latent space.

    #     Args:
    #         raw_spectrum_data (Any): Raw spatial target profiling arrays.

    #     Raises:
    #         ModelNotInitializedError: If the network graph is uninitialized.

    #     Returns:
    #         torch.Tensor: Compressed latent embeddings tensor output.
    #     """
    #     if self.model is None:
    #         raise ModelNotInitializedError("Inference step blocked: Neural graph is uninitialized.")
        
    #     self.model.eval()
    #     with torch.no_grad():
    #         tensor_data = torch.tensor(raw_spectrum_data, dtype=torch.float32).to(self.device)
    #         if hasattr(self.model, 'encoder'):
    #             return self.model.encoder(tensor_data)
    #         else:
    #             # Fallback path if custom architecture combines pipelines implicitly
    #             return self.model(tensor_data)[1]