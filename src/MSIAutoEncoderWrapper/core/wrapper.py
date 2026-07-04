"""
Core Fasade (Wrapper) module orchestrating the subcomponents of the MSI Library.
Provides a clean, high-level API for end-users to manage Mass Spectrometry Imaging pipelines.
"""

import os
import json
import torch
from typing import Union, Dict, Any, Optional

# Core system imports
from msi_lib.utils.logger import get_logger
from msi_lib.loader.manager import LoaderManager
from msi_lib.binners.manager import BinningManager
from msi_lib.models.architecture.manager import ArchitectureManager
from msi_lib.models.datasets.manager import DatasetManager
from msi_lib.training.manager import TrainingManager
from msi_lib.core.utils.exceptions import (
    ProjectConfigError, 
    ModelNotInitializedError, 
    IncompatibleInterfaceError
)

logger = get_logger(__name__)


class MSIAutoEncoderWrapper:
    """
    Main Facade class that integrates data loading, binning, modeling, and training into a single workspace session.
    
    This wrapper abstracts underlying manager complexities and automates structural directory setup,
    state tracking, configuration persistence, and checkpoint loading.

    Attributes:
        project_path (str): Absolute path to the active project working directory.
        device (str): PyTorch compute device ('cuda', 'cpu', etc.).
        dirs (Dict[str, str]): Evaluated absolute paths for project subdirectories.
        config (Dict[str, Any]): Runtime parameters, tracking data components and architectural setups.
        model (Optional[torch.nn.Module]): The underlying PyTorch model compiled by the ArchitectureManager.
        history_tracker (Optional[Dict[str, Any]]): Training phase loss metrics history.
    """

    def __init__(self, project_path: str, device: Optional[str] = None):
        """
        Initializes the MSI session workspace and generates the folder structure.

        Args:
            project_path (str): Target filesystem location to generate or read the project workspace.
            device (Optional[str]): Explicit device mapping. If None, automatically infers CUDA availability.
        """
        self.project_path: str = os.path.abspath(project_path)
        self.device: str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Automated directory structural mapping
        self.dirs: Dict[str, str] = {
            "base": self.project_path,
            "models": os.path.join(self.project_path, "models"),
            "imgs": os.path.join(self.project_path, "imgs"),
            "configs": os.path.join(self.project_path, "configs"),
            "history": os.path.join(self.project_path, "history")
        }
        self._init_project_structures()

        # Session internal states
        self.config: Dict[str, Any] = {}
        self.model: Optional[torch.nn.Module] = None
        self.history_tracker: Optional[Dict[str, Any]] = None
        
        logger.info("MSI project initialized at %s on device: %s", self.project_path, self.device)

    def _init_project_structures(self) -> None:
        """
        Idempotently builds the structural subdirectory trees for project persistence.
        """
        for name, path in self.dirs.items():
            if not os.path.exists(path):
                os.makedirs(path)
                logger.debug("Created workspace directory [%s]: %s", name, path)

    def define_model(
        self, 
        encoder_config: Dict[str, Any], 
        decoder_config: Dict[str, Any], 
        projector_config: Optional[Dict[str, Any]] = None
    ) -> torch.nn.Module:
        """
        Compiles the full autoencoder architecture using textual configuration specifications.

        Args:
            encoder_config (Dict[str, Any]): Structural hyperparameters for the Encoder module.
            decoder_config (Dict[str, Any]): Structural hyperparameters for the Decoder module.
            projector_config (Optional[Dict[str, Any]]): Optional configuration for contrastive projector heads.

        Returns:
            torch.nn.Module: Compiled PyTorch model deployed to the configured device.
        """
        logger.info("Requesting structural model compilation from ArchitectureManager...")
        
        # Cache configuration schema within the runtime state
        self.config["model_spec"] = {
            "encoder": encoder_config,
            "decoder": decoder_config,
            "projector": projector_config
        }

        # Delegate execution down to the specialized architecture subsystem
        self.model = ArchitectureManager.compile_architecture(
            encoder_cfg=encoder_config,
            decoder_cfg=decoder_config,
            projector_cfg=projector_config
        )
        self.model.to(self.device)
        
        logger.info("Model compiled successfully and mounted onto device: %s", self.device)
        return self.model

    def fit(self, dataloader_config: Dict[str, Any], training_config: Dict[str, Any], epochs: int) -> Dict[str, Any]:
        """
        Orchestrates and executes the training pipeline across data loading, binning, and loss calculation.

        Args:
            dataloader_config (Dict[str, Any]): Direct payload identifying loader strategy, binner, and spatial data.
            training_config (Dict[str, Any]): Solver hyperparams, optimization targets, criteria weights, and freeze schedules.
            epochs (int): Total training runtime limit.

        Raises:
            ModelNotInitializedError: If triggered before executing `define_model` or `load`.

        Returns:
            Dict[str, Any]: History payload containing evaluated training losses per phase/epoch.
        """
        if self.model is None:
            raise ModelNotInitializedError(
                "Execution context blocked: Model not initialized. Run `define_model()` or `load()` first."
            )

        logger.info("Initiating training session execution loop for %d epochs.", epochs)
        
        # 1. Resolve domain extraction strategy (Loaders and Binners setup)
        loader_instance = LoaderManager.get_strategy(dataloader_config.get("loader"))
        binner_instance = BinningManager.get_strategy(dataloader_config.get("binner"))
        
        # Track spatial array processing shape dependencies
        self.config["dataloader_setup"] = dataloader_config
        self.config["dataloader_setup"]["resolved_input_dim"] = binner_instance.output_dim

        # 2. Fabricate unified PyTorch Dataset object via DatasetManager
        dataset = DatasetManager.create_dataset(
            loader=loader_instance,
            binner=binner_instance,
            strategy_name=dataloader_config.get("sampling_strategy", "pixel_level")
        )
        
        # 3. Request abstract execution engine configuration via TrainingManager
        self.config["training_setup"] = training_config
        trainer = TrainingManager.compile_trainer(
            model=self.model,
            dataset=dataset,
            training_cfg=training_config,
            device=self.device
        )

        # 4. Trigger training loop and isolate optimization execution traces
        logger.info("Training engine built. Starting loop execution...")
        self.history_tracker = trainer.run_training(epochs=epochs)
        
        # Append historical performance charts to metadata registry
        self.config["history"] = self.history_tracker
        logger.info("Training runtime successfully concluded. Performance tracking synchronized.")
        
        # Automated checkpointing fallback for runtime session continuity
        self.save(model_name="auto_fallback_checkpoint")
        
        return self.history_tracker

    def save(self, model_name: str) -> str:
        """
        Saves the complete configuration profile and structural model checkpoints separately.

        Ensures clean detachment of binary state_dicts (.pt) from textual metadata manifests (.json).

        Args:
            model_name (str): Identifier name to bind to file exports.

        Raises:
            ModelNotInitializedError: If the internal model instance is missing.

        Returns:
            str: Absolute file path location of the exported configuration manifest.
        """
        if self.model is None:
            raise ModelNotInitializedError("Persistence failed: No initialized model structure found to save.")

        model_io_path = os.path.join(self.dirs["models"], f"{model_name}.pt")
        config_io_path = os.path.join(self.dirs["configs"], f"{model_name}.json")

        # 1. Isolate and save binary tensors
        torch.save(self.model.state_dict(), model_io_path)
        logger.info("PyTorch weights exported successfully to: %s", model_io_path)

        # 2. Persist comprehensive metadata registry blueprint
        with open(config_io_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        logger.info("Session configuration manifest saved to: %s", config_io_path)
        
        return config_io_path

    def load(self, model_name: str) -> None:
        """
        Restores a historical project session environment dynamically from filesystem payloads.

        Reads the structural JSON blueprint to re-compile the neural network dynamically, 
        then maps the saved weights into the graph.

        Args:
            model_name (str): Core target identifier tag to parse out of folders.

        Raises:
            FileNotFoundError: If matching runtime payloads are missing on disk.
            ProjectConfigError: If the blueprint lacks structural definition entries.
        """
        model_io_path = os.path.join(self.dirs["models"], f"{model_name}.pt")
        config_io_path = os.path.join(self.dirs["configs"], f"{model_name}.json")

        if not os.path.exists(config_io_path) or not os.path.exists(model_io_path):
            raise FileNotFoundError(f"Project session reload failed. Missing elements for checkpoint target: '{model_name}'")

        # 1. Parse JSON project metadata manifest
        with open(config_io_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        model_spec = self.config.get("model_spec")
        if not model_spec:
            raise ProjectConfigError(f"Corrupted configuration structure in {config_io_path}. Missing 'model_spec' entries.")

        # 2. Re-compile graph architecture via structural dictionaries
        self.define_model(
            encoder_config=model_spec["encoder"],
            decoder_config=model_spec["decoder"],
            projector_config=model_spec.get("projector")
        )

        # 3. Mount binary state maps back into runtime memory structures
        self.model.load_state_dict(torch.load(model_io_path, map_location=self.device))
        self.model.eval()
        
        # Synchronize analytics monitoring arrays if available
        if "history" in self.config:
            self.history_tracker = self.config["history"]

        logger.info("Project workspace '%s' re-activated successfully in evaluation mode.", model_name)

    def transform(self, raw_spectrum_data: Any) -> torch.Tensor:
        """
        Performs direct forward-pass feature extraction (inference) into the latent space.

        Args:
            raw_spectrum_data (Any): Raw spatial target profiling arrays.

        Raises:
            ModelNotInitializedError: If the network graph is uninitialized.

        Returns:
            torch.Tensor: Compressed latent embeddings tensor output.
        """
        if self.model is None:
            raise ModelNotInitializedError("Inference step blocked: Neural graph is uninitialized.")
        
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.tensor(raw_spectrum_data, dtype=torch.float32).to(self.device)
            if hasattr(self.model, 'encoder'):
                return self.model.encoder(tensor_data)
            else:
                # Fallback path if custom architecture combines pipelines implicitly
                return self.model(tensor_data)[1]