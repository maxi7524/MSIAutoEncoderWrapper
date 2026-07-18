"""
Core multi-phase training execution engine handling optimization loops, gradient routing, and CSV metrics logging.
"""

import random
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..criterions.criterions_manager import CriterionsManager
from ...utils.logger import get_custom_logger
from ...utils.exceptions import raise_project_config_error, raise_validation_error
from ...utils.configuration import ConfigurableComponent, make_json_compatible

# Logger initialization
logger = get_custom_logger(__name__)


class MSIPyTorchTrainer(ConfigurableComponent):
    """
    High-performance, multi-phase execution engine automating optimization loops for generic MSI neural networks.
    """

    def __init__(self, wrapper_ref: Any, patience_limit: int = 10) -> None:
        """
        Initializes the multi-phase execution trainer engine.

        :param wrapper_ref: Reference to the coordinating master facade facade facade object instance.
        :type wrapper_ref: Any
        :param patience_limit: Iteration limit allowed before early stopping, defaults to 10.
        :type patience_limit: int
        """
        self._wrapper = wrapper_ref
        self.patience_limit = patience_limit
        self.best_loss = float("inf")
        self.patience_counter = 0
        self._config = {"patience_limit": patience_limit}

# --------------------------------------------------
# Section: Main training loop 
# --------------------------------------------------

    #  Training Process Orchestration Pass)
    def fit(self, training_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes sequential optimization passes loop by loop across the structured multi-phase configuration ledger array.

        :param training_config: Complete multi-phase configuration setup dict containing metrics, targets, parameters.
        :type training_config: Dict[str, Any]
        :return: Aggregated metrics history logs collected across all execution loops.
        :rtype: List[Dict[str, Any]]
        """
        # Execute pre-flight validations
        self.validate_training_setup(training_config=training_config)
        self._config["training"] = make_json_compatible(training_config)

        # Deterministic reproducibility enforcement checks
        seed = training_config.get("seed")
        if seed is not None:
            logger.info("Enforcing global deterministic execution pipeline using seed token: %s", seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        active_context = self._wrapper.active_context
        model = self._wrapper.active_model
        dataset = self._wrapper.active_dataset

        model_type = getattr(self._wrapper.models_manager, "active_model_type", "unknown_model")
        model_name = (
            getattr(self._wrapper.models_manager, "_active_model_name", None)
            or model_type
        )
        image_key = active_context._instantiated_image_key

        global_history: List[Dict[str, Any]] = []
        phases_list: List[Dict[str, Any]] = training_config.get("phases", [])
        transient_cache = getattr(self._wrapper.models_manager, "_training_transient_cache", {})

        # Heading 1 (Sequential Phase Processing Framework)
        for current_step, phase_config in enumerate(phases_list):
            phase_name = phase_config.get("phase_name", f"phase_{current_step + 1}")
            epochs = phase_config.get("epochs", 10)
            logger.info("Initiating sequential training loop phase: %s (%s/%s)", phase_name, current_step + 1, len(phases_list))

            ## Heading 2 (Gradient Lock Adjustments)
            ### Adjust layer weight modifications status dynamically by traversing model child boundaries
            freeze_targets = phase_config.get("freeze", [])
            for name, child_module in model.named_children():
                if name in freeze_targets:
                    logger.info("Gradient routing policy: Freezing component parameter gradients for: %s", name)
                    for param in child_module.parameters():
                        param.requires_grad = False
                else:
                    for param in child_module.parameters():
                        param.requires_grad = True

            ## Heading 2 (Dynamic Mathematical Criteria Graph Compilation)
            criterions_setup = phase_config.get("criterions", {})
            composite_loss = CriterionsManager.build_composite_loss(
                model_type=model_type,
                loss_setup=criterions_setup
            )

            ## Heading 2 (Dynamic Optimizer Reflection Allocation)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            opt_config = phase_config.get("optimizer")
            
            if opt_config:
                opt_type = opt_config.get("type", "AdamW")
                opt_params = opt_config.get("params", {})
            else:
                ### State-of-the-Art automatic fallback parameters allocation based on current model type definitions
                opt_type = "AdamW"
                opt_params = {"lr": 1e-3, "weight_decay": 1e-4}
                logger.info("Optimizer blueprint unassigned: Deploying standard SOTA optimizer fallback parameters.")

            if not hasattr(torch.optim, opt_type):
                raise_project_config_error(
                    context_name="Trainer",
                    message=f"Optimizer '{opt_type}' is not available in torch.optim.",
                )
            
            optimizer_class = getattr(torch.optim, opt_type)
            optimizer = optimizer_class(trainable_params, **opt_params)

            ## Heading 2 (Lifecycle Hook Initialization Pass)
            ### Trigger broad pre-computations hooks across individual sub-criterions matrices blocks
            for loss_fn in composite_loss.loss_functions.values():
                loss_fn.on_phase_start(model=model, dataset=dataset, transient_cache=transient_cache)

            # Heading 1 (Epoch Processing Execution Loop Partition)
            ## Resolve batch size prioritization: Phase Payload -> Wrapper Manager -> Fallback Default
            batch_size = phase_config.get(
                "batch_size", 
                getattr(self._wrapper.models_manager, "batch_size", 64)
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
            total_batches = len(dataloader)
            
            self.best_loss = float("inf")
            self.patience_counter = 0

            for epoch in range(epochs):
                model.train()
                epoch_start_time = time.time()
                accumulated_metrics: Dict[str, float] = {}

                ### Heading 3 (Batch Data Stream Execution Pass)
                for step_idx, batch in enumerate(dataloader):
                    #### Apply localized batch signal transformation hooks before triggering forward evaluation passes
                    for loss_fn in composite_loss.loss_functions.values():
                        batch = loss_fn.on_batch_start(batch_data=batch, transient_cache=transient_cache)

                    optimizer.zero_grad()
                    
                    global_device = getattr(self._wrapper, "device", "cpu")
                    spectra_tensor = batch[1].to(global_device)
                    
                    #### Execute model forward computation step
                    model_outputs = model(spectra_tensor)
                    
                    #### Evaluate composite objective loss sum vector matrix calculations scores
                    loss, loss_logs = composite_loss(model_outputs=model_outputs, batch_data=batch)
                    
                    loss.backward()
                    optimizer.step()

                    #### Accumulate numerical step tracking parameters metrics into localized registries
                    for key, scalar_val in loss_logs.items():
                        accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + scalar_val

                    #### Evaluate step progress and calculate mathematical timeline approximations
                    ##### Setup log interval reporting limits dynamically to prevent excessive output
                    log_interval = max(1, total_batches // 5)
                    if step_idx % log_interval == 0 or step_idx == total_batches - 1:
                        batches_completed = step_idx + 1
                        elapsed_seconds = time.time() - epoch_start_time
                        avg_seconds_per_batch = elapsed_seconds / batches_completed
                        remaining_batches = total_batches - batches_completed
                        eta_seconds = remaining_batches * avg_seconds_per_batch
                        
                        eta_minutes = int(eta_seconds // 60)
                        eta_secs = int(eta_seconds % 60)
                        
                        logger.info(
                            "[%s] Epoch %s/%s | Batch %s/%s | Loss: %s | Elapsed: %s s | ETA: %02d:%02d",
                            phase_name,
                            epoch + 1,
                            epochs,
                            batches_completed,
                            total_batches,
                            f"{loss.item():.4f}",
                            f"{elapsed_seconds:.1f}",
                            eta_minutes,
                            eta_secs
                        )

                ### Heading 3 (Epoch Performance Summarization Pass)
                mean_metrics: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "duration": time.time() - epoch_start_time
                }
                for key, running_sum in accumulated_metrics.items():
                    mean_metrics[key] = running_sum / total_batches

                global_history.append({"phase": phase_name, "metrics": mean_metrics})
                current_epoch_loss = mean_metrics["total_loss"]

                ### Heading 3 (Workspace Metrics Stream Flushing Pass)
                #### Delegate file appending directly to the workspace manager to maintain absolute decoupling
                self._wrapper.workspace.save_history(
                    img_name=image_key,
                    model_name=model_name,
                    history_dict=global_history,
                )

                ### Heading 3 (Early Stopping Validation Checkpoints)
                if current_epoch_loss < self.best_loss:
                    self.best_loss = current_epoch_loss
                    self.patience_counter = 0
                    
                    #### Save optimal tracking model parameters checkpoints onto disk layouts structures via workspace proxy delegation
                    self._wrapper.workspace.save_model_weights(
                        img_name=image_key,
                        model_name=model_name,
                        state_dict=model.state_dict(),
                    )
                    logger.debug("Performance milestone verified: Model weights updated via workspace delegation.")
                else:
                    self.patience_counter += 1

                logger.info(
                    "=== Epoch Summary [%s] %03d/%03d | Avg Loss: %s | Patience: %s/%s | Duration: %s s ===",
                    phase_name,
                    epoch + 1,
                    epochs,
                    f"{current_epoch_loss:.4f}",
                    self.patience_counter,
                    self.patience_limit,
                    f"{mean_metrics['duration']:.2f}"
                )

                if self.patience_counter >= self.patience_limit:
                    logger.info("Early stopping barrier triggered. Terminating active optimization loop sequence.")
                    break

        logger.info("All configured sequential multi-phase training loops successfully completed.")
        return global_history

# --------------------------------------------------
# Section: Helpers
# --------------------------------------------------

    def validate_training_setup(self, training_config: Dict[str, Any]) -> None:
        """
        Validates the state of all session dependencies and configuration parameters before starting the training process.

        :param training_config: The structural training configuration dictionary sheet.
        :type training_config: Dict[str, Any]
        :raises ValueError: If any required core components or data properties are missing from the active session.
        """
        # Stateful session requirements checks
        ## 1. Verify existence of active context, compiled model and bounded dataset
        active_context = getattr(self._wrapper, "active_context", None)
        model = getattr(self._wrapper, "active_model", None)
        dataset = getattr(self._wrapper, "active_dataset", None)

        if active_context is None or model is None or dataset is None:
            raise_validation_error(
                context_name="Trainer",
                message=(
                    "An active image context, compiled model, and initialized dataset "
                    "are required."
                ),
            )

        ## 2. Verify reader attachment inside the current execution context
        if not getattr(active_context, "reader", None):
            raise_validation_error(
                context_name="Trainer",
                message="The active image context has no reader instance.",
            )

        ## 3. Enforce structural validation on the phases layout definition list
        phases = training_config.get("phases", [])
        if not phases or not isinstance(phases, list):
            raise_validation_error(
                context_name="Trainer",
                message="Training configuration must contain a non-empty 'phases' list.",
            )

        logger.info("Pre-flight validation successful. Training environment maps verified.")
