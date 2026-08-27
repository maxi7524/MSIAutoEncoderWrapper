"""
Core multi-phase training execution engine handling optimization loops, gradient routing, and CSV metrics logging.
"""

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from ..criterions.criterions_manager import CriterionsManager
from ...utils.logger import get_custom_logger
from ...utils.exceptions import raise_project_config_error, raise_validation_error
from ...configuration import ConfigurableComponent, make_json_compatible
from ...data import (
    BatchPreprocessor,
    RawDatasetView,
    RawSpectrumBatch,
    SharedAxisRawBatch,
    RawSpectrumCollator,
    SpectrumBatch,
)

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
        create_partitions = getattr(dataset, "create_partitions", None)
        if callable(create_partitions):
            partitions = create_partitions()
            dataset_partitions = {
                "train": partitions.train,
                "validation": partitions.validation,
                "test": partitions.test,
            }
        else:
            dataset_partitions = {
                "train": dataset,
                "validation": None,
                "test": None,
            }
        normalization = getattr(active_context, "normalization", None)
        if normalization is not None:
            logger.info("Fitting normalization state on the training partition.")
            normalization.fit(dataset_partitions["train"])

        model_type = getattr(self._wrapper.models_manager, "active_model_type", "unknown_model")
        model_name = (
            getattr(self._wrapper.models_manager, "_active_model_name", None)
            or model_type
        )
        image_key = active_context._instantiated_image_key
        if image_key is None:
            image_key = getattr(self._wrapper.workspace, "active_img_name", None)
        checkpoint_config = self._resolve_checkpoint_config(training_config)

        runtime_config = training_config.get("runtime", {})
        runtime_model_name = runtime_config.get("model_name")
        if runtime_model_name is not None and not isinstance(runtime_model_name, str):
            raise_validation_error(
                context_name="Trainer",
                message="runtime.model_name must be a string when provided.",
            )
        if runtime_model_name:
            # Runtime artifact identity
            ## Checkpoints and histories must be isolated per task, while the
            ## serialized model metadata continues to contain the architecture name.
            model_name = runtime_model_name
        runtime_checkpoint = None
        if runtime_config.get("enabled", False) and runtime_config.get("resume", True) and not training_config.get("test_mode", False):
            from ...runtime.checkpoints import load_training_checkpoint

            runtime_checkpoint = load_training_checkpoint(
                Path(runtime_config["checkpoint_path"]),
                task_fingerprint=runtime_config["task_fingerprint"],
            )
        global_history: List[Dict[str, Any]] = (
            list(runtime_checkpoint["history"]) if runtime_checkpoint else []
        )
        epoch_metric_config = self._resolve_epoch_metric_config(training_config)
        phases_list: List[Dict[str, Any]] = training_config.get("phases", [])
        transient_cache = getattr(self._wrapper.models_manager, "_training_transient_cache", {})

        # Heading 1 (Sequential Phase Processing Framework)
        for current_step, phase_config in enumerate(phases_list):
            if runtime_checkpoint and current_step < int(runtime_checkpoint["phase_index"]):
                continue
            phase_name = phase_config.get("phase_name", f"phase_{current_step + 1}")
            epochs = phase_config.get("epochs", 10)
            compute_device = torch.device(
                phase_config.get(
                    "compute_device",
                    training_config.get(
                        "compute_device", getattr(self._wrapper, "device", "cpu")
                    ),
                )
            )
            preprocessing_device = torch.device(
                phase_config.get(
                    "preprocessing_device",
                    training_config.get("preprocessing_device", compute_device),
                )
            )
            if compute_device.type == "cuda" and not torch.cuda.is_available():
                raise_validation_error("Trainer", "compute_device requires available CUDA.")
            if preprocessing_device.type == "cuda" and not torch.cuda.is_available():
                raise_validation_error(
                    "Trainer", "preprocessing_device requires available CUDA."
                )
            model.to(compute_device)
            logger.info("Initiating sequential training loop phase: %s (%s/%s)", phase_name, current_step + 1, len(phases_list))

            ## Heading 2 (Gradient Lock Adjustments)
            ### Adjust layer weight modifications status dynamically by traversing model child boundaries
            freeze_targets = phase_config.get("freeze", [])
            self._apply_freeze_configuration(model, freeze_targets)

            ## Heading 2 (Dynamic Mathematical Criteria Graph Compilation)
            criterions_setup = phase_config.get("criterions", {})
            composite_loss = CriterionsManager.build_model_composite_loss(
                model_type=model_type,
                loss_setup=criterions_setup,
                head_specs=getattr(model, "head_specs", {}),
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

            # Runtime continuation
            ## Restore the exact optimizer, weights and random streams for this phase
            start_epoch = 0
            if runtime_checkpoint and current_step == int(runtime_checkpoint["phase_index"]):
                from ...runtime.checkpoints import restore_random_state

                model.load_state_dict(runtime_checkpoint["model_state"])
                optimizer.load_state_dict(runtime_checkpoint["optimizer_state"])
                restore_random_state(runtime_checkpoint)
                start_epoch = int(runtime_checkpoint["epoch"]) + 1

            ## Heading 2 (Lifecycle Hook Initialization Pass)
            ### Trigger broad pre-computations hooks across individual sub-criterions matrices blocks
            for loss_fn in composite_loss.loss_functions.values():
                loss_fn.on_phase_start(model=model, dataset=dataset, transient_cache=transient_cache)

            # Heading 1 (Epoch Processing Execution Loop Partition)
            ## Resolve DataLoader settings from the same configuration used by estimation
            dataloader = self._build_dataloader(
                dataset=dataset_partitions["train"],
                phase_config=phase_config,
                device=preprocessing_device,
            )
            batch_preprocessor = (
                BatchPreprocessor(dataset, preprocessing_device, compute_device)
                if isinstance(dataloader.dataset, RawDatasetView)
                else None
            )
            validation_loader = None
            validation_preprocessor = None
            validation_dataset = dataset_partitions.get("validation")
            if validation_dataset is not None and len(validation_dataset) > 0:
                validation_phase = dict(phase_config)
                validation_phase["dataloader"] = {
                    **phase_config.get("dataloader", {}),
                    "shuffle": False,
                    "drop_last": False,
                }
                validation_loader = self._build_dataloader(
                    dataset=validation_dataset,
                    phase_config=validation_phase,
                    device=preprocessing_device,
                )
                validation_preprocessor = (
                    BatchPreprocessor(dataset, preprocessing_device, compute_device)
                    if isinstance(validation_loader.dataset, RawDatasetView)
                    else None
                )
            epoch_metric_loaders = self._build_epoch_metric_loaders(
                phase_config=phase_config,
                dataset_partitions=dataset_partitions,
                preprocessing_device=preprocessing_device,
                compute_device=compute_device,
                metric_config=epoch_metric_config,
            )
            total_batches = len(dataloader)
            if total_batches < 1:
                raise_validation_error(
                    context_name="Trainer",
                    message=(
                        f"Phase '{phase_name}' produced an empty DataLoader. "
                        "Reduce batch_size or disable drop_last."
                    ),
                )
            
            max_batches = phase_config.get("max_batches")
            if max_batches is not None and (not isinstance(max_batches, int) or max_batches < 1):
                raise_validation_error("Trainer", "max_batches must be a positive integer.")
            processed_batches_limit = min(total_batches, max_batches or total_batches)
            self.best_loss = (
                float(runtime_checkpoint["best_loss"])
                if runtime_checkpoint and current_step == int(runtime_checkpoint["phase_index"])
                else float("inf")
            )
            self.patience_counter = (
                int(runtime_checkpoint["patience_counter"])
                if runtime_checkpoint and current_step == int(runtime_checkpoint["phase_index"])
                else 0
            )
            task_progress = None
            if runtime_config and not training_config.get("test_mode", False):
                from ...runtime.progress import create_terminal_progress

                task_progress = create_terminal_progress(
                    total=(epochs - start_epoch) * processed_batches_limit,
                    description=runtime_config.get("task_label", model_name),
                    position=1,
                    leave=False,
                )

            for epoch in range(start_epoch, epochs):
                model.train()
                epoch_start_time = time.time()
                accumulated_metrics: Dict[str, float] = {}

                ### Heading 3 (Batch Data Stream Execution Pass)
                for step_idx, batch in enumerate(dataloader):
                    if step_idx >= processed_batches_limit:
                        break
                    if isinstance(batch, (RawSpectrumBatch, SharedAxisRawBatch)):
                        if batch_preprocessor is None:
                            raise_validation_error(
                                "Trainer", "Raw batches require a batch preprocessor."
                            )
                        batch = batch_preprocessor(batch)
                    elif isinstance(batch, SpectrumBatch):
                        batch = batch.to(
                            compute_device,
                            non_blocking=compute_device.type == "cuda",
                        )
                    #### Apply localized batch signal transformation hooks before triggering forward evaluation passes
                    for loss_fn in composite_loss.loss_functions.values():
                        batch = loss_fn.on_batch_start(batch_data=batch, transient_cache=transient_cache)

                    self._ensure_finite_tensors(
                        batch,
                        location=(
                            f"phase '{phase_name}', epoch {epoch + 1}, "
                            f"batch {step_idx + 1} input"
                        ),
                    )

                    optimizer.zero_grad(set_to_none=True)
                    
                    spectra_tensor = (
                        batch.model_input()
                        if isinstance(batch, SpectrumBatch)
                        else batch[1].to(
                            compute_device,
                            non_blocking=dataloader.pin_memory,
                        )
                    )
                    
                    #### Execute model forward computation step
                    #### REMARK: Here we push only spectra, as this is the model input 
                    model_outputs = model(spectra_tensor)
                    self._ensure_finite_tensors(
                        model_outputs,
                        location=(
                            f"phase '{phase_name}', epoch {epoch + 1}, "
                            f"batch {step_idx + 1} model output"
                        ),
                    )
                    
                    #### Evaluate composite objective loss sum vector matrix calculations scores
                    #### REMARK: Here we push whole batch (idx(img positions), spectra, targets)
                    loss, loss_logs = composite_loss(model_outputs=model_outputs, batch_data=batch)
                    if not bool(torch.isfinite(loss).all()):
                        raise_validation_error(
                            context_name="Trainer",
                            message=(
                                f"Non-finite loss in phase '{phase_name}', epoch "
                                f"{epoch + 1}, batch {step_idx + 1}. Check input "
                                "normalization, learning rate, and criterion parameters."
                            ),
                        )
                    
                    loss.backward()
                    gradient_clip_norm = phase_config.get("gradient_clip_norm")
                    if gradient_clip_norm is not None:
                        if (
                            isinstance(gradient_clip_norm, bool)
                            or float(gradient_clip_norm) <= 0
                        ):
                            raise_validation_error(
                                context_name="Trainer",
                                message="gradient_clip_norm must be greater than zero.",
                            )
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            max_norm=float(gradient_clip_norm),
                            error_if_nonfinite=False,
                        )
                        if not bool(torch.isfinite(gradient_norm)):
                            raise_validation_error(
                                context_name="Trainer",
                                message=(
                                    f"Non-finite gradients in phase '{phase_name}', "
                                    f"epoch {epoch + 1}, batch {step_idx + 1}."
                                ),
                            )
                    optimizer.step()
                    if task_progress is not None:
                        task_progress.update(1)

                    #### Accumulate numerical step tracking parameters metrics into localized registries
                    for key, scalar_val in loss_logs.items():
                        accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + scalar_val

                    #### Evaluate step progress and calculate mathematical timeline approximations
                    ##### Setup log interval reporting limits dynamically to prevent excessive output
                    log_interval = max(1, processed_batches_limit // 5)
                    if step_idx % log_interval == 0 or step_idx == processed_batches_limit - 1:
                        batches_completed = step_idx + 1
                        elapsed_seconds = time.time() - epoch_start_time
                        avg_seconds_per_batch = elapsed_seconds / batches_completed
                        remaining_batches = processed_batches_limit - batches_completed
                        eta_seconds = remaining_batches * avg_seconds_per_batch
                        remaining_epoch_count = epochs - epoch - 1
                        task_eta_seconds = eta_seconds + (
                            remaining_epoch_count
                            * processed_batches_limit
                            * avg_seconds_per_batch
                        )
                        
                        logger.info(
                            "[%s] Epoch %s/%s | Batch %s/%s | Loss: %s | Epoch elapsed: %s s | Model ETA: %02d:%02d:%02d",
                            phase_name,
                            epoch + 1,
                            epochs,
                            batches_completed,
                            processed_batches_limit,
                            f"{loss.item():.4f}",
                            f"{elapsed_seconds:.1f}",
                            int(task_eta_seconds // 3600),
                            int((task_eta_seconds % 3600) // 60),
                            int(task_eta_seconds % 60),
                        )
                        if task_progress is not None:
                            task_progress.set_postfix(
                                epoch=f"{epoch + 1}/{epochs}",
                                batch=f"{batches_completed}/{processed_batches_limit}",
                                loss=f"{loss.item():.4f}",
                                eta=f"{int(task_eta_seconds // 60)}m",
                                refresh=False,
                            )
                        if runtime_config and not training_config.get("test_mode", False):
                            from ...runtime.progress import update_progress

                            update_progress(
                                Path(runtime_config["progress_path"]),
                                {
                                    "status": "running",
                                    "phase": phase_name,
                                    "phase_index": current_step,
                                    "epoch": epoch + 1,
                                    "epochs_total": epochs,
                                    "batch": batches_completed,
                                    "batches_total": processed_batches_limit,
                                    "elapsed_seconds": elapsed_seconds,
                                    "estimated_remaining_seconds": eta_seconds,
                                    "estimated_task_remaining_seconds": task_eta_seconds,
                                    "loss": float(loss.item()),
                                    "task_label": runtime_config.get("task_label"),
                                },
                            )

                ### Heading 3 (Epoch Performance Summarization Pass)
                mean_metrics: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "duration": time.time() - epoch_start_time
                }
                for key, running_sum in accumulated_metrics.items():
                    mean_metrics[key] = running_sum / processed_batches_limit

                if validation_loader is not None:
                    validation_metrics = self._evaluate_loader(
                        model=model,
                        dataloader=validation_loader,
                        preprocessor=validation_preprocessor,
                        composite_loss=composite_loss,
                        compute_device=compute_device,
                        transient_cache=transient_cache,
                        max_batches=(
                            processed_batches_limit
                            if training_config.get("test_mode", False)
                            else None
                        ),
                    )
                    mean_metrics.update(
                        {f"validation_{name}": value for name, value in validation_metrics.items()}
                    )
                    current_epoch_loss = validation_metrics["total_loss"]
                    checkpoint_scope = "validation"
                else:
                    current_epoch_loss = mean_metrics["total_loss"]
                    checkpoint_scope = "train"
                if epoch_metric_config is not None:
                    for split_name, (metric_loader, metric_preprocessor) in epoch_metric_loaders.items():
                        mean_metrics[
                            f"{split_name}_{epoch_metric_config['metric_name']}"
                        ] = self._evaluate_multilabel_average_precision(
                            model=model,
                            dataloader=metric_loader,
                            preprocessor=metric_preprocessor,
                            compute_device=compute_device,
                            head_id=epoch_metric_config["head_id"],
                            target_field=epoch_metric_config["target_field"],
                        )
                mean_metrics["checkpoint_scope"] = checkpoint_scope
                improved = (
                    current_epoch_loss
                    < self.best_loss - checkpoint_config["min_delta"]
                )
                mean_metrics["is_best"] = improved
                if improved:
                    self.best_loss = current_epoch_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                mean_metrics["best_loss"] = self.best_loss
                global_history.append({"phase": phase_name, "metrics": mean_metrics})

                ### Heading 3 (Workspace Metrics Stream Flushing Pass)
                #### Delegate file appending directly to the workspace manager to maintain absolute decoupling
                if not training_config.get("test_mode", False):
                    self._wrapper.workspace.save_history(
                        img_name=image_key,
                        model_name=model_name,
                        history_dict=global_history,
                    )

                ### Heading 3 (Early Stopping Validation Checkpoints)
                if improved and checkpoint_config["enabled"]:
                    self._wrapper.workspace.save_model(
                        img_name=checkpoint_config["context_name"] or image_key,
                        model_name=checkpoint_config["model_name"] or model_name,
                        history=global_history,
                    )
                    logger.info(
                        "Saved new best checkpoint for phase '%s' at epoch %s.",
                        phase_name,
                        epoch + 1,
                    )

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

                # Runtime state persistence
                ## Persist continuation state after every successfully completed epoch
                checkpoint_interval = int(runtime_config.get("checkpoint_every_epochs", 1))
                if (
                    runtime_config.get("enabled", False)
                    and not training_config.get("test_mode", False)
                    and (epoch + 1) % checkpoint_interval == 0
                ):
                    from ...runtime.checkpoints import save_training_checkpoint
                    from ...runtime.progress import update_progress

                    save_training_checkpoint(
                        Path(runtime_config["checkpoint_path"]),
                        model=model,
                        optimizer=optimizer,
                        phase_index=current_step,
                        epoch=epoch,
                        history=global_history,
                        best_loss=self.best_loss,
                        patience_counter=self.patience_counter,
                        task_fingerprint=runtime_config["task_fingerprint"],
                    )
                    update_progress(
                        Path(runtime_config["progress_path"]),
                        {
                            "status": "running",
                            "phase": phase_name,
                            "phase_index": current_step,
                            "epoch": epoch + 1,
                            "epochs_total": epochs,
                            "global_epochs_completed": len(global_history),
                        },
                    )

                if self.patience_counter >= self.patience_limit:
                    logger.info("Early stopping barrier triggered. Terminating active optimization loop sequence.")
                    break

            if task_progress is not None:
                task_progress.close()

            if checkpoint_config["enabled"] and checkpoint_config["restore_best"]:
                best_weights = self._wrapper.workspace.load_model_weights(
                    img_name=checkpoint_config["context_name"] or image_key,
                    model_name=checkpoint_config["model_name"] or model_name,
                )
                model.load_state_dict(best_weights)
                logger.info(
                    "Restored the best checkpoint after phase '%s'.",
                    phase_name,
                )

            test_dataset = dataset_partitions.get("test")
            if current_step == len(phases_list) - 1 and test_dataset is not None and len(test_dataset) > 0:
                test_phase = dict(phase_config)
                test_phase["dataloader"] = {
                    **phase_config.get("dataloader", {}),
                    "shuffle": False,
                    "drop_last": False,
                }
                test_loader = self._build_dataloader(
                    dataset=test_dataset,
                    phase_config=test_phase,
                    device=preprocessing_device,
                )
                test_preprocessor = (
                    BatchPreprocessor(dataset, preprocessing_device, compute_device)
                    if isinstance(test_loader.dataset, RawDatasetView)
                    else None
                )
                test_metrics = self._evaluate_loader(
                    model=model,
                    dataloader=test_loader,
                    preprocessor=test_preprocessor,
                    composite_loss=composite_loss,
                    compute_device=compute_device,
                    transient_cache=transient_cache,
                )
                global_history.append({"phase": phase_name, "split": "test", "metrics": test_metrics})

        logger.info("All configured sequential multi-phase training loops successfully completed.")
        return global_history

    @staticmethod
    def _apply_freeze_configuration(
        model: nn.Module,
        freeze_targets: List[str],
    ) -> None:
        """Reset gradients and freeze configured component paths for one phase.

        :param model: Model whose parameter flags are updated in place.
        :type model: torch.nn.Module
        :param freeze_targets: Attribute paths such as ``encoder`` or
            ``heads.condition_primary``.
        :type freeze_targets: List[str]
        :return: None.
        :rtype: None
        :raises ValidationError: If a component path does not exist.
        """
        for parameter in model.parameters():
            parameter.requires_grad = True
        for target in freeze_targets:
            component: nn.Module = model
            try:
                for part in str(target).split("."):
                    component = (
                        component[part]
                        if isinstance(component, nn.ModuleDict)
                        else getattr(component, part)
                    )
            except (AttributeError, KeyError):
                raise_validation_error(
                    "Trainer", f"Unknown freeze target '{target}'."
                )
            logger.info("Freezing component path: %s", target)
            for parameter in component.parameters():
                parameter.requires_grad = False

    def _build_dataloader(
        self,
        dataset: Any,
        phase_config: Dict[str, Any],
        device: Any,
    ) -> DataLoader:
        """Build a phase DataLoader from validated user configuration.

        :param dataset: Active training dataset.
        :type dataset: Any
        :param phase_config: Current phase configuration.
        :type phase_config: Dict[str, Any]
        :param device: Active torch device or device token.
        :type device: Any
        :return: Configured PyTorch DataLoader.
        :rtype: torch.utils.data.DataLoader
        :raises ValidationError: If DataLoader parameters are incompatible.
        """
        batch_size = int(
            phase_config.get(
                "batch_size",
                getattr(self._wrapper.models_manager, "batch_size", 64),
            )
        )
        if batch_size < 1:
            raise_validation_error(
                context_name="Trainer",
                message="Every phase batch_size must be at least one.",
            )
        loader_config = dict(phase_config.get("dataloader", {}))
        if "batch_size" in loader_config:
            raise_validation_error(
                context_name="Trainer",
                message=(
                    "Set batch_size on the phase, not inside the dataloader block."
                ),
            )
        loader_config.setdefault("shuffle", True)
        worker_options = phase_config.get("preprocessing_num_workers", {})
        if isinstance(worker_options, dict):
            default_workers = worker_options.get(torch.device(device).type, 0)
        else:
            default_workers = worker_options
        loader_config.setdefault("num_workers", int(default_workers))
        loader_config.setdefault("pin_memory", str(device).startswith("cuda"))
        loader_config.setdefault("drop_last", False)
        seed = phase_config.get(
            "dataloader_seed",
            phase_config.get("seed", self._config.get("training", {}).get("seed")),
        )
        if seed is not None:
            from ...runtime.reproducibility import seed_worker

            generator = torch.Generator()
            generator.manual_seed(int(seed))
            loader_config.setdefault("generator", generator)
            loader_config.setdefault("worker_init_fn", seed_worker)
        workers = int(loader_config["num_workers"])
        if workers < 0:
            raise_validation_error(
                context_name="Trainer",
                message="dataloader.num_workers cannot be negative.",
            )
        if workers == 0:
            loader_config.pop("prefetch_factor", None)
            loader_config["persistent_workers"] = False
        else:
            loader_config.setdefault("prefetch_factor", 2)
            loader_config.setdefault("persistent_workers", True)
        try:
            source_dataset = getattr(dataset, "dataset", dataset)
            raw_getter = callable(getattr(source_dataset, "get_raw_item", None))
            if raw_getter:
                if "collate_fn" in loader_config:
                    raise_validation_error(
                        "Trainer",
                        "A custom collate_fn cannot replace the packed raw collator.",
                    )
                schemas_getter = getattr(source_dataset, "get_target_schemas", None)
                schemas = schemas_getter() if callable(schemas_getter) else {}
                loader_config["collate_fn"] = RawSpectrumCollator(schemas)
                dataset = RawDatasetView(dataset)
            return DataLoader(dataset, batch_size=batch_size, **loader_config)
        except (TypeError, ValueError) as error:
            raise_validation_error(
                context_name="Trainer",
                message=f"Invalid DataLoader configuration: {error}",
            )

    def _evaluate_loader(
        self,
        *,
        model: nn.Module,
        dataloader: DataLoader,
        preprocessor: BatchPreprocessor | None,
        composite_loss: nn.Module,
        compute_device: torch.device,
        transient_cache: Dict[str, Any],
        max_batches: int | None = None,
    ) -> Dict[str, float]:
        """Evaluate one dataset split without parameter updates.

        :param max_batches: Optional validation-batch bound used by test mode.
        :type max_batches: int | None
        :return: Mean criterion values over the processed validation batches.
        :rtype: Dict[str, float]
        """
        model.eval()
        accumulated: Dict[str, float] = {}
        processed_batches = 0
        evaluation_context = (
            torch.enable_grad()
            if any(
                loss.requires_input_grad
                for loss in composite_loss.loss_functions.values()
            )
            else torch.inference_mode()
        )
        with evaluation_context:
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                if isinstance(batch, (RawSpectrumBatch, SharedAxisRawBatch)):
                    if preprocessor is None:
                        raise_validation_error("Trainer", "Raw batches require preprocessing.")
                    batch = preprocessor(batch)
                elif isinstance(batch, SpectrumBatch):
                    batch = batch.to(compute_device, non_blocking=compute_device.type == "cuda")
                for loss_fn in composite_loss.loss_functions.values():
                    batch = loss_fn.on_batch_start(batch_data=batch, transient_cache=transient_cache)
                spectra = batch.model_input() if isinstance(batch, SpectrumBatch) else batch[1].to(compute_device)
                outputs = model(spectra)
                _, logs = composite_loss(model_outputs=outputs, batch_data=batch)
                for name, value in logs.items():
                    accumulated[name] = accumulated.get(name, 0.0) + value
                processed_batches += 1
        denominator = max(processed_batches, 1)
        return {name: value / denominator for name, value in accumulated.items()}

    def _build_epoch_metric_loaders(
        self,
        *,
        phase_config: Dict[str, Any],
        dataset_partitions: Dict[str, Dataset | None],
        preprocessing_device: torch.device,
        compute_device: torch.device,
        metric_config: Dict[str, Any] | None,
    ) -> Dict[str, tuple[DataLoader, BatchPreprocessor | None]]:
        """Build deterministic evaluation loaders requested by epoch metrics.

        :param phase_config: Active training phase configuration.
        :type phase_config: Dict[str, Any]
        :param dataset_partitions: Resolved train, validation, and test views.
        :type dataset_partitions: Dict[str, Dataset | None]
        :param preprocessing_device: Device used by raw-spectrum preprocessing.
        :type preprocessing_device: torch.device
        :param compute_device: Model execution device.
        :type compute_device: torch.device
        :param metric_config: Validated optional epoch metric configuration.
        :type metric_config: Dict[str, Any] | None
        :return: One non-shuffled loader and optional preprocessor per requested split.
        :rtype: Dict[str, tuple[DataLoader, BatchPreprocessor | None]]
        """
        if metric_config is None:
            return {}
        loaders: Dict[str, tuple[DataLoader, BatchPreprocessor | None]] = {}
        for split_name in metric_config["splits"]:
            split_dataset = dataset_partitions.get(split_name)
            if split_dataset is None or len(split_dataset) == 0:
                raise_validation_error(
                    "Trainer",
                    f"epoch_metrics requests unavailable split '{split_name}'.",
                )
            evaluation_phase = dict(phase_config)
            evaluation_phase["dataloader"] = {
                **phase_config.get("dataloader", {}),
                "shuffle": False,
                "drop_last": False,
            }
            loader = self._build_dataloader(
                dataset=split_dataset,
                phase_config=evaluation_phase,
                device=preprocessing_device,
            )
            preprocessor = (
                BatchPreprocessor(split_dataset, preprocessing_device, compute_device)
                if isinstance(loader.dataset, RawDatasetView)
                else None
            )
            loaders[split_name] = (loader, preprocessor)
        return loaders

    @staticmethod
    def _resolve_epoch_metric_config(
        training_config: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Validate optional per-epoch macro AP reporting for one multi-label head.

        :param training_config: Complete training configuration.
        :type training_config: Dict[str, Any]
        :return: Normalized AP metric configuration or ``None`` when disabled.
        :rtype: Dict[str, Any] | None
        """
        configured = training_config.get("epoch_metrics")
        if configured is None:
            return None
        if not isinstance(configured, dict):
            raise_validation_error("Trainer", "epoch_metrics must be a mapping.")
        ap_config = configured.get("multi_label_average_precision")
        if not isinstance(ap_config, dict):
            raise_validation_error(
                "Trainer",
                "epoch_metrics.multi_label_average_precision must be a mapping.",
            )
        head_id = ap_config.get("head_id")
        target_field = ap_config.get("target_field")
        splits = ap_config.get("splits", ["train", "test"])
        if not isinstance(head_id, str) or not head_id:
            raise_validation_error(
                "Trainer",
                "epoch_metrics.multi_label_average_precision.head_id must be a string.",
            )
        if not isinstance(target_field, str) or not target_field:
            raise_validation_error(
                "Trainer",
                "epoch_metrics.multi_label_average_precision.target_field must be a string.",
            )
        if (
            not isinstance(splits, list)
            or not splits
            or any(split not in {"train", "validation", "test"} for split in splits)
        ):
            raise_validation_error(
                "Trainer",
                "epoch_metrics.multi_label_average_precision.splits must name train, validation, or test.",
            )
        return {
            "metric_name": f"{head_id}_average_precision",
            "head_id": head_id,
            "target_field": target_field,
            "splits": list(dict.fromkeys(splits)),
        }

    @staticmethod
    def _evaluate_multilabel_average_precision(
        *,
        model: nn.Module,
        dataloader: DataLoader,
        preprocessor: BatchPreprocessor | None,
        compute_device: torch.device,
        head_id: str,
        target_field: str,
    ) -> float:
        """Return macro AP for one head without changing model state.

        Entries excluded by the dataset availability mask are excluded class by
        class. Classes without an observed positive are omitted from the macro
        mean because AP is undefined for them.

        :param model: Model being evaluated.
        :type model: nn.Module
        :param dataloader: Deterministic split loader.
        :type dataloader: DataLoader
        :param preprocessor: Optional raw-spectrum preprocessor.
        :type preprocessor: BatchPreprocessor | None
        :param compute_device: Model execution device.
        :type compute_device: torch.device
        :param head_id: Head identifier without the ``head_`` prefix.
        :type head_id: str
        :param target_field: Dataset multi-label target key.
        :type target_field: str
        :return: Macro average precision across defined class AP values.
        :rtype: float
        """
        output_key = f"head_{head_id}"
        logits_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        mask_batches: list[np.ndarray] = []
        model_was_training = model.training
        model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                if isinstance(batch, (RawSpectrumBatch, SharedAxisRawBatch)):
                    if preprocessor is None:
                        raise_validation_error("Trainer", "Raw batches require preprocessing.")
                    batch = preprocessor(batch)
                elif isinstance(batch, SpectrumBatch):
                    batch = batch.to(compute_device, non_blocking=compute_device.type == "cuda")
                if not isinstance(batch, SpectrumBatch):
                    raise_validation_error("Trainer", "Epoch metrics require SpectrumBatch inputs.")
                outputs = model(batch.model_input())
                if output_key not in outputs:
                    raise_validation_error("Trainer", f"Epoch metrics cannot find '{output_key}'.")
                if target_field not in batch.targets.values or target_field not in batch.targets.masks:
                    raise_validation_error(
                        "Trainer", f"Epoch metrics cannot find target '{target_field}'."
                    )
                logits_batches.append(outputs[output_key].detach().cpu().numpy())
                target_batches.append(batch.targets.values[target_field].detach().cpu().numpy())
                mask_batches.append(batch.targets.masks[target_field].detach().cpu().numpy())
        if model_was_training:
            model.train()
        logits = np.concatenate(logits_batches, axis=0)  # (N, C)
        targets = np.concatenate(target_batches, axis=0).astype(bool)  # (N, C)
        mask = np.concatenate(mask_batches, axis=0).astype(bool)
        if mask.ndim == 1:
            mask = np.broadcast_to(mask[:, None], targets.shape)
        if logits.shape != targets.shape or mask.shape != targets.shape:
            raise_validation_error("Trainer", "Epoch metric tensors must have equal [N, C] shapes.")
        probabilities = 1.0 / (1.0 + np.exp(-logits))  # (N, C)
        values = []
        for class_index in range(targets.shape[1]):
            available = mask[:, class_index]
            truth = targets[available, class_index]
            if bool(truth.any()):
                values.append(average_precision_score(truth, probabilities[available, class_index]))
        if not values:
            raise_validation_error("Trainer", "Epoch metrics found no classes with positive targets.")
        return float(np.mean(values))

    @classmethod
    def _ensure_finite_tensors(cls, value: Any, location: str) -> None:
        """Raise a standardized error when nested tensors contain NaN or infinity.

        :param value: Tensor or nested tensor container.
        :type value: Any
        :param location: Human-readable training stage.
        :type location: str
        :raises ValidationError: If a discovered tensor is non-finite.
        """
        tensors: List[torch.Tensor] = []
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                cls._ensure_finite_tensors(item, location)
            return
        elif isinstance(value, (tuple, list)):
            for item in value:
                if item is not None:
                    cls._ensure_finite_tensors(item, location)
            return
        for tensor in tensors:
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise_validation_error(
                    context_name="Trainer",
                    message=(
                        f"Non-finite tensor detected at {location}. Check reader "
                        "values, dataset normalization, and numerical settings."
                    ),
                )

    @staticmethod
    def _resolve_checkpoint_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize best-checkpoint behavior.

        :param training_config: Full training configuration.
        :type training_config: Dict[str, Any]
        :return: Normalized checkpoint settings.
        :rtype: Dict[str, Any]
        :raises ValidationError: If checkpoint settings are invalid.
        """
        configured = training_config.get("checkpoint", {})
        if not isinstance(configured, dict):
            raise_validation_error(
                context_name="Trainer",
                message="checkpoint must be a configuration dictionary.",
            )
        resolved = {
            "enabled": configured.get("enabled", True),
            "restore_best": configured.get("restore_best", True),
            "min_delta": configured.get("min_delta", 0.0),
            "context_name": configured.get("context_name"),
            "model_name": configured.get("model_name"),
        }
        if not isinstance(resolved["enabled"], bool) or not isinstance(
            resolved["restore_best"], bool
        ):
            raise_validation_error(
                context_name="Trainer",
                message="checkpoint enabled and restore_best values must be booleans.",
            )
        if isinstance(resolved["min_delta"], bool) or resolved["min_delta"] < 0:
            raise_validation_error(
                context_name="Trainer",
                message="checkpoint.min_delta must be non-negative.",
            )
        resolved["min_delta"] = float(resolved["min_delta"])
        return resolved

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

        ## 2. Verify the reader selected by the active dataset source
        source = getattr(dataset, "source", "image")
        data_reader_getter = getattr(active_context, "get_data_reader", None)
        if callable(data_reader_getter):
            selected_reader = data_reader_getter(source)
        else:
            selected_reader = getattr(active_context, "reader", None)
        if selected_reader is None:
            raise_validation_error(
                context_name="Trainer",
                message=f"The active '{source}' data source has no reader instance.",
            )

        ## 3. Enforce structural validation on the phases layout definition list
        phases = training_config.get("phases", [])
        if not phases or not isinstance(phases, list):
            raise_validation_error(
                context_name="Trainer",
                message="Training configuration must contain a non-empty 'phases' list.",
            )

        logger.info("Pre-flight validation successful. Training environment maps verified.")
