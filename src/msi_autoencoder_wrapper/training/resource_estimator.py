"""Pre-flight RAM, VRAM, and disk estimation for model training."""

from __future__ import annotations

import copy
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from ..utils.exceptions import raise_validation_error
from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class TrainingResourceEstimator:
    """Estimate training resources from a probe forward pass and configuration."""

    def __init__(self, wrapper_ref: Any) -> None:
        """Bind the estimator to the current model, dataset, and workspace.

        :param wrapper_ref: Root wrapper exposing runtime training state.
        :type wrapper_ref: Any
        """
        self._wrapper = wrapper_ref

    def estimate(
        self,
        training_config: Dict[str, Any],
        resource_limits: Optional[Dict[str, int | float]] = None,
        auto_adjust_batch_size: bool = False,
        safety_factor: float = 1.25,
    ) -> Dict[str, Any]:
        """Estimate resources and optionally reduce phase batch sizes to fit limits.

        Limit values in ``(0, 1]`` are interpreted as a fraction of the
        currently available resource. Values greater than one are interpreted
        as an absolute byte limit. For example, ``{"vram": 0.8}`` reserves 20
        percent of currently free VRAM, while ``{"ram": 8_000_000_000}`` sets
        an absolute RAM budget.

        The estimator performs one evaluation-mode forward pass on a dataset
        sample to observe activation shapes. It combines those measurements
        with parameter, gradient, optimizer, DataLoader, criterion-workspace,
        checkpoint, and history estimates. Runtime allocator fragmentation,
        third-party reader caches, and OS activity remain outside the model, so
        the result is an estimate rather than a guarantee.

        :param training_config: Multi-phase trainer configuration.
        :type training_config: Dict[str, Any]
        :param resource_limits: Optional RAM, VRAM, and disk fractions or bytes.
        :type resource_limits: Optional[Dict[str, int | float]]
        :param auto_adjust_batch_size: Reduce unsafe batch sizes in a copied config.
        :type auto_adjust_batch_size: bool
        :param safety_factor: Multiplicative reserve applied to memory estimates.
        :type safety_factor: float
        :return: Resource report and a recommended training configuration.
        :rtype: Dict[str, Any]
        :raises ValidationError: If runtime state, limits, or phases are invalid.
        """
        if safety_factor < 1.0:
            raise_validation_error(
                context_name="ResourceEstimator",
                message="safety_factor must be at least 1.0.",
            )
        model = getattr(self._wrapper, "active_model", None)
        dataset = getattr(self._wrapper, "active_dataset", None)
        if model is None or dataset is None:
            raise_validation_error(
                context_name="ResourceEstimator",
                message="An active model and dataset are required for estimation.",
            )
        phases = training_config.get("phases")
        if not isinstance(phases, list) or not phases:
            raise_validation_error(
                context_name="ResourceEstimator",
                message="training_config must contain a non-empty 'phases' list.",
            )

        sample_tensor = self._get_sample_tensor(dataset)
        parameter_bytes = self._tensor_bytes(model.parameters())
        buffer_bytes = self._tensor_bytes(model.buffers())
        trainable_bytes = self._tensor_bytes(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        activation_bytes = self._probe_activation_bytes(model, sample_tensor)
        available = self._available_resources()
        limits = self._resolve_limits(resource_limits or {}, available)
        recommended_config = copy.deepcopy(training_config)

        phase_reports = []
        for phase_index, phase in enumerate(phases):
            configured_batch = int(
                phase.get(
                    "batch_size",
                    getattr(self._wrapper.models_manager, "batch_size", 64),
                )
            )
            if configured_batch < 1:
                raise_validation_error(
                    context_name="ResourceEstimator",
                    message="Every phase batch_size must be at least one.",
                )
            recommended_batch = configured_batch
            estimate = self._estimate_phase(
                phase=phase,
                batch_size=configured_batch,
                sample_tensor=sample_tensor,
                parameter_bytes=parameter_bytes,
                buffer_bytes=buffer_bytes,
                trainable_bytes=trainable_bytes,
                activation_bytes=activation_bytes,
                safety_factor=safety_factor,
            )
            if auto_adjust_batch_size and not self._fits(estimate, limits):
                recommended_batch, estimate = self._find_fitting_batch_size(
                    phase=phase,
                    configured_batch=configured_batch,
                    sample_tensor=sample_tensor,
                    parameter_bytes=parameter_bytes,
                    buffer_bytes=buffer_bytes,
                    trainable_bytes=trainable_bytes,
                    activation_bytes=activation_bytes,
                    safety_factor=safety_factor,
                    limits=limits,
                )
                recommended_config["phases"][phase_index]["batch_size"] = recommended_batch

            phase_reports.append(
                {
                    "phase": phase.get("phase_name", f"phase_{phase_index + 1}"),
                    "configured_batch_size": configured_batch,
                    "recommended_batch_size": recommended_batch,
                    **estimate,
                    "fits_limits": self._fits(estimate, limits),
                }
            )

        disk_bytes = self._estimate_disk_bytes(
            model_bytes=parameter_bytes + buffer_bytes,
            phases=phases,
        )
        disk_fits = disk_bytes <= limits["disk"]
        logger.info(
            "Training resource estimation completed for %s phase(s).",
            len(phases),
        )
        return {
            "method": "probe_forward_plus_static_training_state",
            "safety_factor": safety_factor,
            "available_bytes": available,
            "limit_bytes": limits,
            "model_bytes": parameter_bytes + buffer_bytes,
            "estimated_disk_bytes": disk_bytes,
            "disk_fits_limit": disk_fits,
            "phases": phase_reports,
            "recommended_training_config": recommended_config,
            "limitations": [
                "Reader-native caches and operating-system activity are not measured.",
                "CUDA allocator fragmentation can increase actual peak VRAM.",
                "The estimate is not a substitute for monitoring the first training epoch.",
            ],
        }

    @staticmethod
    def format_report(report: Dict[str, Any]) -> str:
        """Return a concise human-readable resource summary.

        :param report: Report returned by :meth:`estimate`.
        :type report: Dict[str, Any]
        :return: Multiline RAM, VRAM, disk, and batch-size summary.
        :rtype: str
        """
        lines = ["Training resource estimate"]
        for phase in report["phases"]:
            lines.append(
                "- {phase}: batch {configured} -> {recommended}, RAM {ram}, "
                "VRAM {vram}, fits limits: {fits}".format(
                    phase=phase["phase"],
                    configured=phase["configured_batch_size"],
                    recommended=phase["recommended_batch_size"],
                    ram=TrainingResourceEstimator._format_bytes(
                        phase["estimated_ram_bytes"]
                    ),
                    vram=TrainingResourceEstimator._format_bytes(
                        phase["estimated_vram_bytes"]
                    ),
                    fits=phase["fits_limits"],
                )
            )
        lines.append(
            "- disk: {disk}, fits limit: {fits}".format(
                disk=TrainingResourceEstimator._format_bytes(
                    report["estimated_disk_bytes"]
                ),
                fits=report["disk_fits_limit"],
            )
        )
        return "\n".join(lines)

    @staticmethod
    def _format_bytes(value: int) -> str:
        """Format a byte count using binary units."""
        amount = float(value)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if amount < 1024.0 or unit == "TiB":
                return f"{amount:.2f} {unit}"
            amount /= 1024.0
        return f"{amount:.2f} TiB"

    @staticmethod
    def _get_sample_tensor(dataset: Any) -> torch.Tensor:
        """Load one dataset item and return its feature tensor."""
        if len(dataset) < 1:
            raise_validation_error(
                context_name="ResourceEstimator",
                message="The active dataset is empty.",
            )
        sample = dataset[0]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise_validation_error(
                context_name="ResourceEstimator",
                message="Dataset samples must contain a feature tensor at index 1.",
            )
        tensor = sample[1]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor.detach().float()

    def _probe_activation_bytes(
        self,
        model: torch.nn.Module,
        sample_tensor: torch.Tensor,
    ) -> int:
        """Measure module output sizes during one side-effect-free forward pass."""
        activation_bytes = 0

        def record_output(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            del module, inputs
            nonlocal activation_bytes
            activation_bytes += self._value_tensor_bytes(output)

        hooks = [module.register_forward_hook(record_output) for module in model.modules()]
        was_training = model.training
        device = next(model.parameters(), torch.empty(0)).device
        try:
            model.eval()
            with torch.no_grad():
                model(sample_tensor.unsqueeze(0).to(device))
        finally:
            for hook in hooks:
                hook.remove()
            model.train(was_training)
        return activation_bytes

    def _estimate_phase(
        self,
        phase: Dict[str, Any],
        batch_size: int,
        sample_tensor: torch.Tensor,
        parameter_bytes: int,
        buffer_bytes: int,
        trainable_bytes: int,
        activation_bytes: int,
        safety_factor: float,
    ) -> Dict[str, int]:
        """Estimate RAM and VRAM for one phase and batch size."""
        expansion = self._batch_expansion_factor(phase)
        sample_bytes = sample_tensor.numel() * sample_tensor.element_size()
        optimizer_bytes = self._optimizer_state_bytes(phase, trainable_bytes)
        criterion_bytes = self._criterion_workspace_bytes(
            phase,
            batch_size,
            sample_tensor.numel(),
            sample_tensor.element_size(),
        )
        device = str(getattr(self._wrapper, "device", "cpu"))
        loader_config = phase.get("dataloader", {})
        workers = int(loader_config.get("num_workers", 0))
        prefetch = int(loader_config.get("prefetch_factor", 2)) if workers else 0
        loader_copies = 1 + workers * prefetch
        loader_bytes = sample_bytes * batch_size * expansion * loader_copies
        dynamic_training_bytes = (
            sample_bytes * batch_size * expansion
            + activation_bytes * batch_size * expansion * 2
            + criterion_bytes
        )
        persistent_training_bytes = (
            parameter_bytes + buffer_bytes + trainable_bytes + optimizer_bytes
        )

        if device.startswith("cuda") and torch.cuda.is_available():
            vram = persistent_training_bytes + dynamic_training_bytes
            ram = loader_bytes
        else:
            vram = 0
            ram = persistent_training_bytes + dynamic_training_bytes + loader_bytes
        return {
            "estimated_ram_bytes": int(ram * safety_factor),
            "estimated_vram_bytes": int(vram * safety_factor),
        }

    @staticmethod
    def _batch_expansion_factor(phase: Dict[str, Any]) -> int:
        """Return the largest batch expansion required by configured hooks."""
        criterions = phase.get("criterions", {})
        contrastive = criterions.get("contrastive", {})
        if contrastive:
            return 2
        for name, config in criterions.items():
            target = config.get("target", name) if isinstance(config, dict) else config
            if target == "InfoNCELoss":
                return 2
        return 1

    @staticmethod
    def _optimizer_state_bytes(phase: Dict[str, Any], trainable_bytes: int) -> int:
        """Estimate optimizer tensors allocated alongside trainable parameters."""
        optimizer = phase.get("optimizer") or {"type": "AdamW", "params": {}}
        optimizer_type = str(optimizer.get("type", "AdamW")).lower()
        parameters = optimizer.get("params", {})
        if optimizer_type in {"adam", "adamw", "nadam", "radam"}:
            return 2 * trainable_bytes
        if optimizer_type == "sgd" and float(parameters.get("momentum", 0.0)) > 0:
            return trainable_bytes
        if optimizer_type in {"rmsprop", "adagrad", "adadelta"}:
            return 2 * trainable_bytes
        return trainable_bytes

    @staticmethod
    def _criterion_workspace_bytes(
        phase: Dict[str, Any],
        batch_size: int,
        bins: int,
        element_size: int,
    ) -> int:
        """Estimate dominant temporary matrices allocated by known criteria."""
        workspace = 0
        criterions = phase.get("criterions", {})
        entries = []
        for category_or_name, setup in criterions.items():
            if category_or_name == "heads":
                for head_setup in setup.values():
                    entries.extend(head_setup.items())
            elif category_or_name in {
                "reconstruction",
                "contrastive",
                "head",
                "regularization",
            }:
                entries.extend(setup.items())
            else:
                entries.append((category_or_name, setup))
        for name, config in entries:
            target = config.get("target", name) if isinstance(config, dict) else config
            if target == "MassersteinLoss":
                masserstein_workspace = 4 * batch_size * bins * element_size
                workspace = max(workspace, masserstein_workspace)
            if target == "InfoNCELoss":
                workspace += 3 * (2 * batch_size) ** 2 * element_size
            if target == "ContractiveLoss":
                parameters = config.get("params", {}) if isinstance(config, dict) else {}
                calculation_method = parameters.get(
                    "calculation_method", "approximate_hutchinson_vjp"
                )
                multiplier = bins if calculation_method == "exact_autograd_jacobian" else int(
                    parameters.get("num_probes", 1)
                )
                workspace += multiplier * batch_size * bins * element_size
        return workspace

    def _find_fitting_batch_size(
        self,
        phase: Dict[str, Any],
        configured_batch: int,
        limits: Dict[str, int],
        **estimate_kwargs: Any,
    ) -> Tuple[int, Dict[str, int]]:
        """Find the largest power-of-two batch not exceeding the configured size."""
        candidates = []
        candidate = 1
        while candidate <= configured_batch:
            candidates.append(candidate)
            candidate *= 2
        if configured_batch not in candidates:
            candidates.append(configured_batch)
        for candidate in reversed(candidates):
            estimate = self._estimate_phase(
                phase=phase,
                batch_size=candidate,
                **estimate_kwargs,
            )
            if self._fits(estimate, limits):
                return candidate, estimate
        estimate = self._estimate_phase(
            phase=phase,
            batch_size=1,
            **estimate_kwargs,
        )
        return 1, estimate

    @staticmethod
    def _estimate_disk_bytes(model_bytes: int, phases: Iterable[Dict[str, Any]]) -> int:
        """Estimate best-checkpoint, JSON configuration, and history storage."""
        phases_list = list(phases)
        epochs = sum(int(phase.get("epochs", 10)) for phase in phases_list)
        history_bytes = max(1, len(phases_list)) * max(1, epochs) * 2048
        return int(model_bytes + history_bytes + 262_144)

    def _available_resources(self) -> Dict[str, int]:
        """Read currently available system memory, device memory, and disk."""
        try:
            import psutil

            available_ram = int(psutil.virtual_memory().available)
        except ImportError:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            try:
                available_ram = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
            except (OSError, ValueError):
                available_ram = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        if torch.cuda.is_available() and str(self._wrapper.device).startswith("cuda"):
            available_vram, _ = torch.cuda.mem_get_info(torch.device(self._wrapper.device))
        else:
            available_vram = 0
        workspace_path = Path(self._wrapper.workspace.project_path_resolved)
        available_disk = shutil.disk_usage(workspace_path).free
        return {
            "ram": int(available_ram),
            "vram": int(available_vram),
            "disk": int(available_disk),
        }

    @staticmethod
    def _resolve_limits(
        configured: Dict[str, int | float],
        available: Dict[str, int],
    ) -> Dict[str, int]:
        """Convert relative or absolute limits into byte budgets."""
        unknown = set(configured).difference({"ram", "vram", "disk"})
        if unknown:
            raise_validation_error(
                context_name="ResourceEstimator",
                message=f"Unsupported resource limits: {sorted(unknown)}.",
            )
        limits: Dict[str, int] = {}
        for resource, available_bytes in available.items():
            value = configured.get(resource, 0.8)
            if isinstance(value, bool) or value <= 0:
                raise_validation_error(
                    context_name="ResourceEstimator",
                    message=f"The {resource} limit must be positive.",
                )
            if value <= 1:
                limits[resource] = int(available_bytes * float(value))
            else:
                limits[resource] = int(value)
        return limits

    @staticmethod
    def _fits(estimate: Dict[str, int], limits: Dict[str, int]) -> bool:
        """Return whether one phase estimate fits RAM and available VRAM."""
        ram_fits = estimate["estimated_ram_bytes"] <= limits["ram"]
        vram_required = estimate["estimated_vram_bytes"]
        vram_fits = vram_required == 0 or vram_required <= limits["vram"]
        return ram_fits and vram_fits

    @staticmethod
    def _tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
        """Return storage bytes for a tensor iterable."""
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @classmethod
    def _value_tensor_bytes(cls, value: Any) -> int:
        """Recursively total tensor bytes in model outputs."""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(cls._value_tensor_bytes(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(cls._value_tensor_bytes(item) for item in value)
        return 0
