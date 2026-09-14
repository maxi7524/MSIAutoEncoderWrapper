"""Taylor-VPU objective with optional EMA-teacher consistency.

The variational term reproduces Zhao et al., ICCV 2023, equation (8): the
unlabelled expectation is expanded as a finite Taylor series around one.  The
P/U term excludes declared simulated negatives; ``N_sim`` may be attached as
an independent BCE term.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from .supervision_masks import supervised_pu_masks
from .taylor_pu_components import simulated_negative_bce, taylor_variational_risk
from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager
from .....data import SpectrumBatch
from .....utils.exceptions import raise_incompatible_interface_error
from .....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@CriterionsManager.register_criterion("autoencoder", "head", "TaylorVariationalPULoss")
class TaylorVariationalPULoss(MSIHeadCriterion):
    """Compute Taylor-VPU on P/U entries and optional BCE on ``N_sim``.

    :param head_id: Model head identifier.
    :type head_id: str
    :param target_field: Binary annotation field.
    :type target_field: str
    :param class_indices: Optional selected target columns.
    :type class_indices: tuple[int, ...] | list[int] | None
    :param order: Positive truncation order of the Taylor series.
    :type order: int
    :param simulated_negative_weight: Weight of BCE on declared ``N_sim``.
    :type simulated_negative_weight: float
    :param consistency_weight: Weight of equation (19)'s symmetric Bernoulli
        KL term. Set zero to use only equation (8).
    :type consistency_weight: float
    :param teacher_momentum: EMA coefficient ``alpha`` for the teacher.
    :type teacher_momentum: float
    :raises ValueError: If a hyperparameter is outside its valid range.

    REMARK: The teacher is a deep copy of the phase model, created after the
    optimizer exists. It is deliberately not optimized and is updated only by
    :meth:`on_optimizer_step`, matching Algorithm 1 in the paper.
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        order: int = 2,
        simulated_negative_weight: float = 0.0,
        consistency_weight: float = 1.0,
        teacher_momentum: float = 0.999,
    ) -> None:
        super().__init__(head_id, target_field, class_indices)
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer.")
        if any(
            not math.isfinite(value) or value < 0
            for value in (simulated_negative_weight, consistency_weight)
        ):
            raise ValueError("Taylor-VPU weights must be finite and nonnegative.")
        if not math.isfinite(teacher_momentum) or not 0 <= teacher_momentum < 1:
            raise ValueError("teacher_momentum must be finite and lie in [0, 1).")
        self.order = order
        self.simulated_negative_weight = simulated_negative_weight
        self.consistency_weight = consistency_weight
        self.teacher_momentum = teacher_momentum
        self._teacher: torch.nn.Module | None = None
        self._config.update(
            order=order,
            simulated_negative_weight=simulated_negative_weight,
            consistency_weight=consistency_weight,
            teacher_momentum=teacher_momentum,
        )

    def on_phase_start(self, model, dataset, transient_cache) -> None:
        """Initialize the non-trainable EMA teacher for one training phase.

        :param model: Student model optimized in the current phase.
        :type model: torch.nn.Module
        :param dataset: Dataset used in the current phase.
        :type dataset: MSIBaseDataset
        :param transient_cache: Shared mutable training cache.
        :type transient_cache: Dict[str, Any]
        """
        del dataset, transient_cache
        if self.consistency_weight == 0:
            self._teacher = None
            return
        self._teacher = copy.deepcopy(model).eval()
        self._teacher.requires_grad_(False)
        logger.info(
            "Initialized Taylor-VPU EMA teacher: momentum=%s consistency_weight=%s.",
            self.teacher_momentum,
            self.consistency_weight,
        )

    def on_optimizer_step(self, model: torch.nn.Module) -> None:
        """Update the EMA teacher after one student optimizer step.

        :param model: Newly updated student model.
        :type model: torch.nn.Module
        """
        if self._teacher is None:
            return
        with torch.no_grad():
            # EMA parameters
            for teacher_parameter, student_parameter in zip(
                self._teacher.parameters(), model.parameters(), strict=True
            ):
                teacher_parameter.lerp_(
                    student_parameter.detach(), 1.0 - self.teacher_momentum
                )
            # Keep non-floating buffers such as counters exactly synchronized.
            for teacher_buffer, student_buffer in zip(
                self._teacher.buffers(), model.buffers(), strict=True
            ):
                if torch.is_floating_point(teacher_buffer):
                    teacher_buffer.lerp_(
                        student_buffer.detach(), 1.0 - self.teacher_momentum
                    )
                else:
                    teacher_buffer.copy_(student_buffer)

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return the Taylor-VPU risk, consistency, and optional ``N_sim`` BCE.

        :return: Differentiable scalar objective.
        :rtype: torch.Tensor
        """
        del kwargs
        logits, targets, availability = self.multilabel_batch(model_outputs, batch_data)
        positives, unlabelled, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets,
            availability,
            self.class_indices,
        )
        active = (positives.sum(dim=0) > 0) & (unlabelled.sum(dim=0) > 0)  # (C,)
        loss = taylor_variational_risk(logits, positives, unlabelled, self.order)  # ()
        if self.consistency_weight > 0 and bool(active.any()):
            loss = loss + self.consistency_weight * self._consistency_loss(
                logits,
                positives | unlabelled,
                batch_data,
            )  # ()

        # Separate reliable-negative extension; N_sim never changes the P/U risk.
        if self.simulated_negative_weight > 0 and bool(simulated_negative.any()):
            loss = loss + self.simulated_negative_weight * simulated_negative_bce(
                logits, simulated_negative
            )  # ()
        return loss

    def _consistency_loss(self, logits, active, batch_data) -> torch.Tensor:
        """Calculate equation (19)'s symmetric KL for the EMA teacher.

        :param logits: Student binary logits, shape ``(B, C)``.
        :type logits: torch.Tensor
        :param active: Entries participating in P/U training, shape ``(B, C)``.
        :type active: torch.Tensor
        :param batch_data: Batch containing the model input spectra.
        :type batch_data: Tuple[torch.Tensor, ...]
        :return: Mean symmetric Bernoulli KL over active entries.
        :rtype: torch.Tensor
        :raises IncompatibleInterfaceError: If lifecycle setup or teacher output
            does not match the configured student head.
        """
        if self._teacher is None:
            raise_incompatible_interface_error(
                "TaylorVariationalPULoss",
                "EMA consistency requires on_phase_start before the first batch.",
            )
        spectra = (
            batch_data.model_input()
            if isinstance(batch_data, SpectrumBatch)
            else batch_data[1]
        ).to(logits)  # (B, M)
        with torch.no_grad():
            teacher_output = self._teacher(spectra)
            if self.output_key not in teacher_output:
                raise_incompatible_interface_error(
                    "TaylorVariationalPULoss",
                    f"EMA teacher output is missing '{self.output_key}'.",
                )
            teacher_logits = teacher_output[self.output_key][: logits.shape[0]]  # (B, C)
        if teacher_logits.shape != logits.shape:
            raise_incompatible_interface_error(
                "TaylorVariationalPULoss",
                "EMA teacher logits must have the same [B, C] shape as student logits.",
            )
        epsilon = torch.finfo(logits.dtype).eps
        student_probability = logits.sigmoid().clamp(epsilon, 1.0 - epsilon)  # (B, C)
        teacher_probability = teacher_logits.to(logits).sigmoid().clamp(
            epsilon, 1.0 - epsilon
        )  # (B, C)
        forward_kl = (
            teacher_probability * (teacher_probability.log() - student_probability.log())
            + (1.0 - teacher_probability)
            * ((1.0 - teacher_probability).log() - (1.0 - student_probability).log())
        )  # (B, C)
        reverse_kl = (
            student_probability * (student_probability.log() - teacher_probability.log())
            + (1.0 - student_probability)
            * ((1.0 - student_probability).log() - (1.0 - teacher_probability).log())
        )  # (B, C)
        return (forward_kl + reverse_kl)[active].mean()  # ()
