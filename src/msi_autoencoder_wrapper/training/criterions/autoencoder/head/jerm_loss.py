"""JERM with the paper's global modified-spy alternating procedure."""

from __future__ import annotations

import math
import weakref

import torch
import torch.nn.functional as F

from .supervision_masks import supervised_pu_masks
from ...criterions_manager import CriterionsManager
from ...autoencoder_base_criterions import MSIHeadCriterion
from .....data import SpectrumBatch, load_jerm_static_spy_cache


@CriterionsManager.register_criterion("autoencoder", "head", "JERMLoss")
class JERMLoss(MSIHeadCriterion):
    """Joint empirical risk minimization under instance-dependent labeling.

    The criterion follows Rejchel et al.'s Algorithm 1 at epoch granularity.
    A posterior epoch minimizes the observed-label joint likelihood. Its full
    train-pass predictions determine the global modified-spy set ``P_hat``.
    The following propensity epoch minimizes ``H_P_hat``. This resolves the
    posterior/propensity symmetry through the prescribed asymmetric procedure.

    :param optimization_mode: ``alternating_spy`` (the paper algorithm) or
        ``joint`` for the empirical-risk-only ablation.
    :type optimization_mode: str
    :param propensity_selection_weight: Multiplier of ``H_P_hat`` during a
        propensity epoch.
    :type propensity_selection_weight: float
    :param simulated_negative_weight: Optional posterior BCE on declared
        ``N_sim``. This is a project extension, separate from JERM core.
    :type simulated_negative_weight: float

    REMARK: Modified-spy selection is defined over all samples observed in a
    posterior epoch. A JERM phase must use full coverage, not replacement
    sampling, to reproduce the paper's procedure.
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        optimization_mode: str = "alternating_spy",
        propensity_selection_weight: float = 1.0,
        simulated_negative_weight: float = 0.0,
    ) -> None:
        super().__init__(head_id, target_field, class_indices)
        if optimization_mode == "alternating":
            optimization_mode = "alternating_spy"
        if optimization_mode not in {"joint", "alternating_spy"}:
            raise ValueError("optimization_mode must be joint or alternating_spy.")
        if any(
            not math.isfinite(value) or value < 0
            for value in (propensity_selection_weight, simulated_negative_weight)
        ):
            raise ValueError("JERM weights must be finite and nonnegative.")
        self.optimization_mode = optimization_mode
        self.propensity_selection_weight = float(propensity_selection_weight)
        self.simulated_negative_weight = float(simulated_negative_weight)
        self._model_ref: weakref.ReferenceType[torch.nn.Module] | None = None
        self._epoch_records: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._static_source_ids: torch.Tensor | None = None
        self._static_labelled_positive: torch.Tensor | None = None
        self._static_unlabelled: torch.Tensor | None = None
        self._source_id_positions: dict[int, int] = {}
        self._posterior_conditional: torch.Tensor | None = None
        self._posterior_seen: torch.Tensor | None = None
        self._static_spy_source_ids: tuple[torch.Tensor, ...] | None = None
        self._pseudo_ids = torch.empty(0, dtype=torch.long)
        self._pseudo_positive = torch.empty(0, 0, dtype=torch.bool)
        self._alternating_stage = "posterior"
        self.register_buffer("last_inferred_positive_fraction", torch.zeros(()), persistent=False)
        self._config = {
            "head_id": head_id,
            "target_field": target_field,
            "class_indices": self.class_indices,
            "optimization_mode": optimization_mode,
            "propensity_selection_weight": self.propensity_selection_weight,
            "simulated_negative_weight": self.simulated_negative_weight,
        }

    def on_phase_start(self, model, dataset, transient_cache) -> None:
        """Reset modified-spy state before the first posterior epoch.

        :param model: Model optimized during the phase.
        :type model: torch.nn.Module
        :param dataset: Dataset used during the phase.
        :type dataset: MSIBaseDataset
        :param transient_cache: Shared mutable training cache.
        :type transient_cache: Dict[str, Any]
        """
        del transient_cache
        self._model_ref = weakref.ref(model)
        self._epoch_records.clear()
        self._static_source_ids = None
        self._static_labelled_positive = None
        self._static_unlabelled = None
        self._source_id_positions = {}
        self._posterior_conditional = None
        self._posterior_seen = None
        self._static_spy_source_ids = None
        if dataset is not None and self.optimization_mode == "alternating_spy":
            (
                self._static_source_ids,
                self._static_labelled_positive,
                self._static_unlabelled,
                self._static_spy_source_ids,
            ) = load_jerm_static_spy_cache(
                dataset,
                self.target_field,
            )
            self._source_id_positions = {
                int(source_id): position
                for position, source_id in enumerate(self._static_source_ids.tolist())
            }
            self._posterior_conditional = torch.full(
                self._static_labelled_positive.shape,
                torch.nan,
                dtype=torch.float32,
            )  # (N_train, C)
            self._posterior_seen = torch.zeros(
                len(self._static_source_ids), dtype=torch.bool
            )  # (N_train,)
        self._pseudo_ids = torch.empty(0, dtype=torch.long)
        self._pseudo_positive = torch.empty(0, 0, dtype=torch.bool)
        self._alternating_stage = "posterior"
        self.last_inferred_positive_fraction.zero_()

    @staticmethod
    def latent_positive_probability(
        posterior_logits: torch.Tensor,
        propensity_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``P(Y=1 | X, S=0)`` from equation (16)."""
        posterior = posterior_logits.sigmoid()  # (B, C)
        propensity = propensity_logits.sigmoid()  # (B, C)
        observed_positive = posterior * propensity  # (B, C)
        return posterior * (1.0 - propensity) / (1.0 - observed_positive).clamp_min(
            torch.finfo(posterior.dtype).eps
        )  # (B, C)

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return the stage-appropriate JERM objective for one batch."""
        del kwargs
        head_output, targets, availability = self.head_batch(model_outputs, batch_data)
        if head_output.shape != (*targets.shape, 2):
            raise ValueError("JERM head output must have shape (B, C, 2).")
        targets = targets.to(head_output)  # (B, C)
        if availability.shape == targets.shape[:1]:
            availability = availability.unsqueeze(1).expand_as(targets)  # (B, C)
        availability = availability.to(device=head_output.device, dtype=torch.bool)  # (B, C)
        labelled_positive, unlabelled, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets,
            availability,
            self.class_indices,
        )
        posterior_logits = head_output[..., 0]  # (B, C)
        propensity_logits = head_output[..., 1]  # (B, C)
        model = self._model_ref() if self._model_ref is not None else None
        training = model is not None and model.training

        if self.optimization_mode == "alternating_spy" and training and self._alternating_stage == "propensity":
            pseudo_positive = self._lookup_pseudo_positive(batch_data, labelled_positive)  # (B, C)
            loss = posterior_logits.sum() * 0.0  # ()
            if bool(pseudo_positive.any()) and self.propensity_selection_weight > 0:
                propensity_targets = labelled_positive.to(dtype=propensity_logits.dtype)  # (B, C)
                loss = self.propensity_selection_weight * F.binary_cross_entropy_with_logits(
                    propensity_logits[pseudo_positive], propensity_targets[pseudo_positive]
                )  # ()
            return loss

        # Posterior stage: empirical joint risk Q_n(beta, gamma_hat).
        fixed_propensity = (
            propensity_logits.detach()
            if self.optimization_mode == "alternating_spy" and training
            else propensity_logits
        )  # (B, C)
        observed = labelled_positive | unlabelled  # (B, C)
        if bool(observed.any()):
            log_observed_positive = F.logsigmoid(posterior_logits) + F.logsigmoid(
                fixed_propensity
            )  # (B, C)
            observed_positive = log_observed_positive.exp()  # (B, C)
            values = torch.where(
                labelled_positive,
                -log_observed_positive,
                -torch.log1p(
                    -observed_positive.clamp_max(1.0 - torch.finfo(observed_positive.dtype).eps)
                ),
            )  # (B, C)
            loss = values[observed].mean()  # ()
        else:
            loss = posterior_logits.sum() * 0.0  # ()

        if self.simulated_negative_weight > 0 and bool(simulated_negative.any()):
            loss = loss + self.simulated_negative_weight * F.softplus(
                posterior_logits[simulated_negative]
            ).mean()  # ()
        if self.optimization_mode == "alternating_spy" and training:
            conditional = self.latent_positive_probability(
                posterior_logits.detach(), propensity_logits.detach()
            )  # (B, C)
            self._record_posterior_batch(
                batch_data, labelled_positive, unlabelled, conditional
            )
        return loss

    def on_epoch_end(self, model: torch.nn.Module) -> None:
        """Construct global P_hat or switch back to posterior optimization.

        :param model: Model optimized in the completed epoch.
        :type model: torch.nn.Module
        """
        del model
        if self.optimization_mode != "alternating_spy":
            return
        if self._alternating_stage == "posterior":
            self._build_modified_spy_set()
            self._epoch_records.clear()
            self._alternating_stage = "propensity"
        else:
            self._alternating_stage = "posterior"

    def _record_posterior_batch(
        self,
        batch_data,
        labelled_positive: torch.Tensor,
        unlabelled: torch.Tensor,
        conditional_probability: torch.Tensor,
    ) -> None:
        """Store posterior probabilities in the campaign-global source order."""
        sample_ids = (
            batch_data.sample_ids if isinstance(batch_data, SpectrumBatch) else batch_data[0]
        ).detach().to("cpu", dtype=torch.long)  # (B,)
        if self._static_source_ids is not None:
            if self._posterior_conditional is None or self._posterior_seen is None:
                raise RuntimeError("JERM static posterior storage is not initialized.")
            positions = torch.tensor(
                [self._source_id_positions.get(int(sample_id), -1) for sample_id in sample_ids],
                dtype=torch.long,
            )  # (B,)
            if bool((positions < 0).any()):
                raise RuntimeError(
                    "JERM posterior batch contains a source ID outside the campaign train split."
                )
            self._posterior_conditional[positions] = conditional_probability.detach().to(
                "cpu", dtype=torch.float32
            )  # (B, C)
            self._posterior_seen[positions] = True
            return

        # Unit-test fallback for an explicitly supplied spy relation.
        ## Production alternating-spy runs always use the compact static storage above.
        positives = labelled_positive.detach().to("cpu", dtype=torch.bool)  # (B, C)
        unlabelled_cpu = unlabelled.detach().to("cpu", dtype=torch.bool)  # (B, C)
        conditional_cpu = conditional_probability.detach().to("cpu", dtype=torch.float32)  # (B, C)
        for row, sample_id in enumerate(sample_ids.tolist()):
            self._epoch_records[sample_id] = (
                torch.empty(0), positives[row], unlabelled_cpu[row], conditional_cpu[row]
            )

    def _build_modified_spy_set(self) -> None:
        """Build L union SP union A globally according to Algorithm 1."""
        if self._static_source_ids is not None:
            if (
                self._static_labelled_positive is None
                or self._static_unlabelled is None
                or self._posterior_conditional is None
                or self._posterior_seen is None
            ):
                raise RuntimeError("JERM static posterior state is incomplete.")
            if not bool(self._posterior_seen.all()):
                missing = int((~self._posterior_seen).sum().item())
                raise RuntimeError(
                    "JERM posterior epoch must cover the complete train split; "
                    f"{missing} source rows were not observed."
                )
            sample_ids = self._static_source_ids
            labelled_positive = self._static_labelled_positive
            unlabelled = self._static_unlabelled
            conditional = self._posterior_conditional
        elif not self._epoch_records:
            self._pseudo_ids = torch.empty(0, dtype=torch.long)
            self._pseudo_positive = torch.empty(0, 0, dtype=torch.bool)
            return
        else:
            sample_ids = torch.tensor(sorted(self._epoch_records), dtype=torch.long)  # (N,)
            records = [self._epoch_records[int(sample_id)] for sample_id in sample_ids]
            labelled_positive = torch.stack([record[1] for record in records])  # (N, C)
            unlabelled = torch.stack([record[2] for record in records])  # (N, C)
            conditional = torch.stack([record[3] for record in records])  # (N, C)
        pseudo_positive = labelled_positive.clone()  # (N, C)

        # Global modified-spy selection, independently for each molecular task.
        for column in range(labelled_positive.shape[1]):
            labelled_rows = labelled_positive[:, column].nonzero(as_tuple=True)[0]  # (N_L,)
            unlabelled_rows = unlabelled[:, column].nonzero(as_tuple=True)[0]  # (N_U,)
            if len(labelled_rows) == 0 or len(unlabelled_rows) == 0:
                continue
            if self._static_spy_source_ids is None:
                raise RuntimeError(
                    "JERM requires campaign-global nearest-spy precompute when "
                    "running alternating_spy mode."
                )
            static_ids = self._static_spy_source_ids[column]
            spy_rows = torch.searchsorted(sample_ids, static_ids)  # (N_SP,)
            valid_spy = spy_rows < len(sample_ids)
            if bool(valid_spy.any()):
                valid_positions = valid_spy.nonzero(as_tuple=True)[0]
                valid_spy[valid_positions] = (
                    sample_ids[spy_rows[valid_positions]] == static_ids[valid_positions]
                )
            spy_rows = spy_rows[valid_spy]
            if spy_rows.numel() == 0:
                continue
            threshold = conditional[spy_rows, column].min()  # ()
            additional_rows = unlabelled_rows[
                conditional[unlabelled_rows, column] > threshold
            ]  # (N_A_or_SP,)
            pseudo_positive[spy_rows, column] = True
            pseudo_positive[additional_rows, column] = True
        self._pseudo_ids = sample_ids
        self._pseudo_positive = pseudo_positive
        self.last_inferred_positive_fraction = (
            (pseudo_positive & ~labelled_positive).to(torch.float32).mean()
        )
        if self._posterior_seen is not None:
            self._posterior_seen.zero_()

    def _lookup_pseudo_positive(
        self,
        batch_data,
        labelled_positive: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve global P_hat rows for one propensity-training batch."""
        if self._pseudo_ids.numel() == 0:
            return labelled_positive
        sample_ids = (
            batch_data.sample_ids if isinstance(batch_data, SpectrumBatch) else batch_data[0]
        ).detach().to("cpu", dtype=torch.long)  # (B,)
        positions = torch.searchsorted(self._pseudo_ids, sample_ids)  # (B,)
        matched = positions < len(self._pseudo_ids)  # (B,)
        candidate_rows = matched.nonzero(as_tuple=True)[0]  # (B_matched,)
        matched[candidate_rows] = (
            self._pseudo_ids[positions[candidate_rows]] == sample_ids[candidate_rows]
        )
        selected = labelled_positive.detach().to("cpu", dtype=torch.bool)  # (B, C)
        if bool(matched.any()):
            selected[matched] = self._pseudo_positive[positions[matched]]
        return selected.to(device=labelled_positive.device)  # (B, C)
