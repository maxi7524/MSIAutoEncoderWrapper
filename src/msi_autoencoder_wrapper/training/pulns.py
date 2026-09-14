"""Exact episode mathematics for PULNS negative-sample selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PULNSEpisode:
    """One selector trajectory for all valid molecular tasks."""

    selected_negative: torch.Tensor
    log_probabilities: tuple[torch.Tensor, ...]
    intermediate_rewards: tuple[torch.Tensor, ...]
    class_indices: tuple[int, ...]


def pulns_intermediate_reward(
    positive_probability: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    """Return PULNS's clipped action reward from equation (paper page 4).

    :param positive_probability: Classifier positive probability, shape ``(...)``.
    :type positive_probability: torch.Tensor
    :param selected: Boolean action; true selects a reliable negative.
    :type selected: torch.Tensor
    :return: Clipped intermediate reward, shape ``(...)``.
    :rtype: torch.Tensor
    """
    probability = positive_probability.clamp(
        torch.finfo(positive_probability.dtype).eps,
        1.0 - torch.finfo(positive_probability.dtype).eps,
    )  # (...,)
    select_reward = ((1.0 - probability) / probability).log()  # (...,)
    reject_reward = (probability / (1.0 - probability)).log()  # (...,)
    return torch.where(selected, select_reward, reject_reward).clamp(-1.0, 1.0)  # (...,)


def pulns_reinforce_loss(
    episode: PULNSEpisode,
    terminal_rewards: torch.Tensor,
    terminal_weight: float = 1.0,
    discount: float = 1.0,
) -> torch.Tensor:
    """Return the negative REINFORCE objective from the PULNS update rule.

    :param episode: Sampled selector trajectory.
    :type episode: PULNSEpisode
    :param terminal_rewards: ``z^(l) - b^(l)`` per class, shape ``(C,)``.
    :type terminal_rewards: torch.Tensor
    :param terminal_weight: PULNS coefficient alpha.
    :type terminal_weight: float
    :param discount: Intermediate-reward discount beta in ``[0, 1]``.
    :type discount: float
    :return: Negative objective to minimize, scalar.
    :rtype: torch.Tensor
    """
    if not 0 <= discount <= 1:
        raise ValueError("discount must belong to [0, 1].")
    if terminal_weight < 0:
        raise ValueError("terminal_weight must be nonnegative.")
    terms: list[torch.Tensor] = []
    for log_probabilities, rewards, column in zip(
        episode.log_probabilities,
        episode.intermediate_rewards,
        episode.class_indices,
        strict=True,
    ):
        terminal = terminal_weight * terminal_rewards[column].to(log_probabilities)  # ()
        future = terminal
        values: list[torch.Tensor] = []
        for reward in reversed(rewards.unbind()):
            future = reward + discount * (future - terminal) + terminal  # ()
            values.append(future)
        returns = torch.stack(list(reversed(values)))  # (N_U_class,)
        terms.append(-(returns.detach() * log_probabilities).sum())  # ()
    if not terms:
        return terminal_rewards.sum() * 0.0  # ()
    return torch.stack(terms).mean()  # ()


class PULNSSelectorController:
    """Generate PULNS trajectories and classifier P/N objectives.

    The caller supplies ``selector_logits`` from :class:`PULNSHead`. A
    training procedure must evaluate its terminal reward on held-out P/N
    validation data and then use :func:`pulns_reinforce_loss` to update the
    selector. This prevents treating U as validation N.
    """

    def sample_episode(
        self,
        representations: torch.Tensor,
        classifier_logits: torch.Tensor,
        positives: torch.Tensor,
        unlabelled: torch.Tensor,
        selector_logits,
        *,
        deterministic: bool = False,
    ) -> PULNSEpisode:
        """Sample sequential selection actions for every class with P and U."""
        selected_negative = torch.zeros_like(unlabelled)  # (B, C)
        log_probabilities: list[torch.Tensor] = []
        intermediate_rewards: list[torch.Tensor] = []
        class_indices: list[int] = []
        probabilities = classifier_logits.sigmoid().detach()  # (B, C)
        for column in range(classifier_logits.shape[1]):
            positive_rows = positives[:, column].nonzero(as_tuple=True)[0]  # (N_P,)
            unlabelled_rows = unlabelled[:, column].nonzero(as_tuple=True)[0]  # (N_U,)
            if len(positive_rows) == 0 or len(unlabelled_rows) == 0:
                continue
            positive_context = representations[positive_rows].mean(dim=0)  # (D,)
            fallback_context = representations[unlabelled_rows].mean(dim=0)  # (D,)
            selected_sum = torch.zeros_like(fallback_context)  # (D,)
            selected_count = 0
            class_log_probabilities: list[torch.Tensor] = []
            class_rewards: list[torch.Tensor] = []
            for row in unlabelled_rows:
                negative_context = selected_sum / selected_count if selected_count else fallback_context  # (D,)
                state = torch.cat((representations[row], negative_context, positive_context), dim=0)  # (3D,)
                probability = selector_logits(state.unsqueeze(0)).sigmoid().squeeze(0)  # ()
                action = probability >= 0.5 if deterministic else torch.bernoulli(probability).to(torch.bool)  # ()
                class_log_probabilities.append(
                    torch.where(action, probability, 1.0 - probability).clamp_min(
                        torch.finfo(probability.dtype).eps
                    ).log()
                )
                class_rewards.append(pulns_intermediate_reward(probabilities[row, column], action))
                if bool(action):
                    selected_negative[row, column] = True
                    selected_sum = selected_sum + representations[row]
                    selected_count += 1
            log_probabilities.append(torch.stack(class_log_probabilities))  # (N_U,)
            intermediate_rewards.append(torch.stack(class_rewards))  # (N_U,)
            class_indices.append(column)
        return PULNSEpisode(
            selected_negative=selected_negative,
            log_probabilities=tuple(log_probabilities),
            intermediate_rewards=tuple(intermediate_rewards),
            class_indices=tuple(class_indices),
        )

    @staticmethod
    def classifier_loss(
        classifier_logits: torch.Tensor,
        positives: torch.Tensor,
        selected_negative: torch.Tensor,
        simulated_negative: torch.Tensor | None = None,
        simulated_negative_weight: float = 0.0,
    ) -> torch.Tensor:
        """Return PULNS P/N-selected loss plus optional separate N_sim BCE."""
        selected = positives | selected_negative  # (B, C)
        labels = positives.to(dtype=classifier_logits.dtype)  # (B, C)
        loss = classifier_logits.sum() * 0.0  # ()
        if bool(selected.any()):
            loss = F.binary_cross_entropy_with_logits(classifier_logits[selected], labels[selected])  # ()
        if simulated_negative is not None and simulated_negative_weight > 0 and bool(simulated_negative.any()):
            loss = loss + simulated_negative_weight * F.softplus(classifier_logits[simulated_negative]).mean()  # ()
        return loss
