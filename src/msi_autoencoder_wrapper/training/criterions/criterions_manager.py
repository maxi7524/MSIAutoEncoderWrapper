"""
Central orchestration factory and composite loss builder for managing multi-component MSI criteria.
"""

import inspect
from typing import Type, Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn

from ...utils.logger import get_custom_logger
from ...utils.exceptions import raise_validation_error
from ...utils.module_search import discover_modules
from ...utils.validators import resolve_component, validate_subclass
from ...utils.printing import extract_component_signatures
from .base_criterion import (
    MSIBaseCriterion,
)

from .autoencoder_base_criterions import (
    MSIContrastiveCriterion,
    MSIHeadCriterion,
    MSIReconstructionCriterion,
)

# Logger initialization
logger = get_custom_logger(__name__)


class CompositeLoss(nn.Module):
    """
    Weighted compound execution module bundling individual sub-criteria blocks into a singular pass.
    """

    def __init__(self, loss_functions: Dict[str, MSIBaseCriterion], weights: Dict[str, float]) -> None:
        """
        Initializes the joint optimization execution loss block.
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.weights = weights

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the weighted linear combination sum across all initialized sub-loss components.

        :param model_outputs: Evaluation tensor tokens maps derived from the active architecture.
        :type model_outputs: Dict[str, torch.Tensor]
        :param batch_data: Target variable matrices provided by the active data loader.
        :type batch_data: Tuple[torch.Tensor, ...]
        :return: Jointly accumulated graph loss tensor paired with an isolated scalar values logging map.
        :rtype: Tuple[torch.Tensor, Dict[str, float]]
        """
        if not self.loss_functions:
            raise_validation_error(
                context_name="CompositeLoss",
                message=(
                    "No loss functions were configured. Ensure that 'criterions' "
                    "contains at least one registered implementation."
                ),
            )

        total_loss = torch.tensor(0.0, device=batch_data[1].device if len(batch_data) > 1 else torch.device("cpu"))
        loss_logs: Dict[str, float] = {}

        # Linear aggregation loop
        for token, loss_fn in self.loss_functions.items():
            component_loss = loss_fn(model_outputs, batch_data)
            weighted_loss = component_loss * self.weights.get(token, 1.0)
            total_loss = total_loss + weighted_loss
            loss_logs[token] = component_loss.item()

        loss_logs["total_loss"] = total_loss.item()
        return total_loss, loss_logs


class CriterionsManager:
    """
    Factory engine coordinating registration, reflection mapping, and orchestration across mathematical loss metrics.
    """

    # Registry layout: [model type][criterion type][criterion name].
    _REGISTRY: Dict[str, Dict[str, Dict[str, Type[MSIBaseCriterion]]]] = {}
    _CATEGORY_BASES: Dict[str, Type[MSIBaseCriterion]] = {
        "reconstruction": MSIReconstructionCriterion,
        "contrastive": MSIContrastiveCriterion,
        "head": MSIHeadCriterion,
    }

    # --------------------------------------------------
    # Section: Criterions registration
    # --------------------------------------------------

    @classmethod
    def register_criterion(cls, model_type: str, criterion_type: str, name: str) -> Any:
        """
        Decorator factory to register a specific loss implementation under a targeted model family scope.

        :param model_type: Parent family identifier token defining compatibility restrictions (e.g. 'autoencoder').
        :type model_type: str
        :param criterion_type: Execution category such as ``reconstruction``.
        :type criterion_type: str
        :param name: Unique lookup token string representing the loss strategy.
        :type name: str
        :return: Inner modifier decorator closure.
        :rtype: Any
        """
        def decorator(subclass: Type[MSIBaseCriterion]) -> Type[MSIBaseCriterion]:
            expected_base = cls._CATEGORY_BASES.get(criterion_type)
            if expected_base is None:
                raise_validation_error(
                    context_name="CriterionRegistry",
                    message=(
                        f"Unsupported criterion type '{criterion_type}'. "
                        f"Available types: {sorted(cls._CATEGORY_BASES)}."
                    ),
                )
            validate_subclass(subclass, expected_base, "CriterionRegistry")
            model_registry = cls._REGISTRY.setdefault(model_type, {})
            model_registry.setdefault(criterion_type, {})[name] = subclass
            logger.debug(
                "Registered '%s' criterion '%s' for model family '%s'.",
                criterion_type,
                name, 
                model_type
            )
            return subclass
        return decorator

    @classmethod
    def get_available_criterions(
        cls,
        model_type: str,
        criterion_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Queries the database registry to compile structural definitions and parameter sheets for a given model type.

        :param model_type: Master model architecture type token string filter.
        :type model_type: str
        :param criterion_type: Optional execution category filter.
        :type criterion_type: Optional[str]
        :return: Categorized component descriptions or one selected category.
        :rtype: Dict[str, Any]
        """
        if model_type not in cls._REGISTRY:
            logger.debug("No custom loss criteria found or registered for model type: %s", model_type)
            return {}

        model_registry = cls._REGISTRY[model_type]
        if criterion_type is not None:
            if criterion_type not in model_registry:
                raise_validation_error(
                    context_name="CriterionsManager",
                    message=(
                        f"No '{criterion_type}' criteria are registered for model "
                        f"family '{model_type}'."
                    ),
                )
            return extract_component_signatures(model_registry[criterion_type])
        return {
            category: extract_component_signatures(registry)
            for category, registry in model_registry.items()
        }
    
    @classmethod
    def discover_criterions(cls, package_path: Optional[List[str]] = None, package_name: Optional[str] = None) -> None:
        """
        Recursively scans package layout structures to dynamically load and register concrete loss criterions.
        """
        del package_path
        package_root = package_name or __package__
        discover_modules(package_root)

    # --------------------------------------------------
    # Section: Combining Loss function
    # --------------------------------------------------

    @classmethod
    def build_composite_loss(cls, model_type: str, loss_setup: Dict[str, Any]) -> CompositeLoss:
        """
        Dynamically instantiates and chains separate loss criteria blocks into a single CompositeLoss object.

        :param model_type: Targeted model category identifier token enforcing dictionary lookup alignments.
        :type model_type: str
        :param loss_setup: Phase setup block mapping loss names to their specific weight values and initial parameters.
        :type loss_setup: Dict[str, Any]
        :return: Completed weighted joint loss computational function container.
        :rtype: CompositeLoss
        """
        if model_type not in cls._REGISTRY:
            raise_validation_error(
                context_name="CriterionsManager",
                message=f"No criterion registry exists for model family '{model_type}'.",
            )

        instantiated_losses: Dict[str, MSIBaseCriterion] = {}
        loss_weights: Dict[str, float] = {}

        for criterion_type, typed_setup in cls._normalize_loss_setup(
            model_type,
            loss_setup,
        ).items():
            registry = cls._REGISTRY[model_type].get(criterion_type)
            if registry is None:
                raise_validation_error(
                    context_name="CriterionsManager",
                    message=(
                        f"No '{criterion_type}' criterion registry exists for "
                        f"model family '{model_type}'."
                    ),
                )
            for token, config_ledger in typed_setup.items():
                cls._build_loss_component(
                    model_type=model_type,
                    criterion_type=criterion_type,
                    token=token,
                    config_ledger=config_ledger,
                    registry=registry,
                    instantiated_losses=instantiated_losses,
                    loss_weights=loss_weights,
                )

        return CompositeLoss(loss_functions=instantiated_losses, weights=loss_weights)

    @classmethod
    def _build_loss_component(
        cls,
        model_type: str,
        criterion_type: str,
        token: str,
        config_ledger: Any,
        registry: Dict[str, Type[MSIBaseCriterion]],
        instantiated_losses: Dict[str, MSIBaseCriterion],
        loss_weights: Dict[str, float],
    ) -> None:
        """Build and record one typed criterion component."""
        if isinstance(config_ledger, dict):
            target = config_ledger.get("target", token)
            weight = config_ledger.get("weight", 1.0)
            params = config_ledger.get("params", {})
        else:
            target = config_ledger
            weight = 1.0
            params = {}

        metric_token = token
        if metric_token in instantiated_losses:
            metric_token = f"{criterion_type}_{token}"
        logger.info(
            "Assembling '%s' objective '%s' with weight %s.",
            criterion_type,
            metric_token,
            weight,
        )
        instantiated_losses[metric_token] = resolve_component(
            target=target,
            registry=registry,
            component_type=f"{model_type}.{criterion_type}.criterion",
            expected_type=CriterionsManager._CATEGORY_BASES[criterion_type],
            **params,
        )
        loss_weights[metric_token] = float(weight)

    @classmethod
    def _normalize_loss_setup(
        cls,
        model_type: str,
        loss_setup: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Return the categorized criterion configuration format.

        A flat mapping is accepted only when every target can be resolved to one
        unambiguous category. This preserves existing configurations while the
        public API moves to explicit execution categories.
        """
        explicit_categories = set(loss_setup).intersection(cls._CATEGORY_BASES)
        if explicit_categories:
            unknown = set(loss_setup).difference(cls._CATEGORY_BASES)
            if unknown:
                raise_validation_error(
                    context_name="CriterionsManager",
                    message=(
                        "Categorized criterion configuration cannot mix category "
                        f"keys with flat entries: {sorted(unknown)}."
                    ),
                )
            return loss_setup

        normalized: Dict[str, Dict[str, Any]] = {}
        for token, config_ledger in loss_setup.items():
            if isinstance(config_ledger, dict):
                target = config_ledger.get("target", token)
            else:
                target = config_ledger
            matching_categories = [
                category
                for category, registry in cls._REGISTRY.get(model_type, {}).items()
                if (
                    not isinstance(target, str)
                    and (
                        isinstance(target, cls._CATEGORY_BASES[category])
                        or (
                            inspect.isclass(target)
                            and issubclass(target, cls._CATEGORY_BASES[category])
                        )
                    )
                )
                or (isinstance(target, str) and target in registry)
            ]
            if len(matching_categories) != 1:
                raise_validation_error(
                    context_name="CriterionsManager",
                    message=(
                        f"Criterion '{token}' cannot be assigned to one execution "
                        "category. Use the categorized configuration format."
                    ),
                )
            normalized.setdefault(matching_categories[0], {})[token] = config_ledger
        return normalized
