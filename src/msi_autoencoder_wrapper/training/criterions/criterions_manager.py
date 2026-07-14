"""
Central orchestration factory and composite loss builder for managing multi-component MSI criteria.
"""

import inspect
from typing import Type, Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn

from ...utils.logger import get_custom_logger
from .base_criterion import MSIBaseCriterion

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

    # Multi-level structural criterion database layout mapping [model_type][criterion_name] to concrete classes
    _REGISTRY: Dict[str, Dict[str, Type[MSIBaseCriterion]]] = {}

    # --------------------------------------------------
    # Section: Criterions registration
    # --------------------------------------------------

    @classmethod
    def register_criterion(cls, model_type: str, name: str) -> Any:
        """
        Decorator factory to register a specific loss implementation under a targeted model family scope.

        :param model_type: Parent family identifier token defining compatibility restrictions (e.g. 'autoencoder').
        :type model_type: str
        :param name: Unique lookup token string representing the loss strategy.
        :type name: str
        :return: Inner modifier decorator closure.
        :rtype: Any
        """
        def decorator(subclass: Type[MSIBaseCriterion]) -> Type[MSIBaseCriterion]:
            if not issubclass(subclass, MSIBaseCriterion):
                raise TypeError(f"Class '{subclass.__name__}' must inherit from MSIBaseCriterion.")
            if model_type not in cls._REGISTRY:
                cls._REGISTRY[model_type] = {}
            cls._REGISTRY[model_type][name] = subclass
            logger.debug(
                "Successfully registered loss criterion '%s' for model family '%s'.", 
                name, 
                model_type
            )
            return subclass
        return decorator

    @classmethod
    def get_available_criterions(cls, model_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Queries the database registry to compile structural definitions and parameter sheets for a given model type.

        :param model_type: Master model architecture type token string filter.
        :type model_type: str
        :return: Blueprint map tracking parameters, signatures, and docstrings of compatible loss functions.
        :rtype: Dict[str, Dict[str, Any]]
        """
        if model_type not in cls._REGISTRY:
            logger.debug("No custom loss criteria found or registered for model type: %s", model_type)
            return {}

        available_blueprints: Dict[str, Dict[str, Any]] = {}
        for name, class_ref in cls._REGISTRY[model_type].items():
            constructor_signature = inspect.signature(class_ref.__init__)
            parameters_map = {
                p_name: p_obj.default if p_obj.default != inspect.Parameter.empty else "Required"
                for p_name, p_obj in constructor_signature.parameters.items()
                if p_name != "self"
            }
            available_blueprints[name] = {
                "docstring": class_ref.__doc__,
                "parameters": parameters_map
            }
        return available_blueprints
    
    @classmethod
    def discover_criterions(cls, package_path: Optional[List[str]] = None, package_name: Optional[str] = None) -> None:
        """
        Recursively scans package layout structures to dynamically load and register concrete loss criterions.
        """
        import os
        import sys
        import pkgutil
        import importlib

        if package_path is None or package_name is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            package_path = [base_dir]
            
            current_module = cls.__module__
            if "." in current_module:
                package_name = ".".join(current_module.split(".")[:-1])
            else:
                package_name = current_module

        logger.debug("Starting recursive criterions discovery for: %s", package_name)

        # Walk package structures
        ## Convert all path elements to explicit strings to ensure compatibility across platform boundaries
        str_paths = [str(p) for p in package_path]
        for _, module_name, is_pkg in pkgutil.walk_packages(str_paths, package_name + "."):
            ### Explicit boundary enforcement to bypass temporary files or single private modules
            if module_name.split(".")[-1].startswith("_"):
                continue
                
            if module_name not in sys.modules:
                logger.debug("Auto-discovery framework importing loss module: %s", module_name)
                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    logger.error("Failed to dynamically import criterion module '%s': %s", module_name, str(e), exc_info=True)

    # --------------------------------------------------
    # Section: Combining Loss function
    # --------------------------------------------------

    @classmethod
    def build_composite_loss(cls, model_type: str, loss_setup: Dict[str, Dict[str, Any]]) -> CompositeLoss:
        """
        Dynamically instantiates and chains separate loss criteria blocks into a single CompositeLoss object.

        :param model_type: Targeted model category identifier token enforcing dictionary lookup alignments.
        :type model_type: str
        :param loss_setup: Phase setup block mapping loss names to their specific weight values and initial parameters.
        :type loss_setup: Dict[str, Dict[str, Any]]
        :return: Completed weighted joint loss computational function container.
        :rtype: CompositeLoss
        """
        if model_type not in cls._REGISTRY:
            raise KeyError(f"Cannot compile loss functions: No registry ledger exists for model family '{model_type}'.")

        instantiated_losses: Dict[str, MSIBaseCriterion] = {}
        loss_weights: Dict[str, float] = {}

        for token, config_ledger in loss_setup.items():
            if token not in cls._REGISTRY[model_type]:
                raise KeyError(f"Loss criterion '{token}' is unregistered or incompatible with model family '{model_type}'.")

            criterion_class = cls._REGISTRY[model_type][token]
            weight = config_ledger.get("weight", 1.0)
            params = config_ledger.get("params", {})

            logger.info("Assembling objective sub-criterion component: Name='%s' with assigned Weight=%s", token, weight)
            instantiated_losses[token] = criterion_class(**params)
            loss_weights[token] = float(weight)

        return CompositeLoss(loss_functions=instantiated_losses, weights=loss_weights)