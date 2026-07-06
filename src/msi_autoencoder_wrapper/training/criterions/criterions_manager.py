from typing import Type, Dict, Any, Union, Tuple
import torch
import torch.nn as nn

# Purely relative imports reflecting the new structure
from ...utils.logger import get_custom_logger
from ...models.datasets.strategies.base_dataset import MSIBaseDataset
from .base_criterion import MSIBaseCriterion

# Telemetry tracking setup
logger = get_custom_logger(__name__)


class CriterionsManager:
    """
    Central registration factory for managing individual and aggregate MSI loss strategies.
    """

    _REGISTRY: Dict[str, Type[MSIBaseCriterion]] = {}

    @classmethod
    def register_criterion(cls, name: str) -> Any:
        """
        Decorator factory to register a concrete MSIBaseCriterion implementation math block.

        :param name: Unique registration lookup string for the loss strategy.
        :type name: str
        :return: Inner decorator closure wrapping the target class.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseCriterion]) -> Type[MSIBaseCriterion]:
            if not issubclass(subclass, MSIBaseCriterion):
                raise TypeError(f"Class '{subclass.__name__}' must inherit from MSIBaseCriterion.")
            cls._REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def build_composite_loss(cls, loss_setup: dict[str, dict[str, Any]]) -> "CompositeLoss":
        """
        Compiles separate math criteria configurations into a single weighted CompositeLoss engine.

        :param loss_setup: Mapping detailing target criterion names, optimization weights, and parameters.
        :type loss_setup: dict
        :return: Validated integrated CompositeLoss instance module.
        :rtype: CompositeLoss
        """
        resolved_components: Dict[str, MSIBaseCriterion] = {}
        component_weights: Dict[str, float] = {}

        for token, config in loss_setup.items():
            if token not in cls._REGISTRY:
                raise KeyError(f"Criterion token '{token}' not found within registries.")
            
            criterion_instance = cls._REGISTRY[token](**config.get("params", {}))
            resolved_components[token] = criterion_instance
            component_weights[token] = float(config.get("weight", 1.0))

        compiled_loss = CompositeLoss(components=resolved_components, weights=component_weights)
        logger.info("Composite mathematical criteria compiled and balanced successfully.")
        return compiled_loss


class CompositeLoss(nn.Module):
    """
    Aggregator module managing multi-component loss orchestration and optimization routing.

    This class sums sub-criterion loss items using configurable linear weights, while 
    dynamically exposing aggregate information dependencies to reduce framework overhead.
    """

    def __init__(self, components: Dict[str, MSIBaseCriterion], weights: Dict[str, float]) -> None:
        """
        Initializes the joint criterion container module.

        :param components: Map binding unique string identifiers to active MSIBaseCriterion objects.
        :type components: dict
        :param weights: Map binding identical token tags to floating-point multiplier coefficients.
        :type weights: dict
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict(components)
        self.weights = weights

        # Aggregate logical properties execution mapping
        self._requires_reconstruction = any(fn.requires_reconstruction for fn in self.loss_functions.values())
        self._requires_projection = any(fn.requires_projection for fn in self.loss_functions.values())

    def REQUIRED_SETUP(self, dataset: MSIBaseDataset) -> None:
        """Triggers the pre-computation sequences sequentially across all wrapped criteria."""
        for fn in self.loss_functions.values():
            fn.REQUIRED_SETUP(dataset)

    @property
    def requires_reconstruction(self) -> bool:
        """Reports aggregate dependency constraints regarding spectral reconstruction layers."""
        return self._requires_reconstruction

    @property
    def requires_projection(self) -> bool:
        """Reports aggregate dependency constraints regarding contrastive projection heads."""
        return self._requires_projection

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the joint structural loss sum and extracts individual logs.

        :param model_outputs: Active computed tensor tokens collection derived from the architecture graph.
        :type model_outputs: dict
        :param batch_data: Aligned target data arrays provided by the DataLoader sequence.
        :type batch_data: tuple
        :return: Matched tuple holding the total accumulated graph loss tensor and an isolated metrics logs map.
        :rtype: tuple(torch.Tensor, dict)
        """
        total_loss = torch.tensor(0.0, device=batch_data[1].device)
        loss_logs: Dict[str, float] = {}

        for token, loss_fn in self.loss_functions.items():
            component_loss = loss_fn(model_outputs, batch_data)
            weighted_loss = component_loss * self.weights[token]
            total_loss = total_loss + weighted_loss
            loss_logs[token] = component_loss.item()

        loss_logs["total_loss"] = total_loss.item()
        return total_loss, loss_logs