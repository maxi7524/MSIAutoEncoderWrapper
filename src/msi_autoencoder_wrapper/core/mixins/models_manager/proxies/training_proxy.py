# Heading 1 (Training Proxy Implementation)
## Specialized component managing training execution, loss function registry lookup, and optimization loops

from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Base class and manager imports
from .base_models_manager_proxy import BaseModelsManagerProxy
from .....training.training_manager import TrainingManager
from .....training.criterions.criterions_manager import CriterionsManager
from .....training.resource_estimator import TrainingResourceEstimator

# Centralized utilities imports
from .....utils.logger import get_custom_logger
from .....utils.exceptions import raise_validation_error
from .....utils.printing import present_components_info

if TYPE_CHECKING:
    pass

# Logger initialization
logger = get_custom_logger(__name__)


class TrainingProxy(BaseModelsManagerProxy):
    """
    Proxy component executing optimization passes, criterion management, and learning loop control.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the training manager proxy and triggers dynamic loss function discovery.
        """
        super().__init__(*args, **kwargs)
        
        # Core initialization sequence
        ## Execute criterion lookup to ensure loss functions database is fully indexed
        logger.debug("TrainingProxy: Initiating criterions factory self-discovery sequences.")
        CriterionsManager.discover_criterions()

    # --------------------------------------------------
    # Section: Criterions Discovery
    # --------------------------------------------------

    def get_available_criterions(
        self,
        criterion_type: Optional[str] = None,
        print_return: bool = True,
        return_value: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Queries the central registry factory to extract parameter sheets and docstrings compatible with the active model family.

        :param criterion_type: Optional reconstruction, contrastive, or head filter.
        :type criterion_type: Optional[str]
        :return: Categorized compatible loss-function descriptions.
        :rtype: Optional[Dict[str, Any]]
        """
        # Active state validation
        ## Verify if the parent models manager has selected an architecture layout
        model_type = self.active_model_type
        if not model_type:
            logger.error("Registry lookup rejected: active_model_type descriptor is undefined.")
            raise_validation_error(
                context_name="ModelsManager",
                message="Cannot query available criterions: active_model_type is undefined inside the models manager."
            )

        logger.info("Ferrying database capabilities ledger for active model family target: %s", model_type)
        
        # Factory dispatch
        ## Retrieve list of valid criterions from registry database
        criterion_info = CriterionsManager.get_available_criterions(
            model_type=model_type,
            criterion_type=criterion_type,
        )
        if print_return:
            grouped_info = (
                {criterion_type: criterion_info}
                if criterion_type is not None
                else criterion_info
            )
            for category, category_info in grouped_info.items():
                present_components_info(
                    category_info,
                    title=f"Available {category.title()} Criterions for '{model_type}'",
                    key_label="Criterion",
                    print_return=True,
                    return_value=False,
                )
        return criterion_info if return_value else None

    # --------------------------------------------------
    # Section: Training Optimization Loop Execution
    # --------------------------------------------------

    def clear_training_cache(self) -> None:
        """Clear criterion data cached across phases and consecutive fit calls."""
        self._training_transient_cache.clear()
        logger.info("Cleared the loaded model training transient cache.")

    def fit(self, training_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Launches the backpropagation optimization sequence on the registered dataset and architecture.

        :param training_config: Dictionary specifying the parameters for model optimization.
        :type training_config: Dict[str, Any]
        :return: Training history entries collected across phases and epochs.
        :rtype: List[Dict[str, Any]]
        :raises ValidationError: If active_model or active_dataset is unassigned.
        """
        #TODO - model powininen być zapisywany co epoke, w przypadku gdy jest najlepszy itd. - to powinno byc aktywnie śledzone i pilnowane 
        self._training_config = training_config

        # Heading 1 (Pre-flight Validation Pass)
        ## Verify that critical training components are actively bound to the wrapper context
        active_model = getattr(self._wrapper, "active_model", None)
        active_dataset = getattr(self._wrapper, "active_dataset", None)

        if active_model is None:
            raise_validation_error(
                context_name="ModelsManager",
                message="Cannot start training because no active model is loaded.",
            )

        if active_dataset is None:
            raise_validation_error(
                context_name="ModelsManager",
                message="Cannot start training because no active dataset is loaded.",
            )

        # Heading 2 (Orchestration Engine Dispatch)
        ## Instantiates a temporary TrainingManager instance bound to the master context
        logger.info("Instantiating execution training manager instance.")
        training_orchestrator = TrainingManager(wrapper_ref=self._wrapper)

        execution_history = training_orchestrator.fit(training_config=training_config)
        self._training_history = execution_history

        self.mark_model_trained()

        return execution_history

    def estimate_training_resources(
        self,
        training_config: Dict[str, Any],
        resource_limits: Optional[Dict[str, int | float]] = None,
        auto_adjust_batch_size: bool = False,
        safety_factor: float = 1.25,
    ) -> Dict[str, Any]:
        """Estimate RAM, VRAM, and disk requirements before training.

        :param training_config: Multi-phase trainer configuration.
        :type training_config: Dict[str, Any]
        :param resource_limits: Relative fractions or absolute byte limits.
        :type resource_limits: Optional[Dict[str, int | float]]
        :param auto_adjust_batch_size: Recommend smaller unsafe phase batches.
        :type auto_adjust_batch_size: bool
        :param safety_factor: Reserve multiplier applied to memory estimates.
        :type safety_factor: float
        :return: Detailed estimate and recommended copied configuration.
        :rtype: Dict[str, Any]
        """
        estimator = TrainingResourceEstimator(wrapper_ref=self._wrapper)
        return estimator.estimate(
            training_config=training_config,
            resource_limits=resource_limits,
            auto_adjust_batch_size=auto_adjust_batch_size,
            safety_factor=safety_factor,
        )
