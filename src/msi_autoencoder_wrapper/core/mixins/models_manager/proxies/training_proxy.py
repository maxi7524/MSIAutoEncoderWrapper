# Heading 1 (Training Proxy Implementation)
## Specialized component managing training execution, loss function registry lookup, and optimization loops

from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Base class and manager imports
from .base_models_manager_proxy import BaseModelsManagerProxy
from .....training.training_manager import TrainingManager
from .....training.criterions.criterions_manager import CriterionsManager

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
        print_return: bool = True,
        return_value: bool = False,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Queries the central registry factory to extract parameter sheets and docstrings compatible with the active model family.

        :return: Map tracking blueprint capabilities, requirements, and descriptions of compatible loss functions.
        :rtype: Dict[str, Dict[str, Any]]
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
        criterion_info = CriterionsManager.get_available_criterions(model_type=model_type)
        return present_components_info(
            criterion_info,
            title=f"Available Criterions for '{model_type}'",
            key_label="Criterion",
            print_return=print_return,
            return_value=return_value,
        )

    # --------------------------------------------------
    # Section: Training Optimization Loop Execution
    # --------------------------------------------------

    def fit(self, training_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Launches the backpropagation optimization sequence on the registered dataset and architecture.

        :param training_config: Dictionary specifying the parameters for model optimization.
        :type training_config: Dict[str, Any]
        :return: Training history entries collected across phases and epochs.
        :rtype: List[Dict[str, Any]]
        :raises ValidationError: If active_model or active_dataset is unassigned.
        """
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

        # Heading 2 (Loss Criterion Cross-Check Validation)
        ## Consult the criteria manager proxy to verify loss selection alignment
        target_loss = None
        if "loss" in training_config:
            target_loss = training_config["loss"]
        elif "criterion" in training_config:
            target_loss = training_config["criterion"]
        elif "phases" in training_config:
            try:
                first_phase = training_config["phases"][0]
                if "criterions" in first_phase:
                    target_loss = list(first_phase["criterions"].keys())[0]
            except (IndexError, AttributeError):
                pass

        if target_loss:
            compatible_criterions = self.get_available_criterions(
                print_return=False,
                return_value=True,
            ) or {}
            if target_loss not in compatible_criterions:
                logger.warning(
                    "Loss criterion '%s' may be incompatible with active model category '%s'.",
                    target_loss,
                    self.active_model_type,
                )

        # Heading 2 (Orchestration Engine Dispatch)
        ## Instantiates a temporary TrainingManager instance bound to the master context
        logger.info("Instantiating execution training manager instance.")
        training_orchestrator = TrainingManager(wrapper_ref=self._wrapper)

        execution_history = training_orchestrator.fit(training_config=training_config)
        self._training_history = execution_history

        ### Post-training model registration hook
        #### Re-attach compiled models to high-level interfaces if active model mixin is present
        if hasattr(self._wrapper, "attach_local_model"):
            logger.debug("Synchronizing local interface wrappers with updated weights.")
            self._wrapper.attach_local_model(torch_model=active_model, model_type=self.active_model_type)

        return execution_history
