"""
Module providing a dedicated user-facing proxy interface for managing training criteria.
"""

from typing import Any, Dict
from ....training.criterions.criterions_manager import CriterionsManager
from ....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


class TrainingCriterionsProxy:
    """
    Client-facing execution proxy exposing dynamic query methods for loss functions tied to active model configurations.
    """

    def __init__(self, manager_ref: Any) -> None:
        """
        Initializes the criteria operations boundary proxy layer.

        :param manager_ref: Pointer back to the live coordinating ModelsManagerProxy instance.
        :type manager_ref: Any
        """
        self._manager_ref = manager_ref
        logger.debug("TrainingCriterionsProxy proxy interface successfully anchored.")

    def get_available(self) -> Dict[str, Dict[str, Any]]:
        """
        Queries the central registry factory to extract parameter sheets and docstrings compatible with the active model family.

        :return: Map tracking blueprint capabilities, requirements and descriptions of compatible loss functions.
        :rtype: Dict[str, Dict[str, Any]]
        :raises ValueError: If active_model_type is currently undefined in the models manager.
        """
        # Heading 1 (Active Model Family Query Validation)
        ## Extract the current active model type descriptor token from the bound models manager reference
        model_type = getattr(self._manager_ref, "active_model_type", None)
        if not model_type:
            logger.error("Registry lookup rejected: active_model_type descriptor is undefined.")
            raise ValueError("Cannot query available criterions: active_model_type is undefined inside the models manager.")

        logger.info("Ferrying database capabilities ledger for active model family target: %s", model_type)
        
        ## Heading 2 (Factory Delegation Routing Pass)
        ### Fetch parameters specifications and documentation strings using the central registration factory manager
        return CriterionsManager.get_available_criterions(model_type=model_type)