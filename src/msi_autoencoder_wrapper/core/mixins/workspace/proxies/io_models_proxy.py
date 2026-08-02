"""User-facing coordination API for model persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .base_workspace_proxy import BaseWorkspaceProxy
from .getters_and_setters_proxy import GLOBAL_CONTEXT_KEY
from .....utils.exceptions import raise_validation_error
from .....utils.logger import get_custom_logger
from ..model_store import ModelStore

logger = get_custom_logger(__name__)


class IoModelsProxy(BaseWorkspaceProxy):
    """Coordinate active contexts and delegate filesystem work to :class:`ModelStore`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model persistence service."""
        super().__init__(*args, **kwargs)
        self._model_store = ModelStore(
            workspace_root=self.project_path_resolved,
            layout=self._layout,
        )

    def save_config_json(
        self,
        img_name: str,
        model_name: str,
        config_dict: Dict[str, Any],
    ) -> None:
        """Save a consolidated configuration as JSON.

        :param img_name: Image key or ``global`` model context.
        :type img_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param config_dict: Consolidated portable model configuration.
        :type config_dict: Dict[str, Any]
        """
        self._model_store.save_config(img_name, model_name, config_dict)

    def load_config_json(self, img_name: str, model_name: str) -> Dict[str, Any]:
        """Load a consolidated JSON model configuration.

        :param img_name: Image key or ``global`` model context.
        :type img_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :return: Consolidated portable model configuration.
        :rtype: Dict[str, Any]
        """
        return self._model_store.load_config(img_name, model_name)

    def save_model_weights(
        self,
        img_name: str,
        model_name: str,
        state_dict: Dict[str, Any],
    ) -> None:
        """Save a model state dictionary.

        :param img_name: Image key or ``global`` model context.
        :type img_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param state_dict: Model weight state dictionary.
        :type state_dict: Dict[str, Any]
        """
        self._model_store.save_weights(img_name, model_name, state_dict)

    def load_model_weights(self, img_name: str, model_name: str) -> Dict[str, Any]:
        """Load a model state dictionary on CPU.

        :param img_name: Image key or ``global`` model context.
        :type img_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :return: Model weight state dictionary.
        :rtype: Dict[str, Any]
        """
        return self._model_store.load_weights(img_name, model_name)

    def save_history(
        self,
        img_name: str,
        model_name: str,
        history_dict: Any,
    ) -> None:
        """Save training history as JSON.

        :param img_name: Image key or ``global`` model context.
        :type img_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param history_dict: Training history dictionary or sequence.
        :type history_dict: Any
        """
        self._model_store.save_history(img_name, model_name, history_dict)

    def save_model(
        self,
        img_name: Optional[str] = None,
        model_name: Optional[str] = None,
        history: Any = None,
    ) -> Path:
        """Save the active model as portable JSON configuration plus weights.

        :param img_name: Storage context. Uses the active image when omitted.
        :type img_name: Optional[str]
        :param model_name: Model name. Uses the loaded-model context when omitted.
        :type model_name: Optional[str]
        :param history: Optional history override. Uses the model manager history when omitted.
        :type history: Any
        :return: Directory containing the saved model artifacts.
        :rtype: pathlib.Path
        :raises ValidationError: If no active model or required context is available.
        """
        active_model = getattr(self._wrapper, "active_model", None)
        if active_model is None:
            raise_validation_error(
                context_name="WorkspaceIO",
                message="Cannot save a model because no model is currently loaded.",
            )

        cohort_context = getattr(getattr(self._wrapper, "cohorts", None), "active_context", None)
        context_name = (
            img_name
            or (cohort_context.key if self.execution_scope == "cohort" and cohort_context else None)
            or self.active_img_name
            or self.active_image_key
        )
        if not context_name:
            raise_validation_error(
                context_name="WorkspaceIO",
                message=(
                    "Cannot infer the model storage context. Pass img_name='global' "
                    "for a global model or select an active image."
                ),
            )

        models_manager = self._wrapper.models_manager
        resolved_model_name = (
            model_name
            or self.active_model_name
            or getattr(models_manager, "_active_model_name", None)
        )
        if not resolved_model_name:
            raise_validation_error(
                context_name="WorkspaceIO",
                message="Cannot save a model without a model name.",
            )

        model_config = models_manager.get_model_config()
        image_config = None
        cohort_config = None
        if cohort_context is not None and context_name == cohort_context.key:
            cohort_config = cohort_context.get_config()
        elif context_name != GLOBAL_CONTEXT_KEY:
            image_config = self._wrapper.context_manager.get_context_config(context_name)
        consolidated_config = {
            "schema_version": 2,
            "experiment": {
                "name": resolved_model_name,
                "coordinate_order": self._wrapper.coordinate_order,
                "context": {
                    "type": "cohort" if cohort_config is not None else "image",
                    "key": context_name,
                },
            },
            "data": {
                "context": cohort_config or image_config,
                "dataset": model_config.get("dataset"),
                "split": model_config.get("split"),
            },
            "model": model_config["model"],
            "training": model_config["training"],
        }

        if history is None:
            history = getattr(models_manager, "_training_history", None)
        model_dir = self._model_store.save_model(
            context_name=context_name,
            model_name=resolved_model_name,
            config=consolidated_config,
            state_dict=active_model.state_dict(),
            history=history,
        )
        self.active_model_name = resolved_model_name
        logger.info(
            "Active model '%s' saved under context '%s'.",
            resolved_model_name,
            context_name,
        )
        return model_dir

    def load_model_artifacts(
        self,
        img_name: str,
        model_name: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load configuration and weights without constructing a runtime model.

        Runtime construction belongs to the loaded-model manager and is introduced in
        the model-context pull request.

        :param img_name: Image key or ``global`` model context.
        :type img_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :return: Pair containing configuration and state dictionary.
        :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
        """
        return (
            self._model_store.load_config(img_name, model_name),
            self._model_store.load_weights(img_name, model_name),
        )

    def export_model_folder(
        self,
        destination: Path | str,
        img_name: Optional[str] = None,
        model_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """Export the complete saved model folder to an explicit destination.

        This operation is opt-in and is never triggered by :meth:`save_model`.

        :param destination: Exact destination directory for the portable model folder.
        :type destination: pathlib.Path | str
        :param img_name: Saved image or global context key.
        :type img_name: Optional[str]
        :param model_name: Saved model name.
        :type model_name: Optional[str]
        :param overwrite: Replace an existing destination when explicitly true.
        :type overwrite: bool
        :return: Exported model folder.
        :rtype: pathlib.Path
        """
        context_name = img_name or self.active_img_name or self.active_image_key
        resolved_model_name = model_name or self.active_model_name
        if not context_name or not resolved_model_name:
            raise_validation_error(
                context_name="WorkspaceIO",
                message="A saved model context and model name are required for export.",
            )
        return self._model_store.export_model_folder(
            context_name=context_name,
            model_name=resolved_model_name,
            destination=Path(destination),
            overwrite=overwrite,
        )
