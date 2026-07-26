"""Filesystem persistence for portable model configurations and weights."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..utils.configuration import make_json_compatible
from ..utils.exceptions import raise_workspace_error
from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ModelStore:
    """Persist model artifacts independently from the user-facing workspace mixin."""

    CONFIG_FILENAME = "config.json"
    WEIGHTS_FILENAME = "weights.pt"
    HISTORY_FILENAME = "history.json"
    FULL_MODEL_FILENAME = "model_deployment_full.pt"

    def __init__(self, workspace_root: Path, layout: Dict[str, str]) -> None:
        """Initialize model persistence paths.

        :param workspace_root: Root directory of the MSI project workspace.
        :type workspace_root: pathlib.Path
        :param layout: Normalized workspace directory layout.
        :type layout: Dict[str, str]
        """
        self.workspace_root = workspace_root
        self.layout = layout

    def get_model_dir(self, context_name: str, model_name: str) -> Path:
        """Return the directory containing all artifacts for one model.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :return: Model artifact directory.
        :rtype: pathlib.Path
        """
        return self.workspace_root / self.layout["models_root"] / context_name / model_name

    def get_config_dir(self, context_name: str, model_name: str) -> Path:
        """Return the configuration subdirectory for one model.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :return: Configuration artifact directory.
        :rtype: pathlib.Path
        """
        return self.get_model_dir(context_name, model_name) / self.layout["model_config_subdir"]

    def save_config(
        self,
        context_name: str,
        model_name: str,
        config: Dict[str, Any],
    ) -> Path:
        """Save the portable JSON model configuration atomically.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param config: Consolidated model and pipeline configuration.
        :type config: Dict[str, Any]
        :return: Written configuration path.
        :rtype: pathlib.Path
        :raises WorkspaceConfigError: If the file cannot be written.
        """
        path = self.get_config_dir(context_name, model_name) / self.CONFIG_FILENAME
        payload = make_json_compatible(config)
        self._write_json(path, payload)
        logger.info("Model configuration saved at: %s", path)
        return path

    def load_config(self, context_name: str, model_name: str) -> Dict[str, Any]:
        """Load the portable JSON model configuration.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :return: Consolidated model and pipeline configuration.
        :rtype: Dict[str, Any]
        :raises WorkspaceConfigError: If the configuration is missing or invalid.
        """
        path = self.get_config_dir(context_name, model_name) / self.CONFIG_FILENAME
        if not path.is_file():
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Configuration file does not exist: {path}",
            )
        try:
            with path.open(mode="r", encoding="utf-8") as config_file:
                payload = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Failed to load configuration from '{path}': {error}",
            )
        if not isinstance(payload, dict):
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Configuration root in '{path}' must be a JSON object.",
            )
        return payload

    def save_weights(
        self,
        context_name: str,
        model_name: str,
        state_dict: Dict[str, Any],
    ) -> Path:
        """Save a PyTorch state dictionary atomically.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param state_dict: Model weight state dictionary.
        :type state_dict: Dict[str, Any]
        :return: Written weights path.
        :rtype: pathlib.Path
        :raises WorkspaceConfigError: If the checkpoint cannot be written.
        """
        path = self.get_config_dir(context_name, model_name) / self.WEIGHTS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            torch.save(state_dict, temporary_path)
            temporary_path.replace(path)
        except Exception as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Failed to save model weights to '{path}': {error}",
            )
        logger.info("Model weights saved at: %s", path)
        return path

    def load_weights(self, context_name: str, model_name: str) -> Dict[str, Any]:
        """Load a state dictionary on CPU using PyTorch's weights-only mode.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :return: Model weight state dictionary.
        :rtype: Dict[str, Any]
        :raises WorkspaceConfigError: If the checkpoint is missing or invalid.
        """
        path = self.get_config_dir(context_name, model_name) / self.WEIGHTS_FILENAME
        if not path.is_file():
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Model weights do not exist: {path}",
            )
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Failed to load model weights from '{path}': {error}",
            )
        if not isinstance(state_dict, dict):
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Checkpoint '{path}' does not contain a state dictionary.",
            )
        return state_dict

    def save_history(
        self,
        context_name: str,
        model_name: str,
        history: Any,
    ) -> Path:
        """Save JSON-compatible training history atomically.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param history: Training metrics dictionary or sequence.
        :type history: Any
        :return: Written history path.
        :rtype: pathlib.Path
        """
        path = self.get_config_dir(context_name, model_name) / self.HISTORY_FILENAME
        self._write_json(path, make_json_compatible(history, "history"))
        logger.info("Training history saved at: %s", path)
        return path

    def save_model(
        self,
        context_name: str,
        model_name: str,
        config: Dict[str, Any],
        state_dict: Dict[str, Any],
        history: Any = None,
    ) -> Path:
        """Save the default portable model representation: JSON plus weights.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param config: Consolidated model configuration.
        :type config: Dict[str, Any]
        :param state_dict: Model weight state dictionary.
        :type state_dict: Dict[str, Any]
        :param history: Optional training history.
        :type history: Any
        :return: Directory containing the saved model artifacts.
        :rtype: pathlib.Path
        """
        self.save_config(context_name, model_name, config)
        self.save_weights(context_name, model_name, state_dict)
        if history is not None:
            self.save_history(context_name, model_name, history)
        return self.get_model_dir(context_name, model_name)

    def save_full_model(
        self,
        context_name: str,
        model_name: str,
        model: torch.nn.Module,
    ) -> Path:
        """Save a full pickled PyTorch model for legacy compatibility.

        This representation is intentionally not part of :meth:`save_model` because
        JSON plus a state dictionary is safer and more portable.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param model: Compiled PyTorch model.
        :type model: torch.nn.Module
        :return: Written full-model path.
        :rtype: pathlib.Path
        """
        path = self.get_config_dir(context_name, model_name) / self.FULL_MODEL_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(model, path)
        except Exception as error:
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Failed to save full PyTorch model to '{path}': {error}",
            )
        return path

    def export_model_folder(
        self,
        context_name: str,
        model_name: str,
        destination: Path,
        overwrite: bool = False,
    ) -> Path:
        """Copy the complete model folder to an explicit portable destination.

        :param context_name: Image key or global model context key.
        :type context_name: str
        :param model_name: User-facing model name.
        :type model_name: str
        :param destination: Exact destination directory for the exported bundle.
        :type destination: pathlib.Path
        :param overwrite: Replace an existing destination directory when true.
        :type overwrite: bool
        :return: Exported model directory.
        :rtype: pathlib.Path
        :raises WorkspaceConfigError: If the source is incomplete or the destination is unsafe.
        """
        source = self.get_model_dir(context_name, model_name)
        required_files = {
            source / self.layout["model_config_subdir"] / self.CONFIG_FILENAME,
            source / self.layout["model_config_subdir"] / self.WEIGHTS_FILENAME,
        }
        missing_files = sorted(str(path) for path in required_files if not path.is_file())
        if missing_files:
            raise_workspace_error(
                context_name="ModelExport",
                message=f"Model folder is incomplete. Missing files: {missing_files}",
            )

        destination = destination.resolve()
        if source.resolve() == destination:
            raise_workspace_error(
                context_name="ModelExport",
                message="Export destination must differ from the workspace model directory.",
            )
        if destination.exists() and not overwrite:
            raise_workspace_error(
                context_name="ModelExport",
                message=(
                    f"Export destination already exists: {destination}. "
                    "Pass overwrite=True to replace it explicitly."
                ),
            )

        try:
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
        except OSError as error:
            raise_workspace_error(
                context_name="ModelExport",
                message=f"Failed to export model folder to '{destination}': {error}",
            )
        logger.info("Portable model folder exported to: %s", destination)
        return destination

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        """Write a JSON payload atomically.

        :param path: Target JSON path.
        :type path: pathlib.Path
        :param payload: JSON-compatible payload.
        :type payload: Any
        :raises WorkspaceConfigError: If the payload cannot be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, indent=2, ensure_ascii=False)
                temporary_file.write("\n")
            temporary_path.replace(path)
        except (OSError, TypeError, ValueError) as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise_workspace_error(
                context_name="ModelStore",
                message=f"Failed to write JSON file '{path}': {error}",
            )
