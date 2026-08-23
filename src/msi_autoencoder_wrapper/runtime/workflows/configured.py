"""Built-in wrapper factory for declarative single-image autoencoder campaigns."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ...configuration import get_component_config
from ...core.wrapper import MSIAutoEncoderWrapper
from ...models.architectures.architectures_manager import ArchitecturesManager
from ...models.datasets.dataset_manager import DatasetManager
from ...models.model_loader import ModelLoader


# Process-local shared resources
## Persistent local workers reuse native readers; each worker owns its own safe handle
_READER_CACHE: dict[tuple[Any, ...], Any] = {}


def _resolve_image_path(parameters: dict[str, Any]) -> Path:
    """Resolve the image path against the declared workspace root."""
    workspace = Path(parameters["project_path"]).resolve()
    image_path = Path(parameters["image_path"])
    return image_path.resolve() if image_path.is_absolute() else (workspace / image_path).resolve()


def _freeze(value: Any) -> Any:
    """Convert nested configuration values into a deterministic cache key."""
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _get_or_create_reader(
    wrapper: MSIAutoEncoderWrapper,
    *,
    image_path: Path,
    definition: dict[str, Any],
) -> Any:
    """Attach a cached reader or initialize it once in the current worker."""
    strategy = definition.get("strategy", "PyImzMLReader")
    reader_parameters = deepcopy(definition.get("parameters", {}))
    reader_config = {"type": strategy, "parameters": reader_parameters}
    cache_key = (str(strategy), str(image_path), _freeze(reader_parameters))
    reader = _READER_CACHE.get(cache_key)
    if reader is None:
        # First task for this dataset in the worker
        ## Native reader initialization may scan and normalize the complete MSI image
        reader = wrapper.context_manager.load_reader(
            reader_config,
            str(image_path),
        )
        _READER_CACHE[cache_key] = reader
        return reader

    # Subsequent task for the same dataset in the worker
    ## Rebind only the lightweight context reference; retain the initialized native handle
    reader.active_context = wrapper.active_context
    return wrapper.context_manager.load_reader(
        reader_config,
        str(image_path),
        reader_instance=reader,
    )


def _attach_annotation_reader(
    wrapper: MSIAutoEncoderWrapper,
    *,
    image_path: Path,
    definition: Any,
) -> None:
    """Attach the declared annotation source to the planning context.

    :param wrapper: Planning wrapper owning the active image context.
    :type wrapper: MSIAutoEncoderWrapper
    :param image_path: Resolved source image path.
    :type image_path: pathlib.Path
    :param definition: ``auto`` discovery or an explicit reader descriptor.
    :type definition: Any
    """
    if not isinstance(definition, dict):
        return
    strategy = definition.get("strategy", "auto")
    parameters = deepcopy(definition.get("parameters", {}))
    if strategy == "auto":
        wrapper.context_manager.set_annotation_reader(
            None,
            img_name_or_path=str(image_path),
            **parameters,
        )
        return
    wrapper.context_manager.set_annotation_reader(
        strategy,
        img_name_or_path=str(image_path),
        **parameters,
    )


def build_single_image_autoencoder(parameters: dict[str, Any]) -> MSIAutoEncoderWrapper:
    """Build one autoencoder pipeline with optional predictive components.

    :param parameters: Workspace, image, preprocessing, architecture, and dataset
        definitions supplied by the runtime configuration.
    :type parameters: dict[str, Any]
    :return: Compiled wrapper with one active image dataset and model.
    :rtype: MSIAutoEncoderWrapper
    :raises ValueError: If the architecture variant is incomplete.
    """
    # Runtime facade
    ## Resolve filesystem inputs before constructing stateful library components
    project_path = Path(parameters["project_path"]).resolve()
    image_path = _resolve_image_path(parameters)
    wrapper = MSIAutoEncoderWrapper(
        project_path=str(project_path),
        device=parameters.get("device"),
        dtype=parameters.get("dtype", "float32"),
    )

    # Resolved component configuration
    ## Training reconstructs exactly the context components persisted by planning.
    resolved = parameters.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError("Training tasks require resolved component artifacts.")
    context_config = _read_yaml(Path(resolved["context_config"]))
    reader = _get_or_create_reader(
        wrapper,
        image_path=image_path,
        definition=parameters["reader"],
    )
    wrapper.context_manager.load_context_config(
        context_config,
        str(image_path),
        reader_instance=reader,
    )
    model_config = _read_yaml(Path(resolved["model_config"]))
    split_manifest = _read_yaml(Path(resolved["split_manifest"]))
    dataset = parameters["dataset"]
    dataset_parameters = deepcopy(dataset["parameters"])
    dataset_parameters["split"] = {
        "strategy": "predefined",
        "seed": int(split_manifest["seed"]),
        "assignments": split_manifest["assignments"],
        "fractions": dataset_parameters["split"].get("fractions"),
    }
    wrapper.active_dataset = DatasetManager.load_config(
        {"type": dataset["strategy"], "parameters": dataset_parameters},
        active_context=wrapper.active_context,
    )
    model, model_type, model_name = ModelLoader.build(model_config)
    wrapper.models_manager.attach_model(model, model_type=model_type, model_name=model_name, trained=False)
    return wrapper


def resolve_single_image_campaign(
    tasks: list[dict[str, Any]],
    directory: Path,
) -> list[dict[str, Any]]:
    """Materialize shared model, binner, and split configurations for a campaign.

    :param tasks: Grid-expanded tasks containing unresolved factory parameters.
    :type tasks: list[dict[str, Any]]
    :param directory: Plan output directory receiving shared artifacts.
    :type directory: pathlib.Path
    :return: Task parameter mappings referencing resolved artifacts.
    :rtype: list[dict[str, Any]]
    """
    artifact_root = directory / "resolved"
    model_root = artifact_root / "models"
    context_root = artifact_root / "contexts"
    split_root = artifact_root / "splits"
    binner_root = artifact_root / "binners"
    inverse_binner_root = artifact_root / "inverse_binners"
    for path in (model_root, context_root, split_root, binner_root, inverse_binner_root):
        path.mkdir(parents=True, exist_ok=True)

    # Shared resolution caches
    ## One wrapper resolves each unique architecture-binning pair, not each repetition
    model_paths: dict[Any, Path] = {}
    binner_paths: dict[Any, Path] = {}
    inverse_binner_paths: dict[Any, Path] = {}
    context_paths: dict[Any, Path] = {}
    datasets_by_binning: dict[Any, Any] = {}
    split_paths: dict[int, Path] = {}
    split_dataset: Any = None
    resolved_parameters: list[dict[str, Any]] = []

    for task in tasks:
        parameters = deepcopy(task["parameters"])
        factory_parameters = parameters["factory_parameters"]
        binning = factory_parameters["binning"]
        variant = factory_parameters["variant"]
        predictive = factory_parameters.get("predictive", {})
        binning_key = _freeze(binning)
        model_key = (binning_key, _freeze(variant), _freeze(predictive))

        if model_key not in model_paths:
            wrapper, dataset = _build_planning_pipeline(
                factory_parameters,
                split_seed=int(task["reproducibility"]["common_seeds"]["split"]),
            )
            ArchitecturesManager.discover_architectures()
            preset = ArchitecturesManager._PRESET_REGISTRY["autoencoder"][variant["preset"]]
            component_layout = preset(
                wrapper.active_context,
                **deepcopy(variant.get("parameters", {})),
            )
            component_layout, model_parameters = _attach_predictive_components(
                component_layout,
                predictive,
                dataset,
            )
            model_config = {
                "model": {
                    "name": variant["name"],
                    "type": "autoencoder",
                    "parameters": model_parameters,
                    "components": {
                        category: _portable_component_descriptor(descriptor)
                        for category, descriptor in component_layout.items()
                    },
                }
            }
            model_path = model_root / f"model-{len(model_paths):04d}.yaml"
            _write_yaml(model_path, model_config)
            model_paths[model_key] = model_path
            datasets_by_binning.setdefault(binning_key, dataset)
            if split_dataset is None:
                split_dataset = dataset

            if binning_key not in binner_paths:
                binner_path = binner_root / f"binner-{len(binner_paths):04d}.yaml"
                _write_yaml(binner_path, get_component_config(wrapper.active_context.binner))
                binner_paths[binning_key] = binner_path

                # Inverse reconstruction configuration
                ## Resolve and persist the expensive shared reconstruction axis once
                inverse_definition = factory_parameters.get("inverse_binning")
                if isinstance(inverse_definition, dict):
                    inverse_binner = wrapper.context_manager.load_inverse_binner(
                        {
                            "type": inverse_definition["strategy"],
                            "parameters": deepcopy(inverse_definition.get("parameters", {})),
                        },
                        str(_resolve_image_path(factory_parameters)),
                    )
                    wrapper.workspace.set_active_image(
                        str(_resolve_image_path(factory_parameters))
                    )
                    inverse_descriptor = get_component_config(inverse_binner)
                    inverse_descriptor["parameters"]["reconstruction_mass_axis"] = (
                        inverse_binner.reconstruction_mass_axis.detach().cpu().tolist()
                    )
                    inverse_path = (
                        inverse_binner_root
                        / f"inverse-binner-{len(inverse_binner_paths):04d}.yaml"
                    )
                    _write_yaml(inverse_path, inverse_descriptor)
                    inverse_binner_paths[binning_key] = inverse_path
                context_path = context_root / f"context-{len(context_paths):04d}.yaml"
                _write_yaml(context_path, wrapper.context_manager.get_context_config())
                context_paths[binning_key] = context_path

        # Split manifests depend on the dataset and seed, not on bin width or model
        split_key = int(task["reproducibility"]["common_seeds"]["split"])
        if split_key not in split_paths:
            dataset = split_dataset
            split = deepcopy(dataset._split_config)
            split["seed"] = split_key
            dataset._split_config = split
            dataset._partitions = None
            manifest = dataset.create_partitions().manifest.get_config()
            split_path = split_root / f"split-{len(split_paths):04d}.yaml"
            _write_yaml(split_path, manifest)
            split_paths[split_key] = split_path

        parameters["resolved"] = {
            "model_config": str(model_paths[model_key].resolve()),
            "context_config": str(context_paths[binning_key].resolve()),
            "binner_config": str(binner_paths[binning_key].resolve()),
            "split_manifest": str(split_paths[split_key].resolve()),
        }
        if binning_key in inverse_binner_paths:
            parameters["resolved"]["inverse_binner_config"] = str(
                inverse_binner_paths[binning_key].resolve()
            )
        resolved_parameters.append(parameters)
    return resolved_parameters


def _build_planning_pipeline(
    parameters: dict[str, Any],
    *,
    split_seed: int,
) -> tuple[MSIAutoEncoderWrapper, Any]:
    """Initialize data components required to resolve configs, but no model weights."""
    project_path = Path(parameters["project_path"]).resolve()
    image_path = _resolve_image_path(parameters)
    wrapper = MSIAutoEncoderWrapper(
        project_path=str(project_path),
        device=parameters.get("device"),
        dtype=parameters.get("dtype", "float32"),
    )

    # Shared reader and selected binner
    ## Planning needs their resolved dimensions, but never constructs a Torch model
    _get_or_create_reader(
        wrapper,
        image_path=image_path,
        definition=parameters.get("reader", {}),
    )
    _attach_annotation_reader(
        wrapper,
        image_path=image_path,
        definition=parameters.get("annotations"),
    )
    binning = parameters.get("binning", {})
    wrapper.context_manager.load_binner(
        {
            "type": binning.get("strategy", "LinearBinning"),
            "parameters": deepcopy(binning.get("parameters", {})),
        },
        str(image_path),
    )
    wrapper.workspace.set_active_image(str(image_path))

    # Dataset metadata and split source
    ## Dataset construction has no model and initializes no neural-network weights
    dataset_definition = parameters.get("dataset", {})
    dataset_parameters = deepcopy(dataset_definition.get("parameters", {}))
    dataset_parameters["split"]["seed"] = split_seed
    dataset = wrapper.models_manager.load_dataset_config(
        {
            "type": dataset_definition.get("strategy", "PixelDataset"),
            "parameters": dataset_parameters,
        }
    )
    return wrapper, dataset


def _attach_predictive_components(
    component_layout: dict[str, Any],
    predictive: Any,
    dataset: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve projector and head dimensions against dataset target schemas.

    :param component_layout: Architecture preset component descriptors.
    :type component_layout: dict[str, Any]
    :param predictive: Optional predictive component configuration.
    :type predictive: Any
    :param dataset: Planning dataset exposing target schemas.
    :type dataset: Any
    :return: Extended components and model-level ``head_specs`` parameters.
    :rtype: tuple[dict[str, Any], dict[str, Any]]
    """
    layout = deepcopy(component_layout)
    if not isinstance(predictive, dict) or not predictive:
        return layout, {}
    projector = predictive.get("projector")
    if isinstance(projector, dict):
        layout["projector"] = {
            "strategy": projector["strategy"],
            "params": deepcopy(projector.get("parameters", {})),
        }
    schemas = dataset.get_target_schemas()
    head_specs: dict[str, dict[str, str]] = {}
    heads: dict[str, dict[str, Any]] = {}
    for head_id, definition in predictive.get("heads", {}).items():
        target_field = str(definition["target_field"])
        if target_field not in schemas:
            raise ValueError(
                f"Predictive head '{head_id}' references unknown target '{target_field}'."
            )
        parameters = deepcopy(definition.get("parameters", {}))
        if parameters.get("output_dim") == "auto_from_target":
            parameters["output_dim"] = schemas[target_field].class_count
        heads[str(head_id)] = {
            "strategy": definition["strategy"],
            "params": parameters,
        }
        head_specs[str(head_id)] = {"target_field": target_field}
    if heads:
        layout["heads"] = heads
    return layout, {"head_specs": head_specs} if head_specs else {}


def _portable_component_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Convert one architecture component tree to its portable representation.

    :param descriptor: Runtime component descriptor or nested named components.
    :type descriptor: dict[str, Any]
    :return: Portable descriptor accepted by :class:`ModelLoader`.
    :rtype: dict[str, Any]
    """
    if "strategy" in descriptor:
        return {
            "type": descriptor["strategy"],
            "version": 1,
            "parameters": deepcopy(descriptor.get("params", {})),
        }
    return {
        child_name: _portable_component_descriptor(child_descriptor)
        for child_name, child_descriptor in descriptor.items()
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one resolved runtime artifact."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Resolved artifact must contain a mapping: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write one deterministic resolved runtime artifact."""
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)
