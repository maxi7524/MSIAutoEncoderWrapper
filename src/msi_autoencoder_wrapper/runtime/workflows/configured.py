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
from ...data import prepare_jerm_static_spy_cache
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


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
    _resolve_chemistry_path(dataset_parameters, project_path)
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


def build_cohort_autoencoder(parameters: dict[str, Any]) -> MSIAutoEncoderWrapper:
    """Build one shared-axis autoencoder over configured cohort training images.

    :param parameters: Cohort member definitions, shared data settings and resolved
        model/split artifacts produced by :func:`resolve_cohort_campaign`.
    :type parameters: dict[str, typing.Any]
    :return: Wrapper with an active ``CohortPixelDataset`` and model.
    :rtype: MSIAutoEncoderWrapper
    """
    resolved = parameters.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError("Training tasks require resolved component artifacts.")
    split_manifest = _read_yaml(Path(resolved["split_manifest"]))
    wrapper, _ = _build_cohort_pipeline(
        parameters,
        split_seed=int(split_manifest["seed"]),
        predefined_assignments=split_manifest["assignments"],
    )
    model, model_type, model_name = ModelLoader.build(
        _read_yaml(Path(resolved["model_config"]))
    )
    wrapper.models_manager.attach_model(
        model,
        model_type=model_type,
        model_name=model_name,
        trained=False,
    )
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
    planning_pipelines: dict[Any, tuple[MSIAutoEncoderWrapper, Any]] = {}
    jerm_precomputes: set[Any] = set()
    split_paths: dict[int, Path] = {}
    split_dataset: Any = None
    resolved_parameters: list[dict[str, Any]] = []
    reference_selections: dict[Any, Any] = {}

    for task in tasks:
        parameters = deepcopy(task["parameters"])
        factory_parameters = parameters["factory_parameters"]
        # Freeze the real population on a reference axis for paired range comparisons
        reference_binning = factory_parameters.get("split_reference_binning")
        if reference_binning is not None:
            reference_key = (_freeze(reference_binning), _freeze(factory_parameters["dataset"]),
                             int(task["reproducibility"]["common_seeds"]["split"]))
            if reference_key not in reference_selections:
                reference_parameters = deepcopy(factory_parameters)
                reference_parameters["binning"] = deepcopy(reference_binning)
                _, reference_dataset = _build_planning_pipeline(
                    reference_parameters, split_seed=reference_key[-1],
                )
                reference_selections[reference_key] = reference_dataset
                logger.info("Resolved the reference-axis population for split seed %s.", reference_key[-1])
            reference_dataset = reference_selections[reference_key]
            manifest = reference_dataset.create_partitions().manifest
            source_indices = sorted(i for ids in manifest.assignments.values() for i in ids)
            logger.debug("Reusing %s source samples across spectral axes.", len(source_indices))
            factory_parameters["dataset"]["parameters"]["subset"] = {
                "method": "source_indices", "indices": source_indices,
            }
            if split_dataset is None:
                split_dataset = reference_dataset
        binning = factory_parameters["binning"]
        variant = factory_parameters["variant"]
        predictive = factory_parameters.get("predictive", {})
        binning_key = _freeze(binning)
        dataset_key = (binning_key, _freeze(factory_parameters["dataset"]), int(
            task["reproducibility"]["common_seeds"]["split"]
        ))
        model_key = (binning_key, _freeze(variant), _freeze(predictive), _freeze(factory_parameters["dataset"]))

        # Shared dataset precompute
        ## Build evidence-derived target masks exactly once for every immutable
        ## subset/binner/split contract, before distributed workers start.
        if dataset_key not in planning_pipelines:
            wrapper, dataset = _build_planning_pipeline(
                factory_parameters,
                split_seed=int(task["reproducibility"]["common_seeds"]["split"]),
            )
            precompute = getattr(dataset, "precompute_simulated_negatives", None)
            if callable(precompute):
                summary = precompute()
                logger.info(
                    "Prepared campaign simulated negatives: strategy=%s spectra=%s entries=%s.",
                    summary["strategy"], summary["spectra"], summary["entries"],
                )
            planning_pipelines[dataset_key] = (wrapper, dataset)
        wrapper, dataset = planning_pipelines[dataset_key]
        if (
            _requires_criterion(
                parameters.get("training", {}), "JERMLoss"
            )
            and dataset_key not in jerm_precomputes
        ):
            prepare_jerm_static_spy_cache(dataset, "molecule")
            jerm_precomputes.add(dataset_key)

        if model_key not in model_paths:
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


def resolve_cohort_campaign(
    tasks: list[dict[str, Any]],
    directory: Path,
) -> list[dict[str, Any]]:
    """Materialize one shared-axis cohort dataset, split and model per grid cell.

    The cohort list is deliberately part of the immutable dataset key.  Thus an
    external-test image cannot enter a later repetition by cache reuse.

    :param tasks: Grid-expanded runtime tasks.
    :type tasks: list[dict[str, typing.Any]]
    :param directory: Campaign directory receiving resolved artifacts.
    :type directory: pathlib.Path
    :return: Task parameters with portable resolved model and split paths.
    :rtype: list[dict[str, typing.Any]]
    """
    artifact_root = directory / "resolved"
    model_root = artifact_root / "models"
    split_root = artifact_root / "splits"
    model_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)
    pipelines: dict[Any, tuple[MSIAutoEncoderWrapper, Any]] = {}
    model_paths: dict[Any, Path] = {}
    split_paths: dict[Any, Path] = {}
    resolved_parameters: list[dict[str, Any]] = []

    for task in tasks:
        parameters = deepcopy(task["parameters"])
        factory_parameters = parameters["factory_parameters"]
        split_seed = int(task["reproducibility"]["common_seeds"]["split"])
        dataset_key = (
            _freeze(_resolve_cohort_images(factory_parameters, Path(factory_parameters["project_path"]))),
            _freeze(factory_parameters.get("binning", {})),
            _freeze(factory_parameters.get("dataset", {})),
            split_seed,
        )
        if dataset_key not in pipelines:
            pipelines[dataset_key] = _build_cohort_pipeline(
                factory_parameters,
                split_seed=split_seed,
            )
        wrapper, dataset = pipelines[dataset_key]
        variant = factory_parameters["variant"]
        predictive = factory_parameters.get("predictive", {})
        model_key = (dataset_key, _freeze(variant), _freeze(predictive))
        if model_key not in model_paths:
            ArchitecturesManager.discover_architectures()
            preset = ArchitecturesManager._PRESET_REGISTRY["autoencoder"][variant["preset"]]
            layout = preset(wrapper.active_context, **deepcopy(variant.get("parameters", {})))
            layout, model_parameters = _attach_predictive_components(layout, predictive, dataset)
            model_path = model_root / f"model-{len(model_paths):04d}.yaml"
            _write_yaml(
                model_path,
                {
                    "model": {
                        "name": variant["name"],
                        "type": "autoencoder",
                        "parameters": model_parameters,
                        "components": {
                            category: _portable_component_descriptor(descriptor)
                            for category, descriptor in layout.items()
                        },
                    }
                },
            )
            model_paths[model_key] = model_path
        if dataset_key not in split_paths:
            manifest = dataset.create_partitions().manifest.get_config()
            split_path = split_root / f"split-{len(split_paths):04d}.yaml"
            _write_yaml(split_path, manifest)
            split_paths[dataset_key] = split_path
        parameters["resolved"] = {
            "model_config": str(model_paths[model_key].resolve()),
            "split_manifest": str(split_paths[dataset_key].resolve()),
        }
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
    _resolve_chemistry_path(dataset_parameters, project_path)
    dataset_parameters["split"]["seed"] = split_seed
    dataset = wrapper.models_manager.load_dataset_config(
        {
            "type": dataset_definition.get("strategy", "PixelDataset"),
            "parameters": dataset_parameters,
        }
    )
    return wrapper, dataset


def _build_cohort_pipeline(
    parameters: dict[str, Any],
    *,
    split_seed: int,
    predefined_assignments: dict[str, Any] | None = None,
) -> tuple[MSIAutoEncoderWrapper, Any]:
    """Initialize local member contexts and one common-target cohort dataset."""
    project_path = Path(parameters["project_path"]).resolve()
    wrapper = MSIAutoEncoderWrapper(
        project_path=str(project_path),
        device=parameters.get("device"),
        dtype=parameters.get("dtype", "float32"),
    )
    images = _resolve_cohort_images(parameters, project_path)
    image_keys: list[str] = []
    for image in images:
        if not isinstance(image, dict) or not image.get("image_key") or not image.get("image_path"):
            raise ValueError("Every cohort image requires image_key and image_path.")
        image_key = str(image["image_key"])
        image_path = Path(image["image_path"])
        if not image_path.is_absolute():
            image_path = (project_path / image_path).resolve()
        if image_key != image_path.stem:
            raise ValueError(
                "cohort image_key must equal the imzML filename stem; "
                f"got '{image_key}' for '{image_path.name}'."
            )
        _get_or_create_reader(
            wrapper,
            image_path=image_path,
            definition=image.get("reader", parameters.get("reader", {})),
        )
        _attach_annotation_reader(
            wrapper,
            image_path=image_path,
            definition=image.get("annotations", parameters.get("annotations")),
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
        image_keys.append(image_key)

    wrapper.cohorts.create(str(parameters.get("cohort_name", "training")))
    cohort_context = wrapper.cohorts.set_images(image_keys, name=str(parameters.get("cohort_name", "training")))
    wrapper.cohorts.activate(cohort_context.name)
    dataset_definition = parameters["dataset"]
    dataset_parameters = deepcopy(dataset_definition["parameters"])
    _resolve_chemistry_path(dataset_parameters, project_path)
    _inject_cohort_class_mapping(dataset_parameters, project_path)
    dataset_parameters["split"] = {
        "strategy": "predefined" if predefined_assignments is not None else dataset_parameters["split"]["strategy"],
        "seed": split_seed,
        "assignments": predefined_assignments,
        "fractions": dataset_parameters["split"].get("fractions"),
        "parameters": dataset_parameters["split"].get("parameters", {}),
    }
    dataset = DatasetManager.load_config(
        {"type": dataset_definition.get("strategy", "CohortPixelDataset"), "parameters": dataset_parameters},
        cohort_context=cohort_context,
    )
    wrapper.active_dataset = dataset
    return wrapper, dataset


def _resolve_cohort_images(parameters: dict[str, Any], project_path: Path) -> list[dict[str, Any]]:
    """Resolve explicit training members or the persisted held-out selection table."""
    images = parameters.get("images")
    if isinstance(images, list) and images:
        return deepcopy(images)
    selection_path = parameters.get("cohort_selection_path")
    if selection_path is None:
        raise ValueError("Cohort training requires images or cohort_selection_path.")
    path = Path(selection_path)
    if not path.is_absolute():
        path = (project_path / path).resolve()
    import pandas as pd

    selected = pd.read_csv(path, usecols=["image_key", "is_heldout"])
    training_keys = selected.loc[~selected["is_heldout"], "image_key"].astype(str).tolist()
    if not training_keys:
        raise ValueError("Cohort selection does not leave any training images.")
    return [
        {
            "image_key": image_key,
            "image_path": f"datasets/{image_key}/{image_key}.imzML",
        }
        for image_key in training_keys
    ]


def _inject_cohort_class_mapping(parameters: dict[str, Any], project_path: Path) -> None:
    """Inject the canonical training-only molecular vocabulary into member datasets."""
    catalogue_path = parameters.pop("annotation_catalogue_path", None)
    if catalogue_path is None:
        return
    path = Path(catalogue_path)
    if not path.is_absolute():
        path = (project_path / path).resolve()
    import pandas as pd

    labels = pd.read_csv(path, usecols=["label", "training_image_count"])
    labels = labels.loc[labels["training_image_count"] > 0, "label"].astype(str).sort_values()
    if labels.empty:
        raise ValueError("Cohort annotation catalogue contains no training labels.")
    target_specs = parameters.setdefault("target_specs", {})
    molecule_spec = target_specs.setdefault("molecule", {"type": "multi_label"})
    molecule_spec["class_mapping"] = {
        label: index for index, label in enumerate(labels.tolist())
    }


def _resolve_chemistry_path(parameters: dict[str, Any], project_path: Path) -> None:
    """Resolve a dataset chemical snapshot relative to its project workspace."""
    chemistry = parameters.get("chemistry")
    if chemistry and chemistry.get("path"):
        path = Path(chemistry["path"])
        chemistry["path"] = str(path if path.is_absolute() else project_path / path)


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
    head_specs: dict[str, dict[str, Any]] = {}
    heads: dict[str, dict[str, Any]] = {}
    for head_id, definition in predictive.get("heads", {}).items():
        target_field = str(definition["target_field"])
        if target_field not in schemas:
            raise ValueError(
                f"Predictive head '{head_id}' references unknown target '{target_field}'."
            )
        parameters = deepcopy(definition.get("parameters", {}))
        requested_classes = definition.get("class_names")
        if requested_classes is not None:
            if schemas[target_field].target_type != "multi_label":
                raise ValueError(
                    f"Predictive head '{head_id}' class_names is supported only for multi_label targets."
                )
            if not isinstance(requested_classes, list) or not all(
                isinstance(name, str) for name in requested_classes
            ):
                raise ValueError(
                    f"Predictive head '{head_id}' class_names must be a list of strings."
                )
            available_names = schemas[target_field].class_names
            available_indices = {name: index for index, name in enumerate(available_names)}
            try:
                class_indices = tuple(available_indices[name] for name in requested_classes)
            except KeyError as error:
                raise ValueError(
                    f"Predictive head '{head_id}' requested unavailable class '{error.args[0]}'."
                ) from error
        else:
            class_indices = None
        if parameters.get("output_dim") == "auto_from_target":
            parameters["output_dim"] = (
                schemas[target_field].class_count
                if class_indices is None
                else len(class_indices)
            )
        heads[str(head_id)] = {
            "strategy": definition["strategy"],
            "params": parameters,
        }
        head_specs[str(head_id)] = {"target_field": target_field}
        if class_indices is not None:
            head_specs[str(head_id)]["class_indices"] = class_indices
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


def _requires_criterion(training: dict[str, Any], target: str) -> bool:
    """Return whether a training definition contains one criterion target."""
    for phase in training.get("phases", []):
        criterions = phase.get("criterions", {})
        for head_losses in criterions.get("heads", {}).values():
            for name, definition in head_losses.items():
                configured = definition.get("target", name) if isinstance(definition, dict) else definition
                if configured == target:
                    return True
    return False
