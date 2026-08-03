"""
Module managing storage configurations, multi-image registries, and data reader tracking structures.
"""

import pprint
from pathlib import Path
from typing import Dict, Any, Optional
from ...utils.decorators import manage_image_context
from ....utils.printing import present_available_components
from ....utils.logger import get_custom_logger
from ....utils.exceptions import raise_validation_error
from ....utils.validators import resolve_component
from ....configuration import get_component_config
from ....readers.base_reader import MSIBaseReader
from ....readers.readers_manager import ReaderManager
from ....annotations.base_annotation_reader import MSIBaseAnnotationReader
from ....annotations.annotations_manager import AnnotationReaderManager
from ....binners.base_binner import MSIBaseBinner
from ....binners.base_inverse import MSIBaseInverseBinner
from ....binners.binners_manager import BinnerManager
from ....normalization import NormalizationPipeline

logger = get_custom_logger(__name__)


# --------------------------------------------------
# Section: ReadersManagerProxy Infrastructure
# --------------------------------------------------


class ContextManagerProxy:
    """
    Proxy class managing configuration matrices, driver parameters, and temporary session objects across multiple images.
    """

    def __init__(self, wrapper_ref: Any) -> None:
        """
        Initializes the configuration ledger database holding data loader specifications.
        """
        # Store a loose reference back to the coordinating facaded wrapper master object instance
        self._wrapper = wrapper_ref

        # Initialize internal ledger tracking configurations per unique image key
        ## Central data storage mapping image targets to strategy descriptors
        self.config_ledger: Dict[str, Dict[str, Any]] = {}

        # --------------------------------------------------
        # Subsection: Automatic Registration Enforcement
        # --------------------------------------------------
        logger.info(
            "Enforcing automatic module discovery for reader and binner registries."
        )
        ReaderManager.discover_strategies()
        AnnotationReaderManager.load_builtin_readers()
        BinnerManager.discover_strategies()

    # --------------------------------------------------
    # Section: Public Strategy Setters
    # --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Setters - reader
    # --------------------------------------------------

    def set_reader(
        self,
        reader_name_or_instance: Any,
        img_name_or_path: Optional[str] = None,
        annotation_catalog_path: Optional[str] = None,
        auto_load_annotations: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Registers and configures an input data reader strategy for an image context.

        Accepts either a unique registered string token identifier or a pre-initialized
        concrete instance object. Context resolution and filesystem path injections are
        delegated directly down to the internal component registration system.

        :param reader_name_or_instance: Registered strategy identifier string or an initialized reader object.
        :type reader_name_or_instance: Any
        :param img_name_or_path: Explicit target path or standalone image name key token. Defaults to None.
        :type img_name_or_path: Optional[str]
        :param annotation_catalog_path: Optional explicit SQLite catalog path.
            When omitted, first use a registered workspace dataset and then
            detect paired METASPACE CSV exports beside the image.
        :type annotation_catalog_path: Optional[str]
        :param auto_load_annotations: Automatically attach the matching SQLite
            or local METASPACE CSV annotation reader after configuring data.
        :type auto_load_annotations: bool
        :param kwargs: Arbitrary configuration parameters validated and passed to the constructor factory.
        :return: Fully resolved and validated data reader component instance.
        :rtype: Any
        """
        # Strategy routing block
        ## Forward execution properties directly to the unified driver registration manager
        reader = self._set_component(
            component_type="reader",
            target=reader_name_or_instance,
            img_name_or_path=img_name_or_path,
            **kwargs,
        )
        if auto_load_annotations:
            self._load_registered_annotations(
                img_name_or_path=img_name_or_path,
                catalog_path=annotation_catalog_path,
                image_path=getattr(reader, "file_path", None),
            )
        return reader

    def _load_registered_annotations(
        self,
        *,
        img_name_or_path: Optional[str],
        catalog_path: Optional[str],
        image_path: Optional[Path],
    ) -> Optional[MSIBaseAnnotationReader]:
        """Attach the best available annotation source for the selected image.

        :param img_name_or_path: Image identifier or explicit imzML path.
        :type img_name_or_path: str | None
        :param catalog_path: Explicit SQLite path. When omitted, use the single
            catalog in the workspace datasets directory.
        :type catalog_path: str | None
        :param image_path: Resolved path exposed by the configured data reader.
        :type image_path: pathlib.Path | None
        :return: Attached annotation reader, or ``None`` when neither a catalog
            registration nor a complete local METASPACE CSV pair is available.
        :rtype: MSIBaseAnnotationReader | None
        """
        from ....dataset_management.catalog import DatasetCatalog

        workspace = self._wrapper.workspace
        resolved_catalog_path = (
            Path(catalog_path).expanduser().resolve()
            if catalog_path is not None
            else workspace.get_datasets_dir() / "catalog.sqlite"
        )
        if image_path is None:
            logger.warning(
                "No annotations were attached to '%s' because the configured "
                "data reader does not expose an image path.",
                img_name_or_path,
            )
            return None
        resolved_image_path = Path(image_path).expanduser().resolve()
        if catalog_path is not None and not resolved_catalog_path.is_file():
            raise_validation_error(
                "AnnotationReader",
                f"Explicit annotation catalog does not exist: '{resolved_catalog_path}'.",
            )

        if resolved_catalog_path.is_file():
            identity = DatasetCatalog(resolved_catalog_path).resolve_dataset_path(
                resolved_image_path
            )
            if identity is not None:
                reader_options: Dict[str, Any] = {
                    "catalog_path": str(resolved_catalog_path)
                }
                if identity["kind"] == "merged":
                    reader_options["merged_dataset_id"] = identity[
                        "merged_dataset_id"
                    ]
                else:
                    reader_options["source"] = identity["source"]
                    reader_options["dataset_id"] = identity["dataset_id"]
                return self._set_component(
                    component_type="annotation_reader",
                    target="SQLiteAnnotationReader",
                    img_name_or_path=str(resolved_image_path),
                    **reader_options,
                )
            if catalog_path is not None:
                raise_validation_error(
                    "AnnotationReader",
                    (
                        f"Image '{resolved_image_path}' is not registered in the "
                        f"explicit annotation catalog '{resolved_catalog_path}'."
                    ),
                )

        csv_paths = self._find_local_metaspace_annotations(resolved_image_path)
        if csv_paths is not None:
            annotations_path, pixel_intensities_path = csv_paths
            return self._set_component(
                component_type="annotation_reader",
                target="MetaspaceCSVAnnotationReader",
                img_name_or_path=str(resolved_image_path),
                image_path=str(resolved_image_path),
                annotations_path=str(annotations_path),
                pixel_intensities_path=str(pixel_intensities_path),
            )

        logger.warning(
            "No annotations were attached to image '%s'. Register the image in "
            "the workspace catalog, place 'metaspace_annotations.csv' and a "
            "matching '*_pixel_intensities.csv' beside the imzML file, or call "
            "set_annotation_reader(...) explicitly.",
            resolved_image_path,
        )
        return None

    @staticmethod
    def _find_local_metaspace_annotations(
        image_path: Path,
    ) -> Optional[tuple[Path, Path]]:
        """Find one unambiguous pair of local METASPACE CSV exports."""
        directory = image_path.parent
        annotations_path = directory / "metaspace_annotations.csv"
        if not annotations_path.is_file():
            return None

        preferred_path = directory / f"{image_path.stem}_pixel_intensities.csv"
        if preferred_path.is_file():
            return annotations_path, preferred_path

        candidates = sorted(directory.glob("*_pixel_intensities.csv"))
        if len(candidates) == 1:
            return annotations_path, candidates[0]
        if len(candidates) > 1:
            raise_validation_error(
                "AnnotationReader",
                (
                    f"Multiple pixel-intensity CSV files match image '{image_path}': "
                    f"{[path.name for path in candidates]}. Rename the intended file "
                    f"to '{preferred_path.name}' or configure the reader explicitly."
                ),
            )
        return None

    def set_annotation_reader(
        self,
        reader_name_or_instance: Any = None,
        img_name_or_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Register an annotation-reader strategy for an image context.

        When no strategy is supplied, use the configured image reader to detect
        a catalog registration or paired local METASPACE CSV exports.

        :param reader_name_or_instance: Registered annotation reader, class, or
            instance. Omit to auto-detect the source.
        :type reader_name_or_instance: Any
        :param img_name_or_path: Image context receiving the reader.
        :type img_name_or_path: Optional[str]
        :param kwargs: Constructor arguments for the annotation reader.
        :return: Resolved annotation reader.
        :rtype: Any
        """
        if reader_name_or_instance is None:
            active_image_key = self._wrapper.workspace.active_img_name
            requested_image_key = (
                Path(img_name_or_path).stem if img_name_or_path else active_image_key
            )
            image_key = (
                requested_image_key
                if requested_image_key in self.config_ledger
                else active_image_key
            )
            configured_reader = self.config_ledger.get(image_key, {}).get("reader")
            image_path = getattr(configured_reader, "file_path", None)
            return self._load_registered_annotations(
                img_name_or_path=image_key,
                catalog_path=kwargs.pop("catalog_path", None),
                image_path=image_path,
            )
        return self._set_component(
            component_type="annotation_reader",
            target=reader_name_or_instance,
            img_name_or_path=img_name_or_path,
            **kwargs,
        )

    # --------------------------------------------------
    # Subsection: Setters - binners
    # --------------------------------------------------

    def set_binner(
        self,
        binner_name_or_instance: Any,
        img_name_or_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Registers a forward spectral binning compression configuration into the project ledger.

        :param binner_name_or_instance: Registered strategy identifier string or an initialized binner object.
        :type binner_name_or_instance: Any
        :param img_name_or_path: Explicit target path or standalone image name key token. Defaults to None.
        :type img_name_or_path: Optional[str]
        :param kwargs: Arbitrary configuration parameters validated and passed to the constructor factory.
        :return: Fully resolved and validated forward spectrum binner component instance.
        :rtype: Any
        """
        # Strategy routing block
        ## Forward execution properties directly to the unified driver registration manager
        return self._set_component(
            component_type="binner",
            target=binner_name_or_instance,
            img_name_or_path=img_name_or_path,
            **kwargs,
        )

    def set_inverse_binner(
        self,
        inverse_binner_name_or_instance: Any,
        img_name_or_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Registers a reverse reconstruction spatial binner configuration into the project ledger.

        :param inverse_binner_name_or_instance: Registered strategy identifier or initialized inverse binner object.
        :type inverse_binner_name_or_instance: Any
        :param img_name_or_path: Explicit target path or standalone image name key token. Defaults to None.
        :type img_name_or_path: Optional[str]
        :param kwargs: Arbitrary configuration parameters validated and passed to the constructor factory.
        :return: Fully resolved and validated inverse spectrum binner component instance.
        :rtype: Any
        """
        # Strategy routing block
        ## Forward execution properties directly to the unified driver registration manager
        return self._set_component(
            component_type="inverse_binner",
            target=inverse_binner_name_or_instance,
            img_name_or_path=img_name_or_path,
            **kwargs,
        )

    def set_normalization(
        self,
        configuration: Dict[str, Any],
        img_name_or_path: Optional[str] = None,
    ) -> NormalizationPipeline:
        """Replace the complete ordered normalization pipeline for one image.

        :param configuration: Pipeline configuration containing ``stage``, ordered
            ``steps``, and optional ``reconstruction`` policy.
        :type configuration: Dict[str, Any]
        :param img_name_or_path: Image context receiving the pipeline.
        :type img_name_or_path: str | None
        :return: Active normalization pipeline.
        :rtype: NormalizationPipeline
        """
        image_key = self._resolve_image_key(img_name_or_path)
        pipeline = NormalizationPipeline.from_config(configuration)
        self._ensure_image_bucket(image_key)
        self.config_ledger[image_key]["normalization"] = pipeline
        self._wrapper.active_context.clear_active_context()
        logger.info("Set normalization pipeline for image '%s'.", image_key)
        return pipeline

    def update_normalization(
        self,
        configuration: Dict[str, Any],
        img_name_or_path: Optional[str] = None,
    ) -> NormalizationPipeline:
        """Merge settings into the current serialized pipeline and replace it."""
        image_key = self._resolve_image_key(img_name_or_path)
        self._ensure_image_bucket(image_key)
        current = self.config_ledger[image_key].get("normalization")
        merged = current.get_config() if isinstance(current, NormalizationPipeline) else {}
        for key, value in configuration.items():
            if key == "steps" and isinstance(value, dict):
                merged["steps"] = {**merged.get("steps", {}), **value}
            elif key == "reconstruction" and isinstance(value, dict):
                merged["reconstruction"] = {
                    **merged.get("reconstruction", {}),
                    **value,
                }
            else:
                merged[key] = value
        return self.set_normalization(merged, image_key)

    def remove_normalization(
        self,
        name: str,
        img_name_or_path: Optional[str] = None,
    ) -> NormalizationPipeline:
        """Remove one named step while preserving the remaining order."""
        image_key = self._resolve_image_key(img_name_or_path)
        self._ensure_image_bucket(image_key)
        current = self.config_ledger[image_key].get("normalization")
        if not isinstance(current, NormalizationPipeline) or name not in current.steps:
            raise_validation_error("Normalization", f"Unknown normalization step '{name}'.")
        config = current.get_config()
        del config["steps"][name]
        return self.set_normalization(config, image_key)

    def clear_normalization(
        self, img_name_or_path: Optional[str] = None
    ) -> None:
        """Disable normalization for one image context."""
        image_key = self._resolve_image_key(img_name_or_path)
        self._ensure_image_bucket(image_key)
        self.config_ledger[image_key]["normalization"] = None
        self._wrapper.active_context.clear_active_context()

    def _resolve_image_key(self, img_name_or_path: Optional[str]) -> str:
        """Resolve an image key without changing workspace state."""
        candidate = img_name_or_path or self._wrapper.workspace.active_img_name
        if candidate is None:
            raise_validation_error("ContextManager", "An active image is required.")
        path = Path(candidate)
        return path.stem if path.suffix else str(candidate)

    # --------------------------------------------------
    # Section: Public Strategy Getters
    # --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Getters - readers
    # --------------------------------------------------

    def get_available_readers(
        self, print_return: bool = True, return_value: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts documentation strings across all registered data streaming readers.

        :param print_return: If True, prints a beautifully formatted summary of available components and parameters. Defaults to True.
        :type print_return: bool
        :param return_value: Flag determining whether return dict with data logs.
        :type return_value: bool
        :return: Map correlating loader class names to their foundational documentation.
        :rtype: Dict[str, Dict[str, Any]]
        """
        # Scan registration registries
        ## Retrieve explicit dictionary reference mappings from ReaderManager
        return present_available_components(
            registry=ReaderManager.REGISTRY,
            title="REGISTERED MSI READERS & PARAMETERS",
            key_label="Reader Key",
            print_return=print_return,
            return_value=return_value,
        )

    # --------------------------------------------------
    # Subsection: Getters - binners
    # --------------------------------------------------

    def get_available_binners(
        self, print_return: bool = True, return_value: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts documentation strings across all registered forward compression binners.

        :param print_return: If True, prints a beautifully formatted summary of available components and parameters. Defaults to True.
        :type print_return: bool
        :param return_value: Flag determining whether return dict with data logs.
        :type return_value: bool
        :return: Map correlating compression class names to their foundational documentation.
        :rtype: Dict[str, Dict[str, Any]]
        """
        # Scan registration registries
        ## Retrieve explicit dictionary reference mappings from BinnerManager
        return present_available_components(
            registry=BinnerManager.BINNER_REGISTRY,
            title="REGISTERED MSI FORWARD BINNERS & PARAMETERS",
            key_label="Binner Key",
            print_return=print_return,
            return_value=return_value,
        )

    def get_available_inverse_binners(
        self, print_return: bool = True, return_value: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts documentation strings across all registered reverse spatial binners.

        :param print_return: If True, prints a beautifully formatted summary of available components and parameters. Defaults to True.
        :type print_return: bool
        :param return_value: Flag determining whether return dict with data logs.
        :type return_value: bool
        :return: Map correlating decompression class names to their foundational documentation.
        :rtype: Dict[str, Dict[str, Any]]
        """
        # Scan registration registries
        ## Retrieve explicit dictionary reference mappings from BinnerManager INVERSE_REGISTRY
        return present_available_components(
            registry=BinnerManager.INVERSE_REGISTRY,
            title="REGISTERED MSI INVERSE BINNERS & PARAMETERS",
            key_label="Inverse Binner Key",
            print_return=print_return,
            return_value=return_value,
        )

    # --------------------------------------------------
    # Section: Helpers
    # --------------------------------------------------

    def _ensure_image_bucket(self, img_name: str) -> None:
        """
        Ensures a tracking entry bucket structure exists for the specified image context.

        :param img_name: Clean identifier handle representing the target image.
        :type img_name: str
        """
        # Bucket validation sequence
        ## Initialize structured dictionary templates if image key is missing
        if img_name not in self.config_ledger:
            self.config_ledger[img_name] = {
                "reader": {"instance_name": "", "instance_params": {}},
                "annotation_reader": None,
                "binner": {"instance_name": "", "instance_params": {}},
                "inverse_binner": {"instance_name": "", "instance_params": {}},
                "normalization": None,
                "model_functionality": None,
                "tmp": {},
            }

    # --------------------------------------------------
    # Subsection: Helpers - setters
    # --------------------------------------------------

    @manage_image_context
    def _set_component(
        self,
        component_type: str,
        target: Any,
        img_name_or_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Resolves, provisions layout directories, and registers an operational component for an image context.

        The image lifecycle context is implicitly synchronized via the @manage_image_context decorator.
        Once verified, this method maps the component type to its respective central registry database,
        triggers physical workspace verification, and instantiates or tracks the execution driver.

        :param component_type: Structural category of the driver (e.g., 'reader', 'binner', 'inverse_binner').
        :type component_type: str
        :param target: Class string identifier lookup key or concrete pre-initialized instance object.
        :type target: Any
        :param img_name_or_path: Explicit target path or standalone image name key token.
        :type img_name_or_path: Optional[str]
        :param kwargs: Configuration variables passed onto strategy constructor factories during resolution.
        :return: Fully resolved and functional processing component instance.
        :rtype: Any
        :raises ValueError: If the requested component classification is unsupported by the engine.
        """
        # Component manager infrastructure resolution
        ## Define localized registry routing boundaries for known pipeline blocks
        registries = {
            "reader": ReaderManager.REGISTRY,
            "annotation_reader": AnnotationReaderManager.REGISTRY,
            "binner": BinnerManager.BINNER_REGISTRY,
            "inverse_binner": BinnerManager.INVERSE_REGISTRY,
        }
        expected_types = {
            "reader": MSIBaseReader,
            "annotation_reader": MSIBaseAnnotationReader,
            "binner": MSIBaseBinner,
            "inverse_binner": MSIBaseInverseBinner,
        }

        if component_type not in registries:
            raise_validation_error(
                context_name="ContextManager",
                message=(
                    f"Unsupported component type '{component_type}'. "
                    f"Valid types: {sorted(registries)}."
                ),
            )

        workspace = self._wrapper.workspace
        active_img = workspace.active_img_name

        # Dependency injection layer
        ## Automatically inject the active filesystem path if compiling a data loader reader
        if component_type == "reader" and "file_path" not in kwargs:
            resolved_file_path = workspace.get_active_image_file_path()
            if resolved_file_path:
                kwargs["file_path"] = resolved_file_path
                logger.debug(
                    "Dependency injection active: Set 'file_path' to target: %s",
                    resolved_file_path,
                )

        if component_type == "reader" and not isinstance(target, MSIBaseReader):
            reader_path = kwargs.get("file_path")
            if reader_path is None or not Path(reader_path).is_file():
                raise_validation_error(
                    context_name="ContextManager",
                    message=(
                        f"Cannot initialize a reader because image file '{reader_path}' "
                        "does not exist. Add the image to the workspace or pass an "
                        "existing imzML path before configuring the reader."
                    ),
                )

        ## Automatically inject forward binner reference if compiling an inverse spectrum binner
        if component_type == "inverse_binner" and "binner" not in kwargs:
            self._ensure_image_bucket(active_img)
            active_forward_binner = self.config_ledger[active_img].get("binner")
            if active_forward_binner:
                kwargs["binner"] = active_forward_binner
                logger.debug(
                    "Dependency injection active: Injected matching forward binner instance into constructor parameters."
                )

        ## Pass the active context proxy automatically if the component can accept it
        if "active_context" not in kwargs:
            kwargs["active_context"] = self._wrapper.active_context
            logger.debug(
                "Unified Context injection active for component: %s", component_type
            )

        # Workspace structure validation
        ## Trigger directory structural updates to prepare dedicated configuration layout folders
        if hasattr(workspace, "create_required_directories"):
            logger.debug(
                "Ensuring physical layout directories exist for image context: %s",
                active_img,
            )
            workspace.create_required_directories()

        # Strategy compilation block
        ## Delegate strategy selection to the unified validation framework factory
        logger.info(
            "Resolving system component '%s' under image context '%s'",
            component_type,
            active_img,
        )
        resolved_instance = resolve_component(
            target=target,
            registry=registries[component_type],
            component_type=component_type,
            expected_type=expected_types[component_type],
            **kwargs,
        )
        if getattr(resolved_instance, "active_context", None) is None:
            resolved_instance.active_context = self._wrapper.active_context

        # Ledger registration save sequence
        ## Map the initialized component instance into the memory state database container
        self._ensure_image_bucket(active_img)
        self.config_ledger[active_img][component_type] = resolved_instance
        logger.info(
            "Successfully registered component '%s' into ledger for image '%s'",
            component_type,
            active_img,
        )

        return resolved_instance

    # --------------------------------------------------
    # Subsection: Helpers - getters
    # --------------------------------------------------

    def __str__(self) -> str:
        """
        Generates a structured text representation of the internal configuration ledger.

        :return: Pretty-printed formatting of the committed image strategies.
        :rtype: str
        """
        # Formatter alignment sequence
        ## Execute structural dict transformation into readable multi-line layout string
        return pprint.pformat(self.config_ledger)

    def print_setup_summary(self) -> None:
        """
        Outputs highly detailed, structured state representations of the active ledger via logger.
        """
        # Render execution trace logs
        ## Direct standard string rendering from magic presentation layer down to system logger
        logger.info("ReadersManagerProxy Configuration Ledger Trace\n%s", str(self))

    def get_context_config(self, img_name: Optional[str] = None) -> Dict[str, Any]:
        """Return one local image context as a portable configuration dictionary.

        :param img_name: Image key to serialize. Uses the active image when omitted.
        :type img_name: Optional[str]
        :return: Reader, binner, and inverse-binner configuration for one image.
        :rtype: Dict[str, Any]
        :raises ValidationError: If no image context or ledger entry is available.
        """
        image_key = img_name or self._wrapper.workspace.active_img_name
        if not image_key:
            raise_validation_error(
                context_name="ContextManager",
                message="Cannot build image configuration without an image key.",
            )
        if image_key not in self.config_ledger:
            raise_validation_error(
                context_name="ContextManager",
                message=f"No local context is configured for image '{image_key}'.",
            )

        bucket = self.config_ledger[image_key]
        component_configs: Dict[str, Any] = {}
        for component_name in (
            "reader",
            "annotation_reader",
            "binner",
            "inverse_binner",
        ):
            component = bucket.get(component_name)
            if component is not None and not isinstance(component, dict):
                component_configs[component_name] = get_component_config(component)

        active_context = self._wrapper.active_context
        is_instantiated_context = (
            getattr(active_context, "_instantiated_image_key", None) == image_key
        )
        if is_instantiated_context and active_context.latent_reader is not None:
            component_configs["latent_reader"] = get_component_config(
                active_context.latent_reader
            )

        normalization = bucket.get("normalization")

        return {
            "schema_version": 1,
            "scope": "local_image",
            "image_key": image_key,
            "coordinate_order": self._wrapper.coordinate_order,
            "data_source": (
                active_context.data_source if is_instantiated_context else "image"
            ),
            "components": component_configs,
            "normalization": (
                normalization.get_config()
                if isinstance(normalization, NormalizationPipeline)
                else None
            ),
        }

    def load_context_config(
        self,
        config: Dict[str, Any],
        img_name_or_path: Optional[str] = None,
        base_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Restore the reader and binner pipeline owned by an image context.

        Saved reader paths are intentionally ignored. The workspace resolves
        the selected image through ``img_name_or_path`` or its default image.

        :param config: Portable local image-context configuration.
        :type config: Dict[str, Any]
        :param img_name_or_path: Optional image override. Uses the workspace
            default image when omitted.
        :type img_name_or_path: str | None
        :return: Restored component instances keyed by component role.
        :rtype: Dict[str, Any]
        :raises ValidationError: If required configuration sections are invalid.
        """
        if config.get("schema_version") != 1:
            raise_validation_error(
                "ContextConfiguration",
                f"Unsupported schema version '{config.get('schema_version')}'.",
            )
        components = config.get("components")
        if not isinstance(components, dict):
            raise_validation_error(
                "ContextConfiguration", "components must be a dictionary."
            )
        for required_name in ("reader", "binner"):
            if not isinstance(components.get(required_name), dict):
                raise_validation_error(
                    "ContextConfiguration",
                    f"A '{required_name}' component configuration is required.",
                )

        restored: Dict[str, Any] = {}
        reader_config = components["reader"]
        context_target = img_name_or_path
        configured_path = reader_config.get("parameters", {}).get("file_path")
        if configured_path:
            candidate = Path(configured_path)
            if not candidate.is_absolute() and base_path is not None:
                candidate = base_path / candidate
            workspace_path = None
            if img_name_or_path is not None:
                try:
                    workspace_path = self._wrapper.workspace._resolve_and_verify_image_file(
                        img_name_or_path
                    )
                except Exception:
                    workspace_path = None
            if workspace_path is None and candidate.is_file():
                context_target = str(candidate.resolve())
        reader_parameters = self._loadable_parameters(
            reader_config,
            excluded={"file_path", "active_context"},
        )
        restored["reader"] = self.set_reader(
            reader_config["type"],
            img_name_or_path=context_target,
            auto_load_annotations=False,
            **reader_parameters,
        )

        annotation_config = components.get("annotation_reader")
        if isinstance(annotation_config, dict):
            restored["annotation_reader"] = self.set_annotation_reader(
                annotation_config["type"],
                img_name_or_path=context_target,
                **self._loadable_parameters(
                    annotation_config,
                    excluded={"active_context"},
                ),
            )

        binner_config = components["binner"]
        restored["binner"] = self.set_binner(
            binner_config["type"],
            img_name_or_path=context_target,
            **self._loadable_parameters(
                binner_config,
                excluded={"active_context"},
            ),
        )

        inverse_config = components.get("inverse_binner")
        if isinstance(inverse_config, dict):
            restored["inverse_binner"] = self.set_inverse_binner(
                inverse_config["type"],
                img_name_or_path=context_target,
                **self._loadable_parameters(
                    inverse_config,
                    excluded={"active_context", "binner"},
                ),
            )
        normalization_config = config.get("normalization")
        if isinstance(normalization_config, dict):
            restored["normalization"] = self.set_normalization(
                normalization_config,
                img_name_or_path=img_name_or_path,
            )
        self._wrapper.workspace.set_active_image(context_target)
        self._wrapper.set_coordinate_order(config.get("coordinate_order", "xy"))
        return restored

    @staticmethod
    def _loadable_parameters(
        component_config: Dict[str, Any],
        excluded: set[str],
    ) -> Dict[str, Any]:
        """Return constructor parameters safe for runtime restoration.

        :param component_config: Portable component descriptor.
        :type component_config: Dict[str, Any]
        :param excluded: Runtime-injected parameter names to remove.
        :type excluded: set[str]
        :return: Independent constructor parameter dictionary.
        :rtype: Dict[str, Any]
        :raises ValidationError: If the descriptor is incomplete.
        """
        if not component_config.get("type"):
            raise_validation_error(
                "ContextConfiguration", "Every component requires a type."
            )
        parameters = component_config.get("parameters", {})
        if not isinstance(parameters, dict):
            raise_validation_error(
                "ContextConfiguration", "Component parameters must be a dictionary."
            )
        return {
            name: value for name, value in parameters.items() if name not in excluded
        }


# --------------------------------------------------
# Section: ReadersManagerMixin Injection Hook
# --------------------------------------------------


class ContextManagerMixin:
    """
    Mixin class designed to inject centralized reader management proxy features into the main wrapper context.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Instantiates the specialized reader manager proxy and threads it into the wrapper object context.
        """
        # Module instantiation hook
        ## Set reader manager tracking boundary attribute reference
        self.context_manager = ContextManagerProxy(wrapper_ref=self)
        super().__init__(*args, **kwargs)
