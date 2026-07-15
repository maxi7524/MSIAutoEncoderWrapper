"""
Module managing storage configurations, multi-image registries, and data reader tracking structures.
"""

import inspect
import pprint
from typing import Dict, Any, Optional, Union, Type
from pathlib import Path
from ..utils.decorators import manage_image_context
from ..utils.printing import extract_component_signatures, print_formatted_components
from ...utils.logger import get_custom_logger
from ...utils.validators import validate_constructor_kwargs, resolve_component
from ...readers.readers_manager import ReaderManager
from ...binners.binners_manager import BinnerManager

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
        logger.info("Enforcing automatic module discovery for reader and binner registries.")
        ReaderManager.discover_strategies()
        BinnerManager.discover_strategies()

# --------------------------------------------------
# Section: Public Strategy Setters
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Setters - reader
    # --------------------------------------------------

    def set_reader(self, reader_name_or_instance: Any, img_name_or_path: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Registers and configures an input data reader strategy for an image context.

        Accepts either a unique registered string token identifier or a pre-initialized
        concrete instance object. Context resolution and filesystem path injections are
        delegated directly down to the internal component registration system.

        :param reader_name_or_instance: Registered strategy identifier string or an initialized reader object.
        :type reader_name_or_instance: Any
        :param img_name_or_path: Explicit target path or standalone image name key token. Defaults to None.
        :type img_name_or_path: Optional[str]
        :param kwargs: Arbitrary configuration parameters validated and passed to the constructor factory.
        :return: Fully resolved and validated data reader component instance.
        :rtype: Any
        """
        # Strategy routing block
        ## Forward execution properties directly to the unified driver registration manager
        return self._set_component(
            component_type="reader",
            target=reader_name_or_instance,
            img_name_or_path=img_name_or_path,
            **kwargs
        )
    
    # --------------------------------------------------
    # Subsection: Setters - binners
    # --------------------------------------------------

    def set_binner(self, binner_name_or_instance: Any, img_name_or_path: Optional[str] = None, **kwargs: Any) -> Any:
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
            **kwargs
        )

    def set_inverse_binner(self, inverse_binner_name_or_instance: Any, img_name_or_path: Optional[str] = None, **kwargs: Any) -> Any:
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
            **kwargs
        )


# --------------------------------------------------
# Section: Public Strategy Getters
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Getters - readers 
    # --------------------------------------------------

    def get_available_readers(self, print_return: bool = True, return_value: bool=False) -> Dict[str, Dict[str, Any]]:
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
        return self._get_available_components_info(
            registry=ReaderManager.REGISTRY,
            title="REGISTERED MSI READERS & PARAMETERS",
            key_label="Reader Key",
            print_return=print_return,
            return_value=return_value
        )
        
    # --------------------------------------------------
    # Subsection: Getters - binners 
    # --------------------------------------------------

    def get_available_binners(self, print_return: bool = True, return_value: bool=False) -> Dict[str, Dict[str, Any]]:
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
        return self._get_available_components_info(
            registry=BinnerManager.BINNER_REGISTRY,
            title="REGISTERED MSI FORWARD BINNERS & PARAMETERS",
            key_label="Binner Key",
            print_return=print_return,
            return_value=return_value
        )

    def get_available_inverse_binners(self, print_return: bool = True, return_value: bool=False) -> Dict[str, Dict[str, Any]]:
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
        return self._get_available_components_info(
            registry=BinnerManager.INVERSE_REGISTRY,
            title="REGISTERED MSI INVERSE BINNERS & PARAMETERS",
            key_label="Inverse Binner Key",
            print_return=print_return,
            return_value=return_value
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
                "binner": {"instance_name": "", "instance_params": {}},
                "inverse_binner": {"instance_name": "", "instance_params": {}},
                "tmp": {}
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
        **kwargs: Any
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
            "binner": BinnerManager.BINNER_REGISTRY,
            "inverse_binner": BinnerManager.INVERSE_REGISTRY
        }

        if component_type not in registries:
            ### Handle invalid component classification anomalies
            logger.error("Component registration blocked: Unsupported component type '%s'", component_type)
            raise ValueError(f"Unsupported component type: {component_type}. Valid types: {list(registries.keys())}")

        workspace = self._wrapper.workspace
        active_img = workspace.active_img_name

        # Dependency injection layer
        ## Automatically inject the active filesystem path if compiling a data loader reader
        if component_type == "reader" and "file_path" not in kwargs:
            resolved_file_path = workspace.get_active_image_file_path()
            if resolved_file_path:
                kwargs["file_path"] = resolved_file_path
                logger.debug("Dependency injection active: Set 'file_path' to target: %s", resolved_file_path)

        ## Automatically inject forward binner reference if compiling an inverse spectrum binner
        if component_type == "inverse_binner" and "binner" not in kwargs:
            self._ensure_image_bucket(active_img)
            active_forward_binner = self.config_ledger[active_img].get("binner")
            if active_forward_binner:
                kwargs["binner"] = active_forward_binner
                logger.debug("Dependency injection active: Injected matching forward binner instance into constructor parameters.")

        ## Pass the active context proxy automatically if the component can accept it
        if "active_context" not in kwargs:
            kwargs["active_context"] = self._wrapper.active_context
            logger.debug("Unified Context injection active for component: %s", component_type)

        # Workspace structure validation
        ## Trigger directory structural updates to prepare dedicated configuration layout folders
        if hasattr(workspace, "create_required_directories"):
            logger.debug("Ensuring physical layout directories exist for image context: %s", active_img)
            workspace.create_required_directories()

        # Constructor signature verification
        ## Extract target class type from registry if lookup target is provided as a string token
        target_registry = registries[component_type]
        if isinstance(target, str) and target in target_registry:
            logger.debug("Executing static reflection analysis against constructor of class: %s", target)
            validate_constructor_kwargs(target_registry[target], target, kwargs)

        # Strategy compilation block
        ## Delegate strategy selection to the unified validation framework factory
        logger.info("Resolving system component '%s' under image context '%s'", component_type, active_img)
        
        try:
            resolved_instance = resolve_component(
                target=target,
                registry=target_registry,
                component_type=component_type,
                **kwargs
            )
        except Exception as error:
            ### Catch and log resolution errors before propagating exceptions
            logger.error("Failed to resolve component '%s' for target context.", component_type, exc_info=True)
            raise error

        # Ledger registration save sequence
        ## Map the initialized component instance into the memory state database container
        self._ensure_image_bucket(active_img)
        self.config_ledger[active_img][component_type] = resolved_instance
        logger.info("Successfully registered component '%s' into ledger for image '%s'", component_type, active_img)

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

    def _get_available_components_info(
        self, 
        registry: Dict[str, Any], 
        title: str, 
        key_label: str, 
        print_return: bool,
        return_value: bool
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Internal helper utility to extract documentation and constructor signatures across registries.
        Delegated to unified printing module to avoid duplicate inspector code.

        :param registry: Target manager registry dictionary mapping keys to component classes.
        :type registry: Dict[str, Any]
        :param title: Header string used during beautiful print formatting sequences.
        :type title: str
        :param key_label: Contextual descriptor label pointing to the strategy type.
        :type key_label: str
        :param print_return: Flag determining whether data logs are pushed to stdout streams.
        :type print_return: bool
        :param return_value: Flag determining whether return dict with data logs.
        :type return_value: bool
        :return: Deeply nested mapping matching strategy aliases to structural property states.
        :rtype: Optional[Dict[str, Dict[str, Any]]]
        """

        result = extract_component_signatures(registry=registry)

        # Output Flow Redirection
        if print_return:
            print_formatted_components(
                title=title,
                key_label=key_label,
                components_info=result
            )

        if return_value:
            return result
        return None
    


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