"""
Module managing storage configurations, multi-image registries, and data reader tracking structures.
"""

import inspect
import pprint
from typing import Dict, Any, Optional, Union 
from pathlib import Path
from ...utils.logger import get_custom_logger
from ...utils.validators import validate_constructor_kwargs
from ...readers.readers_manager import ReaderManager
from ...binners.binners_manager import BinnerManager

logger = get_custom_logger(__name__)


# --------------------------------------------------
# Section: ReadersManagerProxy Infrastructure
# --------------------------------------------------

class ReadersManagerProxy:
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
    # Subsection: Setters - reader 
    # --------------------------------------------------

    def set_reader(self, img_name: str, reader_name_or_instance: Union[str, Any], **kwargs: Any) -> None:
        """
        Sets and validates the reader configuration for a specific image context.
        
        If a pre-initialized object is passed, its registry compliance is verified.
        If a string key is provided, constructor signature validation is enforced 
        via reflection utilities before committing to the ledger.

        :param img_name: Clean identifier handle representing the target image.
        :type img_name: str
        :param reader_name_or_instance: Registered strategy string name or an already instantiated reader object.
        :type reader_name_or_instance: Union[str, Any]
        :param kwargs: Keyword arguments validated and saved as execution footprints.
        :raises ValueError: If the strategy name is unregistered or missing required parameters.
        """
        self._set_component(
            img_name=img_name,
            target=reader_name_or_instance,
            registry=ReaderManager.REGISTRY,
            ledger_key="reader",
            component_type_label="reader",
            kwargs=kwargs
        )
        logger.info("Committed dataset reader configuration setup under image token context: %s", img_name)

    # --------------------------------------------------
    # Subsection: Setters - binners 
    # --------------------------------------------------

    def set_binner(self, img_name: str, binner_name_or_instance: Any, **kwargs: Any) -> None:
        """
        Registers a forward compression spectrum binner configuration into the state ledger.

        :param img_name: Clean identifier handle representing the target image.
        :type img_name: str
        :param binner_name_or_instance: Registry string token name identifier or strategy class instance.
        :type binner_name_or_instance: Any
        """
        # Processing forward compression configuration pipeline
        self._set_component(
            img_name=img_name,
            target=binner_name_or_instance,
            registry=BinnerManager.BINNER_REGISTRY,
            ledger_key="binner",
            component_type_label="forward binner",
            kwargs=kwargs
        )
        logger.info("Committed forward spectrum binner configuration setup under image token context: %s", img_name)

    def set_inverse_binner(self, img_name: str, inverse_binner_name_or_instance: Any, **kwargs: Any) -> None:
        """
        Registers a reverse reconstruction spatial binner configuration into the state ledger.

        :param img_name: Clean identifier handle representing the target image.
        :type img_name: str
        :param inverse_binner_name_or_instance: Registry string token name identifier or strategy class instance.
        :type inverse_binner_name_or_instance: Any
        """
        # Processing reverse reconstruction configuration pipeline
        self._set_component(
            img_name=img_name,
            target=inverse_binner_name_or_instance,
            registry=BinnerManager.INVERSE_REGISTRY,
            ledger_key="inverse_binner",
            component_type_label="inverse binner",
            kwargs=kwargs
        )
        logger.info("Committed reverse reconstruction binner configuration setup under image token context: %s", img_name)

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
    # Subsection: Helpers
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

    def _set_component(
        self,
        target: Any,
        img_name: str,
        registry: Dict[str, Any],
        ledger_key: str,
        component_type_label: str,
        kwargs: Dict[str, Any]
    ) -> None:
        """
        Centralized internal helper to manage contextual pipeline activation, type verification,
        constructor reflection signatures, and state ledger preservation across all component types.
        """

        try:
            # Uruchomienie metody z workspace mixin za pomocą referencji do rodzica
            if self._wrapper and hasattr(self._wrapper, "workspace"):
                self._wrapper.workspace.set_active_image(img_name)
                img_name = self._wrapper.workspace.get_
                ## Ensure target image storage bucket initialization
                self._ensure_image_bucket(img_name)

            # Handle initialized class objects passed directly
            if not isinstance(target, str):
                cls_name = target.__class__.__name__
                if cls_name not in registry:
                    logger.warning(
                        f"Passed {component_type_label} instance class '{cls_name}' is not explicitly found in registry."
                    )
                if kwargs:
                    logger.warning(
                        f"Keyword arguments {list(kwargs.keys())} were provided but ignored "
                        f"because an already instantiated {component_type_label} object was passed."
                    )
                self.config_ledger[img_name][ledger_key] = {
                    "instance_name": cls_name,
                    "instance_params": {}
                }
                return

            # Handle text strategy configurations with lazy signature verification
            if target not in registry:
                raise ValueError(
                    f"Unknown {component_type_label} strategy identifier '{target}'. "
                    f"Available strategies are: {list(registry.keys())}."
                )

            component_cls = registry[target]
            validate_constructor_kwargs(cls=component_cls, name=target, kwargs=kwargs)

            ## Assign specific parameter configurations inside the tracking dictionary
            self.config_ledger[img_name][ledger_key] = {
                "instance_name": target,
                "instance_params": kwargs
            }
            logger.info("Committed %s configuration setup under image token context: %s", component_type_label, img_name)

        finally:
            # Kontekst po całej akcji musi być bezwzględnie usuwany
            if self._wrapper and hasattr(self._wrapper, "workspace"):
                self._wrapper.workspace.set_active_image(None)

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
        ) -> Dict[str, Dict[str, Any]]:
        """
        Internal helper utility to extract documentation and constructor signatures across registries.

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
        :rtype: Dict[str, Dict[str, Any]]
        """
        result = {}
        
        for name, cls in registry.items():
            init_method = getattr(cls, "__init__", None)
            params = {}
            if init_method:
                try:
                    sign = inspect.signature(init_method)
                    for param_name, param in sign.parameters.items():
                        if param_name in ("self", "args", "kwargs"):
                            continue
                        default_val = "Required" if param.default == inspect.Parameter.empty else param.default
                        params[param_name] = default_val
                except (ValueError, TypeError):
                    pass
            
            result[name] = {
                "docstring": cls.__doc__,
                "parameters": params
            }

        if print_return:
            print("\n" + "=" * 80)
            print(f" {title}")
            print("=" * 80)
            for name, info in result.items():
                print(f"\n[{key_label}]: '{name}'")
                print(f" Description: {info['docstring'].strip() if info['docstring'] else 'No documentation provided.'}")
                print(" Parameters (kwargs):")
                if info["parameters"]:
                    for p_name, p_default in info["parameters"].items():
                        print(f"   - {p_name}: {p_default}")
                else:
                    print("   - None")
            print("=" * 80 + "\n")
        if return_value:
            return result
        else:
            pass
    


# --------------------------------------------------
# Section: ReadersManagerMixin Injection Hook
# --------------------------------------------------

class ReadersManagerMixin:
    """
    Mixin class designed to inject centralized reader management proxy features into the main wrapper context.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Instantiates the specialized reader manager proxy and threads it into the wrapper object context.
        """
        # Module instantiation hook
        ## Set reader manager tracking boundary attribute reference
        self.reader_manager = ReadersManagerProxy(wrapper_ref=self)
        super().__init__(*args, **kwargs)