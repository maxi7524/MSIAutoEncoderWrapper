"""
Module managing storage configurations, multi-image registries, and data reader tracking structures.
"""

import pprint
import inspect
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

    def __init__(self) -> None:
        """
        Initializes the configuration ledger database holding data loader specifications.
        """
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

    def set_reader(self, img_name: str, reader_or_name: Union[str, Any], **kwargs: Any) -> None:
        """
        Sets and validates the reader configuration for a specific image context.
        
        If a pre-initialized object is passed, its registry compliance is verified.
        If a string key is provided, constructor signature validation is enforced 
        via reflection utilities before committing to the ledger.

        :param img_name: Clean identifier handle representing the target image.
        :type img_name: str
        :param reader_or_name: Registered strategy string name or an already instantiated reader object.
        :type reader_or_name: Union[str, Any]
        :param kwargs: Keyword arguments validated and saved as execution footprints.
        :raises ValueError: If the strategy name is unregistered or missing required parameters.
        """
        self._ensure_image_bucket(img_name)

        # Handle initialized class objects passed directly
        if not isinstance(reader_or_name, str):
            cls_name = reader_or_name.__class__.__name__
            if cls_name not in ReaderManager.REGISTRY:
                logger.warning(
                    f"Passed reader instance class '{cls_name}' is not explicitly found in ReaderManager.REGISTRY."
                )
            if kwargs:
                logger.warning(
                    f"Keyword arguments {list(kwargs.keys())} were provided but ignored "
                    f"because an already instantiated reader object was passed."
                )
            self.config_ledger[img_name]["reader"] = {
                "instance_name": cls_name,
                "instance_params": {}
            }
            self.config_ledger[img_name]["tmp"]["reader_instance"] = reader_or_name
            return

        # Handle text strategy configurations with lazy signature verification
        if reader_or_name not in ReaderManager.REGISTRY:
            raise ValueError(
                f"Unknown reader strategy identifier '{reader_or_name}'. "
                f"Available strategies in REGISTRY are: {list(ReaderManager.REGISTRY.keys())}."
            )

        reader_cls = ReaderManager.REGISTRY[reader_or_name]
        validate_constructor_kwargs(cls=reader_cls, name=reader_or_name, kwargs=kwargs)

        self.config_ledger[img_name]["reader"] = {
            "instance_name": reader_or_name,
            "instance_params": kwargs
        }
        logger.info("Committed dataset reader configuration setup under image token context: %s", img_name)

    # --------------------------------------------------
    # Subsection: Setters - binners 
    # --------------------------------------------------

    def set_binner(self, img_name: str, target: Any, **kwargs: Any) -> None:
        """
        Registers a forward compression spectrum binner configuration into the state ledger.

        :param img_name: Clean identifier handle representing the target image.
        :type img_name: str
        :param target: Registry string token name identifier or strategy class instance.
        :type target: Any
        """
        # Processing forward compression configuration pipeline
        ## Ensure target image storage bucket initialization
        self._ensure_image_bucket(img_name)
        
        ## Resolve component naming token signatures
        name_str = target if isinstance(target, str) else type(target).__name__
        
        ## Assign specific parameter configurations inside the tracking dictionary
        self.config_ledger[img_name]["binner"] = {
            "instance_name": name_str,
            "instance_params": kwargs
        }
        logger.info("Committed forward spectrum binner configuration setup under image token context: %s", img_name)

    def set_inverse_binner(self, img_name: str, target: Any, **kwargs: Any) -> None:
        """
        Registers a reverse reconstruction spatial binner configuration into the state ledger.

        :param img_name: Clean identifier handle representing the target image.
        :type img_name: str
        :param target: Registry string token name identifier or strategy class instance.
        :type target: Any
        """
        # Processing reverse reconstruction configuration pipeline
        ## Ensure target image storage bucket initialization
        self._ensure_image_bucket(img_name)
        
        ## Resolve component naming token signatures
        name_str = target if isinstance(target, str) else type(target).__name__
        
        ## Assign specific parameter configurations inside the tracking dictionary
        self.config_ledger[img_name]["inverse_binner"] = {
            "instance_name": name_str,
            "instance_params": kwargs
        }
        logger.info("Committed reverse reconstruction binner configuration setup under image token context: %s", img_name)

    # --------------------------------------------------
    # Subsection: Getters - readers 
    # --------------------------------------------------

    def get_available_readers(self, print_return: bool = True, return_value: bool=False) -> Dict[str, Dict[str, Any]]:
        """
        Extracts documentation strings across all registered data streaming readers.

        :param print_return: If True, prints a beautifully formatted summary of available components and parameters. Defaults to True.
        :type print_return: bool
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
    # Subsection: Getters - Private Helper
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

        return result
    


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
        self.reader_manager = ReadersManagerProxy()
        super().__init__(*args, **kwargs)