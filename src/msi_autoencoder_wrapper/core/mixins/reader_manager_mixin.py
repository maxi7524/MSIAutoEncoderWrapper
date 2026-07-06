"""
Module managing storage configurations, multi-image registries, and data reader tracking structures.
"""

import pprint
from typing import Dict, Any, Optional
from pathlib import Path
from ...utils.logger import get_custom_logger
from ...readers import ReaderManager
from ...binners import BinnerManager

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
    # Subsection: Setters - reader 
    # --------------------------------------------------

    def set_reader(self, img_name: str, target: Any, **kwargs: Any) -> None:
        """
        Registers a dataset reader profile config strategy into the central storage state ledger.

        :param img_name: Clean identifier handle representing the target image.
        :type img_name: str
        :param target: Registry string token name identifier or strategy class instance.
        :type target: Any
        """
        # Processing reader configuration pipeline
        ## Ensure target image storage bucket initialization
        self._ensure_image_bucket(img_name)
        
        ## Resolve component naming token signatures
        name_str = target if isinstance(target, str) else type(target).__name__
        
        ## Assign specific parameter configurations inside the tracking dictionary
        self.config_ledger[img_name]["reader"] = {
            "instance_name": name_str,
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

    def get_available_readers(self) -> Dict[str, Optional[str]]:
        """
        Extracts documentation strings across all registered data streaming readers.

        :return: Map correlating loader class names to their foundational documentation.
        :rtype: Dict[str, Optional[str]]
        """
        # Scan registration registries
        ## Retrieve explicit dictionary reference mappings from ReaderManager
        return {name: cls.__doc__ for name, cls in ReaderManager.REGISTRY.items()}
    
    # --------------------------------------------------
    # Subsection: Getters - binners 
    # --------------------------------------------------

    def get_available_binners(self) -> Dict[str, Optional[str]]:
        """
        Extracts documentation strings across all registered forward compression binners.

        :return: Map correlating compression class names to their foundational documentation.
        :rtype: Dict[str, Optional[str]]
        """
        # Scan registration registries
        ## Retrieve explicit dictionary reference mappings from BinnerManager
        return {name: cls.__doc__ for name, cls in BinnerManager.BINNER_REGISTRY.items()}

    def get_available_inverse_binners(self) -> Dict[str, Optional[str]]:
        """
        Extracts documentation strings across all registered reverse spatial binners.

        :return: Map correlating decompression class names to their foundational documentation.
        :rtype: Dict[str, Optional[str]]
        """
        # Scan registration registries
        ## Retrieve explicit dictionary reference mappings from BinnerManager INVERSE_REGISTRY
        return {name: cls.__doc__ for name, cls in BinnerManager.INVERSE_REGISTRY.items()}
    
    # --------------------------------------------------
    # Subsection: Helpers
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
        logger.info("=== ReadersManagerProxy Configuration Ledger Trace ===\n%s", str(self))

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