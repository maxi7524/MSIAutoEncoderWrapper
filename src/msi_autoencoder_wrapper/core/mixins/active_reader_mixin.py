"""
Module defining the execution proxy and mixin for managing the active mass spectrometry reader context.
"""

from typing import Any, Optional, Dict
import numpy as np
from ...utils.logger import get_custom_logger
from ...readers.readers_manager import ReaderManager
from ...binners.binners_manager import BinnerManager
from ..utils.exceptions import ValidationError

logger = get_custom_logger(__name__)


# --------------------------------------------------
# Section: ActiveReaderProxy Operational Bridge
# --------------------------------------------------

class ActiveReaderProxy:
    """
    Command proxy coordinating runtime execution, lazy loading, and state delegation for the active image.
    """

    def __init__(self, wrapper_ref: Any) -> None:
        """
        Initializes the active execution proxy bounded to the parent wrapper instance.

        :param wrapper_ref: Reference to the hosting main MSIAutoEncoderWrapper instance.
        :type wrapper_ref: Any
        """
        self._wrapper = wrapper_ref
        self._instantiated_image_key: Optional[str] = None
        self._active_reader: Optional[Any] = None
        self._active_binner: Optional[Any] = None
        self._active_inverse_binner: Optional[Any] = None

    def _sync_active_pipeline(self) -> None:
        """
        Synchronizes initialized binary objects against the current active image token in workspace.
        """
        # Context tracking lookup execution
        ## Extract active state identifier from the connected workspace manager
        current_img: Optional[str] = self._wrapper.workspace._active_image

        if not current_img:
            raise ValidationError(["Active Image Context [Not Selected in Workspace]"])

        ## Evaluate whether context switch or initial loading trigger is required
        if current_img != self._instantiated_image_key:
            ### Execute safe eviction protocol to release low-level file handles
            if self._active_reader is not None:
                logger.info("Evicting active binary reader instance to release file hooks for context: %s", self._instantiated_image_key)
                self._active_reader = None
                self._active_binner = None
                self._active_inverse_binner = None

            ### Assert ledger configuration availability inside the central registry
            if current_img not in self._wrapper.reader_manager.config_ledger:
                raise ValidationError([f"Configuration ledger registry entry for image '{current_img}' [Missing]"])

            ### Resolve instance properties from storage ledger records
            ledger_entry = self._wrapper.reader_manager.config_ledger[current_img]
            self._instantiated_image_key = current_img

            #### Factory initialization execution for target mass spectrometry reader
            r_cfg = ledger_entry.get("reader", {})
            if r_cfg.get("instance_name"):
                logger.debug("Lazy-loading reader engine component via factory identifier: %s", r_cfg["instance_name"])
                self._active_reader = ReaderManager.get_reader(
                    r_cfg["instance_name"],
                    **r_cfg.get("instance_params", {})
                )

            #### Factory initialization execution for forward processing binner
            b_cfg = ledger_entry.get("binner", {})
            if b_cfg.get("instance_name") and self._active_reader is not None:
                b_name = b_cfg["instance_name"]
                if b_name in BinnerManager.REGISTRY:
                    logger.debug("Lazy-loading forward binner strategy via registry lookup: %s", b_name)
                    self._active_binner = BinnerManager.REGISTRY[b_name](**b_cfg.get("instance_params", {}))

            #### Factory initialization execution for inverse reconstruction binner
            ib_cfg = ledger_entry.get("inverse_binner", {})
            if ib_cfg.get("instance_name") and self._active_reader is not None:
                ib_name = ib_cfg["instance_name"]
                if hasattr(BinnerManager, "INVERSE_REGISTRY") and ib_name in BinnerManager.INVERSE_REGISTRY:
                    logger.debug("Lazy-loading inverse binner strategy via registry lookup: %s", ib_name)
                    self._active_inverse_binner = BinnerManager.INVERSE_REGISTRY[ib_name](**ib_cfg.get("instance_params", {}))

            #### Interconnect instantiated structural execution drivers together
            if self._active_reader and hasattr(self._active_reader, "attach_binners"):
                self._active_reader.attach_binners(
                    binner=self._active_binner,
                    inverse_binner=self._active_inverse_binner
                )

    @property
    def tmp_values(self) -> Dict[str, Any]:
        """
        Exposes volatile RAM-persistent cache memory bucket allocated for the active image context.

        :return: Reference mapping tracking runtime temporary calculation matrices.
        :rtype: Dict[str, Any]
        """
        self._sync_active_pipeline()
        current_img = self._instantiated_image_key
        return self._wrapper.reader_manager.config_ledger[current_img]["tmp"]  # type: ignore

    # --------------------------------------------------
    # Section: Delegated Reader Core Execution Methods
    # --------------------------------------------------

    # --------------------------------------------------
    # Section: Dynamic Command Routing & Magic Fallbacks
    # --------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically intercepts and routes all method calls and attribute queries directly 
        to the active backend components, ensuring seamless alignment without explicit delegation.

        :param name: Name of the requested attribute or method string token.
        :type name: str
        :return: Dynamically resolved attribute pointer or executable method bound context.
        :rtype: Any
        """
        # Execution routing pipeline
        ## Trigger state synchronization before executing attribute lookup queries
        self._sync_active_pipeline()

        # Prioritize resolution from the primary spectrum reader component instance
        if self._active_reader is not None and hasattr(self._active_reader, name):
            return getattr(self._active_reader, name)

        # Fallback resolution mapping queries onto the forward spectrum binner engine
        if self._active_binner is not None and hasattr(self._active_binner, name):
            return getattr(self._active_binner, name)

        # Raise standard attribute errors if the key is absent across all domain strategies
        raise AttributeError(
            f"'{type(self).__name__}' proxy context has no attribute or delegated method named '{name}' "
            f"configured within the active reader or binner strategies."
        )

    def __dir__(self) -> list[str]:
        """
        Overrides the default dir() behavior to dynamically expose attributes and methods 
        from both the active reader and binner components for IDE autocompletion.

        :return: A combined list of all available attribute and method names.
        :rtype: list[str]
        """
        # Core collection sequence
        ## Gather static local attributes defined directly within the proxy instance
        local_attrs = set(super().__dir__())
        
        try:
            ### Trigger safe synchronization to ensure backends are fully populated
            self._sync_active_pipeline()
            
            ### Aggregate attributes discovered inside the instantiated reader pipeline
            if self._active_reader is not None:
                local_attrs.update(dir(self._active_reader))
                
            ### Aggregate attributes discovered inside the instantiated forward compression binner
            if self._active_binner is not None:
                local_attrs.update(dir(self._active_binner))
        except Exception:
            #### Silent fallback during initialization or if context is missing during inspection
            pass

        return sorted(list(local_attrs))

    def __len__(self) -> int:
        """
        Interceptors routing standard length evaluations onto the active master dataset reader.

        :return: Total number of spatial spectra pixels contained within the dataset boundaries.
        :rtype: int
        """
        self._sync_active_pipeline()
        if self._active_reader is None:
            raise ValidationError(["Active reader initialization failed. Length is unresolved."])
        return len(self._active_reader)

    # Depracated
    # def GetXValues(self, idx: int) -> np.ndarray:
    #     """
    #     Queries aligned mass spectrometry response odciete variables arrays.

    #     :param idx: Flat coordinate position sequential pointer targeting a single tracking node.
    #     :type idx: int
    #     :return: Array containing matching physical mass-to-charge axis dimensions.
    #     :rtype: np.ndarray
    #     """
    #     self._sync_active_pipeline()
    #     return self._active_reader.GetXValues(idx)

    # def GetYValues(self, idx: int) -> np.ndarray:
    #     """
    #     Queries aligned mass spectrometry response intensity parameters arrays.

    #     :param idx: Flat coordinate position sequential pointer targeting a single tracking node.
    #     :type idx: int
    #     :return: Array containing matching response amplitude signal counts.
    #     :rtype: np.ndarray
    #     """
    #     self._sync_active_pipeline()
    #     return self._active_reader.GetYValues(idx)

    # def GetSpectrumPosition(self, idx: int) -> tuple[int, int, int]:
    #     """
    #     Decodes flat position tracking sequences back into spatial tissue pixel coordinates.

    #     :param idx: Flat position tracking sequence integer index.
    #     :type idx: int
    #     :return: Aligned physical coordinate components across axes [X, Y, Z].
    #     :rtype: tuple[int, int, int]
    #     """
    #     self._sync_active_pipeline()
    #     return self._active_reader.GetSpectrumPosition(idx)

    # # --------------------------------------------------
    # # Section: Delegated Binner Grid Execution Methods
    # # --------------------------------------------------

    # def GetGridXMin(self) -> Any:
    #     """
    #     Evaluates starting binned grid mass axis thresholds optimization bounds.

    #     :return: Minimum boundary value of the compiled unified processing mesh grid.
    #     :rtype: Any
    #     """
    #     self._sync_active_pipeline()
    #     return self._active_reader.GetGridXMin()

    # def GetGridXMax(self) -> Any:
    #     """
    #     Evaluates terminal binned grid mass axis thresholds optimization bounds.

    #     :return: Maximum boundary value of the compiled unified processing mesh grid.
    #     :rtype: Any
    #     """
    #     self._sync_active_pipeline()
    #     return self._active_reader.GetGridXMax()

    # def GetGridXAxis(self) -> np.ndarray:
    #     """
    #     Evaluates the complete zunificated calibration grid alignment matrix tracking arrays.

    #     :return: Precalculated uniform reference vector applied across pixel arrays.
    #     :rtype: np.ndarray
    #     """
    #     self._sync_active_pipeline()
    #     return self._active_reader.GetGridXAxis()

    # def GetGridXAxisDepth(self) -> int:
    #     """
    #     Calculates optimized structural capacity limits dimensionality for neural networks input layers.

    #     :return: Number of distinct discrete bins forming the processed spectrum vector.
    #     :rtype: int
    #     """
    #     self._sync_active_pipeline()
    #     return self._active_reader.GetGridXAxisDepth()


# --------------------------------------------------
# Section: ActiveReaderMixin Injection Hook
# --------------------------------------------------

class ActiveReaderMixin:
    """
    Mixin class injecting execution workspace proxy controls targeting active image dataset streaming.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Instantiates active pipeline routing interfaces attached to the execution wrapper context.
        """
        # Module instantiation hook
        ## Set active reader command bridge attribute reference proxy
        self.active_reader = ActiveReaderProxy(wrapper_ref=self)
        super().__init__(*args, **kwargs)