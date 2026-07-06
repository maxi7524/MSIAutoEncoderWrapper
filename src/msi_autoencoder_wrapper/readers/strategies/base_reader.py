from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Any


class MSIBaseReader(ABC):
    """
    Abstract Base Class outlining operational requirements for parsing Mass Spectrometry Imaging containers.
    
    Provides foundational spectrum geometry properties and hands off grid axis calculations
    directly to assigned active binning engines to ensure strategic optimization.
    """

    def __init__(self, file_path: Path | str) -> None:
        """
        Initializes foundational file tracking reference handles.

        :param file_path: Target disk storage file path targeting mass spectrometry image data.
        :type file_path: pathlib.Path | str
        """
        # Explicit conversion routing
        ## Coerce the input path into a concrete Path object to standardize path format across OS platforms
        self.file_path = Path(file_path)
        ## Package foundational configuration metadata details for serialization
        self._config: dict[str, Any] = {"file_path": str(file_path)}
        self._binner_ref: Any = None
        self._inverse_binner_ref: Any = None

    def GetConfig(self) -> dict[str, Any]:
        """
        Exposes internal parameter dictionaries required for automated pipeline construction.

        :return: Storage configuration definition parameters metadata map.
        :rtype: dict
        """
        return self._config

    def attach_binners(self, binner: Any = None, inverse_binner: Any = None) -> None:
        """
        Injects running execution reference hooks targeting active pipeline compression managers.

        :param binner: Primary spectrum forward compression binning engine instance, defaults to None.
        :type binner: Any, optional
        :param inverse_binner: Utilitarian reverse reconstruction spatial un-binning manager, defaults to None.
        :type inverse_binner: Any, optional
        """
        if binner is not None:
            self._binner_ref = binner
        if inverse_binner is not None:
            self._inverse_binner_ref = inverse_binner

    def _ensure_binner(self) -> None:
        """
        Evaluates internal operational safety requirements before routing discrete spatial queries.

        :raises ModelNotInitializedError: If grid conversion is requested without a runtime binner setup.
        """
        if self._binner_ref is None:
            from ...core.utils.exceptions import ModelNotInitializedError
            raise ModelNotInitializedError(
                "Grid calculation interrupted: Active binning context state has not been configured inside Wrapper."
            )

    @abstractmethod
    def GetXMin(self) -> float:
        """
        Extracts boundary starting thresholds recorded across continuous mass spectrum profiling sequences.

        :return: First evaluation scalar index stored inside mass axis vectors.
        :rtype: float
        """
        pass

    @abstractmethod
    def GetXMax(self) -> float:
        """
        Extracts maximum ending parameters bound within continuous mass axis arrays.

        :return: Terminal physical registration value located inside spectrum mass vectors.
        :rtype: float
        """
        pass

    @abstractmethod
    def GetXAxis(self) -> np.ndarray:
        """
        Extracts complete explicit coordinate vectors outlining physical mass spectrometry measurements sequences.

        :return: High-performance numpy data matrix referencing full layout mass axis channels.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def GetXAxisDepth(self) -> int:
        """
        Queries cumulative absolute spectral dimension capacities configured across spatial profiles.

        :return: Total mass axis channel array depth capacity value.
        :rtype: int
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Computes total flat spatial spectrum units (pixels) contained inside the target file.

        :return: Absolute measurement tracking cumulative pixel density inside dataset boundaries.
        :rtype: int
        """
        pass

    @abstractmethod
    def get_raw_spectrum(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts original irregular mass spectrometry arrays from explicit storage locations.

        :param idx: Flat coordinate index mapping targeting a unique tissue image pixel position.
        :type idx: int
        :return: Aligned coordinate pairing containing raw empirical vectors (xs, ys).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        pass

    @abstractmethod
    def GetSpectrumPosition(self, idx: int) -> tuple[int, int, int]:
        """
        Decodes flat array tracking sequences back into authentic 3D spatial pixel coordinates.

        :param idx: Flat position tracking sequence integer index.
        :type idx: int
        :return: Aligned discrete position coordinates mapping array across spatial axes [X, Y, Z].
        :rtype: tuple[int, int, int]
        """
        pass

    # --- GRID PROCESSING OPERATIONS DELEGATED EXPLICITLY TO ATTACHED STRATEGIES ---

    def GetGridXMin(self) -> Any:
        """Delegates optimized starting binned grid mass axis thresholds evaluation onto assigned binner."""
        self._ensure_binner()
        return self._binner_ref.GetXMin()

    def GetGridXMax(self) -> Any:
        """Delegates optimized terminal binned grid mass axis dimension boundary evaluation onto assigned binner."""
        self._ensure_binner()
        return self._binner_ref.GetXMax()

    def GetGridXAxis(self) -> np.ndarray:
        """Delegates complete evaluation of zunificated grid alignment arrays onto assigned binner."""
        self._ensure_binner()
        return self._binner_ref.GetXAxis()

    def GetGridXAxisDepth(self) -> int:
        """Delegates optimized grid target capacity limits calculation onto assigned binner."""
        self._ensure_binner()
        return self._binner_ref.GetXAxisDepth()