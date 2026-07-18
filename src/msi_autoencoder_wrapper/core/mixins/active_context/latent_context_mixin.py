"""Independent latent-space source management for the active context."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from ....latent.imzml_store import LatentImzMLStore
from ....readers.base_reader import MSIBaseReader
from ....readers.readers_manager import ReaderManager
from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)
DataSource = Literal["image", "latent"]


class LatentContextMixin:
    """Add a separately loaded latent reader and an explicit source selector."""

    def _initialize_latent_context(self) -> None:
        """Initialize latent state without resolving the original image."""
        self._latent_reader: Optional[MSIBaseReader] = None
        self._data_source: DataSource = "image"

    @property
    def latent_reader(self) -> Optional[MSIBaseReader]:
        """Return the loaded latent reader without resolving the image reader."""
        return self._latent_reader

    @property
    def data_source(self) -> DataSource:
        """Return the currently selected data source."""
        return self._data_source

    @property
    def data_reader(self) -> MSIBaseReader:
        """Return only the reader selected by :attr:`data_source`."""
        return self.get_data_reader()

    def set_data_source(self, source: DataSource) -> None:
        """Select image or latent data for spatial operations and datasets.

        :param source: ``image`` or ``latent``.
        :type source: Literal["image", "latent"]
        :raises ValidationError: If the source is invalid or latent is not loaded.
        """
        if source not in {"image", "latent"}:
            raise_validation_error(
                context_name="ActiveContext",
                message="Data source must be either 'image' or 'latent'.",
            )
        if source == "latent" and self._latent_reader is None:
            raise_validation_error(
                context_name="LatentContext",
                message="No latent imzML source has been loaded.",
            )
        self._data_source = source

    def load_latent(
        self,
        file_path: Path | str,
        reader: Any = "PyImzMLReader",
        activate: bool = True,
    ) -> MSIBaseReader:
        """Load a latent imzML source without touching the original image reader.

        :param file_path: Latent imzML path.
        :type file_path: pathlib.Path | str
        :param reader: Reader key, class, or ready reader instance.
        :type reader: Any
        :param activate: Select latent as the active data source.
        :type activate: bool
        :return: Loaded latent reader.
        :rtype: MSIBaseReader
        """
        ReaderManager.discover_strategies()
        kwargs = {"file_path": file_path, "active_context": self}
        if not isinstance(reader, str) and not isinstance(reader, type):
            kwargs = {}
        self._latent_reader = ReaderManager.get_reader(reader, **kwargs)
        if getattr(self._latent_reader, "active_context", None) is None:
            self._latent_reader.active_context = self
        if activate:
            self._data_source = "latent"
        logger.info("Loaded latent-space source: %s", file_path)
        return self._latent_reader

    def unload_latent(self) -> None:
        """Release the latent reader and return spatial operations to image data."""
        self._latent_reader = None
        self._data_source = "image"

    def get_data_reader(self, source: Optional[DataSource] = None) -> MSIBaseReader:
        """Resolve one selected reader without loading the other source.

        :param source: Optional source override.
        :type source: Optional[Literal["image", "latent"]]
        :return: Reader for the selected space.
        :rtype: MSIBaseReader
        """
        selected = source or self._data_source
        if selected == "latent":
            if self._latent_reader is None:
                raise_validation_error(
                    context_name="LatentContext",
                    message="No latent imzML source has been loaded.",
                )
            return self._latent_reader
        if selected != "image":
            raise_validation_error(
                context_name="ActiveContext",
                message=f"Unsupported data source '{selected}'.",
            )
        return self.reader

    def get_spectrum(self, target: Any, source: Optional[DataSource] = None) -> Any:
        """Read one spectrum or slice from the selected image-like source.

        :param target: Spectrum index, spatial coordinate, or slice expression.
        :type target: Any
        :param source: Optional image or latent source override.
        :type source: Optional[Literal["image", "latent"]]
        :return: Spectrum data or a spatial selection from the selected reader.
        :rtype: Any
        """
        return self.get_data_reader(source)[target]

    def get_region(
        self,
        first: slice | int = slice(None),
        second: slice | int = slice(None),
        z: slice | int = slice(None),
        source: Optional[DataSource] = None,
    ) -> Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]:
        """Read a spatial region from the selected image-like source.

        :param first: X coordinate or matrix row selector.
        :type first: slice | int
        :param second: Y coordinate or matrix column selector.
        :type second: slice | int
        :param z: Z coordinate selector.
        :type z: slice | int
        :param source: Optional image or latent source override.
        :type source: Optional[Literal["image", "latent"]]
        :return: Mapping from user coordinates to spectra.
        :rtype: Dict[Tuple[int, int, int], Tuple[numpy.ndarray, numpy.ndarray]]
        """
        return self.get_data_reader(source).get_region(first, second, z)

    def save_latent(
        self,
        output_path: Optional[Path | str] = None,
        loader_config: Optional[Dict[str, Any]] = None,
        activate: bool = True,
    ) -> Path:
        """Transform the active image and save/load its latent imzML representation.

        :param output_path: Optional destination; defaults to the active model latent folder.
        :type output_path: Optional[pathlib.Path | str]
        :param loader_config: Optional DataLoader overrides for transformation.
        :type loader_config: Optional[Dict[str, Any]]
        :param activate: Load and select the newly written latent source.
        :type activate: bool
        :return: Written latent imzML path.
        :rtype: pathlib.Path
        """
        autoencoder = self.autoencoder
        if autoencoder is None:
            raise_validation_error(
                context_name="LatentContext",
                message="The currently loaded model is not an autoencoder.",
            )
        source_reader = self.get_data_reader("image")
        latent_values = autoencoder.transform(loader_config)
        target = Path(output_path) if output_path is not None else self._default_latent_path()
        written_path = LatentImzMLStore.write(target, latent_values, source_reader)
        if activate:
            self.load_latent(written_path, activate=True)
        return written_path

    def _default_latent_path(self) -> Path:
        """Resolve the active image/model latent output path."""
        image_key = self._instantiated_image_key or self._wrapper.workspace.active_img_name
        model_name = getattr(self._wrapper.models_manager, "_active_model_name", None)
        if not image_key or not model_name:
            raise_validation_error(
                context_name="LatentContext",
                message="An active image key and loaded model name are required.",
            )
        latent_dir = self._wrapper.workspace.get_latent_dir(image_key, model_name)
        return latent_dir / f"{image_key}.latent.imzML"
