"""Write latent-space images as standard imzML/ibd pairs."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from pyimzml.ImzMLWriter import ImzMLWriter

from ..readers.base_reader import MSIBaseReader
from ..utils.exceptions import raise_validation_error, raise_workspace_error
from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class LatentImzMLStore:
    """Persist dense latent vectors using source-image spatial coordinates."""

    @staticmethod
    def write(
        output_path: Path | str,
        latent_values: np.ndarray,
        source_reader: MSIBaseReader,
    ) -> Path:
        """Write latent vectors with source positions and embedded source metadata.

        :param output_path: Destination imzML path; pyimzML creates the matching ibd.
        :type output_path: pathlib.Path | str
        :param latent_values: Matrix shaped ``[spectra, latent dimensions]``.
        :type latent_values: numpy.ndarray
        :param source_reader: Reader providing original spatial coordinates and metadata.
        :type source_reader: MSIBaseReader
        :return: Written imzML path.
        :rtype: pathlib.Path
        :raises ValidationError: If latent dimensions do not match source spectra.
        :raises WorkspaceConfigError: If pyimzML cannot write the output pair.
        """
        values = np.asarray(latent_values, dtype=np.float32)
        if values.ndim != 2:
            raise_validation_error(
                context_name="LatentStore",
                message="Latent values must have shape [spectra, latent dimensions].",
            )
        spectrum_count = source_reader.GetNumberOfSpectra()
        if values.shape[0] != spectrum_count:
            raise_validation_error(
                context_name="LatentStore",
                message=(
                    f"Latent rows ({values.shape[0]}) do not match source spectra "
                    f"({spectrum_count})."
                ),
            )

        target = Path(output_path).with_suffix(".imzML")
        target.parent.mkdir(parents=True, exist_ok=True)
        latent_axis = np.arange(values.shape[1], dtype=np.float32)
        source_metadata = escape(
            json.dumps(source_reader.GetMetaData(), default=str),
            {'"': "&quot;"},
        )
        source_file = escape(str(source_reader.file_path), {'"': "&quot;"})
        shared_parameters = [
            {"name": "msi_autoencoder_wrapper_space", "value": "latent"},
            {"name": "msi_autoencoder_wrapper_axis", "value": "latent_feature_index"},
            {"name": "msi_autoencoder_wrapper_latent_dimensions", "value": str(values.shape[1])},
            {"name": "msi_autoencoder_wrapper_source_file", "value": source_file},
            {"name": "msi_autoencoder_wrapper_source_metadata", "value": source_metadata},
        ]

        try:
            with ImzMLWriter(
                str(target),
                mode="processed",
                spec_type="profile",
                intensity_dtype=np.float32,
            ) as writer:
                for spectrum_index, latent_vector in enumerate(values):
                    writer.addSpectrum(
                        latent_axis,
                        latent_vector,
                        source_reader.GetSpectrumPosition(spectrum_index),
                        userParams=shared_parameters,
                    )
        except Exception as error:
            target.unlink(missing_ok=True)
            target.with_suffix(".ibd").unlink(missing_ok=True)
            raise_workspace_error(
                context_name="LatentStore",
                message=f"Failed to write latent imzML/ibd pair '{target}': {error}",
            )
        logger.info("Latent imzML/ibd pair saved at: %s", target)
        return target
