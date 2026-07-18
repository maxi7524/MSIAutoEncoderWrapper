"""
Module providing runtime context attachments and operational execution proxies for autoencoder models.
"""

from pathlib import Path
from typing import Any, Optional, Dict, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from ....utils.logger import get_custom_logger
from ....utils.exceptions import raise_validation_error

# Logger initialization
logger = get_custom_logger(__name__)


class AutoencoderContextInterface:
    """
    High-level operational execution proxy providing concrete interface methods for locally mounted autoencoders.
    """

    def __init__(
        self,
        torch_model: torch.nn.Module,
        active_context: Any,
        trained: bool = False,
    ) -> None:
        """
        Binds the compiled neural network and its execution context to the operational interface.

        :param torch_model: Compiled PyTorch master architecture module graph instance.
        :type torch_model: torch.nn.Module
        :param active_context: Reference back to the live hosting ActiveContextProxy session container.
        :type active_context: Any
        :param trained: Whether the model weights are ready for inference.
        :type trained: bool
        """
        self._architecture = torch_model
        self._context = active_context
        self._is_trained = trained
        
        logger.debug("Operational interface container initialized for local Autoencoder strategy.")

# --------------------------------------------------
# Section: Setters and getters 
# --------------------------------------------------

    @property
    def torch_object(self) -> torch.nn.Module:
        """
        Exposes direct access to the underlying raw PyTorch neural network module graph.

        :return: Raw master network instance.
        :rtype: torch.nn.Module
        """
        return self._architecture

    @property
    def is_trained(self) -> bool:
        """
        Returns the training status of the currently bound local model.

        :return: True if the model weights have been optimized.
        :rtype: bool
        """
        return self._is_trained

    @is_trained.setter
    def is_trained(self, status: bool) -> None:
        """
        Updates the training status deployment flag.

        :param status: Target training optimization flag.
        :type status: bool
        """
        self._is_trained = status


# --------------------------------------------------
# Section: Autoencoder functionality 
# --------------------------------------------------

    def encode(self, x: Any) -> np.ndarray:
        """
        Compresses input regular grid spectrometry intensity arrays into bottleneck latent space coordinates.

        :param x: Input matrix tensor profiles or numpy array. Shape: [Batch, Bins].
        :type x: Any
        :return: Extracted dense numeric latent space embeddings array.
        :rtype: np.ndarray
        """
        # Heading 1 (Latent Space Coordinate Compression Pass)
        self._ensure_ready()
        global_device = self._prepare_execution_environment()
        x_tensor = self._prepare_input(x)

        with torch.no_grad():
            z_embeddings = self._architecture.encoder(x_tensor.to(global_device))
            
        return z_embeddings.cpu().numpy()

    def decode(self, z: Any, grid_xs: bool = False) -> Any:
        """
        Decompresses dense bottleneck latent coordinates back into spatial grid layout reconstructions.

        :param z: Input latent space embeddings matrix or numpy array. Shape: [Batch, Latent_Dim].
        :type z: Any
        :param grid_xs: If True, skips inverse mapping and returns raw binned grid intensity configurations, defaults to False.
        :type grid_xs: bool
        :return: Aligned mass-to-charge values tracking pair tuple, or plain binned reconstruction arrays.
        :rtype: Any
        """
        # Heading 1 (Reconstructive Dimensional Decompressions Pass)
        self._ensure_ready(require_inverse=not grid_xs)
        global_device = self._prepare_execution_environment()
        z_tensor = self._prepare_input(z)

        with torch.no_grad():
            x_hat = self._architecture.decoder(z_tensor.to(global_device))

        x_hat_arr = x_hat.cpu().numpy()

        if grid_xs:
            return x_hat_arr

        inverse_binner = self._context.inverse_binner
        if x_hat_arr.ndim == 1:
            return inverse_binner(x_hat_arr)
        return [inverse_binner(row) for row in x_hat_arr]

    def transform(self, torch_loader_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Transforms the complete current context data image into a contiguous dense latent space block.

        :param torch_loader_config: Argument parameters mapping passed down into DataLoader initialization routines.
        :type torch_loader_config: Optional[Dict[str, Any]]
        :return: Concatenated matrix holding compressed footprints for the whole image layout.
        :rtype: np.ndarray
        """
        # Heading 1 (Continuous Image Spatial Scanning Pass)
        self._ensure_ready()
        self._prepare_execution_environment()
        
        active_dataset = getattr(self._context._wrapper, "active_dataset", None)
        if active_dataset is None:
            raise_validation_error(
                context_name="Autoencoder",
                message="No dataset is associated with the currently loaded model.",
            )

        batch_size = getattr(self._context._wrapper.models_manager, "batch_size", 256)
        loader_params = {
            "batch_size": batch_size,
            "pin_memory": True,
            "num_workers": 0,
            "shuffle": False
        }
        if torch_loader_config:
            loader_params.update(torch_loader_config)

        data_loader = DataLoader(active_dataset, **loader_params)
        embeddings_bucket = []

        logger.info("Initiating sequential image feature mapping over active data stream channels.")
        with torch.no_grad():
            for _, batch_tensor in data_loader:
                embeddings_bucket.append(self.encode(batch_tensor))

        logger.info("Sequential structural image data translation complete.")
        return np.concatenate(embeddings_bucket, axis=0)

    def compress_to_file(self, output_path: Union[str, Path]) -> None:
        """
        Transforms the active context imaging footprint into a compressed bin archive file along with layout metadata.

        :param output_path: Absolute or relative disk storage target destination path.
        :type output_path: Union[str, Path]
        """
        raise_validation_error(
            context_name="Autoencoder",
            message=(
                "NPZ latent export has been removed. Use the latent-context imzML "
                "writer introduced by the latent-space API."
            ),
        )


# --------------------------------------------------
# Section: Helpers
# --------------------------------------------------

    def _ensure_ready(self, require_inverse: bool = False) -> None:
        """
        Validates internal execution dependencies and training optimization status.

        :raises ValidationError: If dependencies are missing or weights are not ready.
        """
        if self._architecture is None:
            raise_validation_error(
                context_name="Autoencoder",
                message="The neural architecture is not assigned.",
            )
        
        if not self._is_trained:
            raise_validation_error(
                context_name="Autoencoder",
                message="The model has not been trained or loaded with trained weights.",
            )
            
        if require_inverse and getattr(self._context, "inverse_binner", None) is None:
            raise_validation_error(
                context_name="Autoencoder",
                message="Decoding to sparse m/z values requires an inverse binner.",
            )

    def _prepare_execution_environment(self) -> str:
        """
        Switches the underlying architecture state to evaluation mode and retrieves the target hardware device.

        :return: Hardware processing target device token string.
        :rtype: str
        """
        self._architecture.eval()
        return getattr(self._context._wrapper, "device", "cpu")

    def _prepare_input(self, x: Any) -> torch.Tensor:
        """
        Standardizes alternative input data types into standard aligned PyTorch float tensors.
        """
        if isinstance(x, torch.Tensor):
            tensor = x.float()
        else:
            tensor = torch.tensor(np.array(x), dtype=torch.float32)
        return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
