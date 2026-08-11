"""
Module providing runtime context attachments and operational execution proxies for autoencoder models.
"""

from pathlib import Path
from typing import Any, Optional, Dict, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from ....data import (
    BatchPreprocessor,
    RawDatasetView,
    RawSpectrumBatch,
    SharedAxisRawBatch,
    RawSpectrumCollator,
    LatentBatch,
    SpectrumBatch,
    SpectrumSpace,
)
from ....normalization import OutputSpace

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

    def encode(self, x: Any) -> np.ndarray | LatentBatch:
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
        source_batch = x if isinstance(x, SpectrumBatch) else None
        x_tensor = self._prepare_input(x.spectra if source_batch is not None else x)

        with torch.no_grad():
            z_embeddings = self._architecture.encoder(x_tensor.to(global_device))
            
        if source_batch is not None:
            return LatentBatch(
                sample_ids=source_batch.sample_ids,
                embeddings=z_embeddings,
                reconstruction_space=source_batch.space.as_reconstruction(),
                targets=source_batch.targets,
                metadata=source_batch.metadata,
                normalization_trace=source_batch.normalization_trace,
            )
        return z_embeddings.cpu().numpy()

    def decode(
        self,
        z: Any,
        grid_xs: bool = False,
        *,
        output_space: OutputSpace | None = None,
    ) -> Any:
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
        latent_batch = z if isinstance(z, LatentBatch) else None
        z_tensor = self._prepare_input(z.embeddings if latent_batch is not None else z)

        with torch.no_grad():
            x_hat = self._architecture.decoder(z_tensor.to(global_device))

        normalization = self._context.normalization
        trace = latent_batch.normalization_trace if latent_batch is not None else None
        resolved_space = self._resolve_output_space(output_space, normalization)
        if resolved_space == "source" and normalization is not None and trace is None:
            raise_validation_error(
                "Autoencoder",
                "Source-space decoding requires a LatentBatch carrying normalization state.",
            )
        if (
            resolved_space == "source"
            and normalization is not None
            and normalization.reconstruction.denormalization_stage == "after_decode"
        ):
            x_hat = normalization.inverse(x_hat, trace)

        x_hat_arr = x_hat.cpu().numpy()

        if grid_xs:
            if resolved_space == "source" and normalization is not None and normalization.reconstruction.denormalization_stage == "after_inverse_binning":
                return normalization.inverse(x_hat, trace).cpu().numpy()
            return x_hat_arr

        inverse_binner = self._context.inverse_binner
        dense_values = x_hat if x_hat.ndim == 2 else x_hat.unsqueeze(0)  # (B, F)
        dense_batch = SpectrumBatch(
            sample_ids=torch.arange(dense_values.shape[0], device=dense_values.device),
            spectra=dense_values,
            space=SpectrumSpace(
                mass_axis=torch.as_tensor(
                    self._context.binner.GetXAxis(),
                    device=dense_values.device,
                    dtype=dense_values.dtype,
                )
            ),
            normalization_trace=trace,
        )
        inverse_result = inverse_binner(dense_batch)
        inverse_rows = []
        for index in range(dense_values.shape[0]):
            start = int(inverse_result.offsets[index])
            end = int(inverse_result.offsets[index + 1])
            inverse_rows.append((
                inverse_result.mass_values[start:end].cpu().numpy(),
                inverse_result.intensities[start:end].cpu().numpy(),
            ))
        if resolved_space == "source" and normalization is not None and normalization.reconstruction.denormalization_stage == "after_inverse_binning":
            restored_rows = []
            for index, (mass_axis, values) in enumerate(inverse_rows):
                state = type(trace)(
                    pipeline_name=trace.pipeline_name,
                    stage=trace.stage,
                    states=tuple(
                        {name: tensor[index : index + 1] for name, tensor in item.items()}
                        for item in trace.states
                    ),
                    capabilities=trace.capabilities,
                )
                restored = normalization.inverse(
                    torch.as_tensor(values, device=x_hat.device, dtype=x_hat.dtype).unsqueeze(0),
                    state,
                ).squeeze(0)
                restored_rows.append((mass_axis, restored.cpu().numpy()))
            inverse_rows = restored_rows
        return inverse_rows[0] if x_hat_arr.ndim == 1 else inverse_rows

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
        compute_device = torch.device(
            loader_params.pop(
                "compute_device", getattr(self._context._wrapper, "device", "cpu")
            )
        )
        preprocessing_device = torch.device(
            loader_params.pop("preprocessing_device", compute_device)
        )
        loader_dataset = active_dataset
        preprocessor = None
        if callable(getattr(active_dataset, "get_raw_item", None)):
            schemas_getter = getattr(active_dataset, "get_target_schemas", None)
            schemas = schemas_getter() if callable(schemas_getter) else {}
            loader_params["collate_fn"] = RawSpectrumCollator(schemas)
            loader_params["pin_memory"] = preprocessing_device.type == "cuda"
            loader_dataset = RawDatasetView(active_dataset)
            preprocessor = BatchPreprocessor(
                active_dataset,
                preprocessing_device,
                compute_device,
            )
        data_loader = DataLoader(loader_dataset, **loader_params)
        embeddings_bucket = []
        self._architecture.to(compute_device)
        self._architecture.eval()

        logger.info("Initiating sequential image feature mapping over active data stream channels.")
        with torch.inference_mode():
            # REMARK: We take batch and always second index ([1]) - it solves problem with possible third value
            for batch in data_loader:
                if isinstance(batch, (RawSpectrumBatch, SharedAxisRawBatch)):
                    batch = preprocessor(batch)
                spectra = batch[1].to(
                    compute_device,
                    non_blocking=compute_device.type == "cuda",
                )
                embeddings_bucket.append(
                    self._architecture.encoder(spectra).detach().cpu().numpy()
                )

        logger.info("Sequential structural image data translation complete.")
        return np.concatenate(embeddings_bucket, axis=0)

    def compress_to_file(self, output_path: Union[str, Path]) -> Path:
        """
        Save the active image transform as a latent imzML/ibd pair.

        :param output_path: Absolute or relative disk storage target destination path.
        :type output_path: Union[str, Path]
        :return: Written latent imzML path.
        :rtype: pathlib.Path
        """
        return self._context.save_latent(output_path=output_path)


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

    @staticmethod
    def _resolve_output_space(output_space: OutputSpace | None, normalization: Any) -> OutputSpace:
        """Resolve explicit, active-policy, and stage-derived output behavior."""
        if output_space is not None:
            return output_space
        if normalization is None:
            return "source"
        configured = normalization.reconstruction.output_space
        if configured is not None:
            return configured
        return "normalized" if normalization.stage == "raw" else "source"
