"""Configurable convolutional autoencoder preset for one-dimensional spectra."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from ....architectures_manager import ArchitecturesManager
from ......utils.exceptions import raise_validation_error

if TYPE_CHECKING:
    from ......core.mixins.active_context.active_context_mixin import ActiveContextProxy


@ArchitecturesManager.register_preset("autoencoder", "CNNAutoencoder")
def get_cnn_autoencoder_preset(
    active_context: ActiveContextProxy,
    latent_dim: int,
    channels: Sequence[int] = (1, 32, 16, 8),
    kernels: Sequence[int] = (10, 7, 5),
    strides: Sequence[int] = (3, 3, 3),
    output_normalization: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a symmetric Conv1D autoencoder without auxiliary heads.

    :param active_context: Active context providing the binned feature count.
    :type active_context: ActiveContextProxy
    :param latent_dim: Latent representation width.
    :type latent_dim: int
    :param channels: Channel widths including the input channel.
    :type channels: Sequence[int]
    :param kernels: Kernel width for each convolutional layer.
    :type kernels: Sequence[int]
    :param strides: Stride for each convolutional layer.
    :type strides: Sequence[int]
    :param output_normalization: Samplewise decoder-output normalization. New
        experiments should set this explicitly to match the dataset space.
    :type output_normalization: Mapping[str, Any] | None
    :return: Architecture-manager component configuration.
    :rtype: dict[str, Any]
    :raises ValidationError: If layer dimensions are inconsistent or non-positive.
    """
    del kwargs
    resolved_channels = _positive_dimensions("channels", channels)
    resolved_kernels = _positive_dimensions("kernels", kernels)
    resolved_strides = _positive_dimensions("strides", strides)
    layer_count = len(resolved_channels) - 1
    if (
        layer_count < 1
        or len(resolved_kernels) != layer_count
        or len(resolved_strides) != layer_count
    ):
        raise_validation_error(
            "CNNAutoencoderPreset",
            "channels must contain one more value than kernels and strides.",
        )
    if resolved_channels[0] != 1:
        raise_validation_error("CNNAutoencoderPreset", "channels must start with 1.")
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
        raise_validation_error("CNNAutoencoderPreset", "latent_dim must be positive.")

    input_dim = int(active_context.binner.GetXAxisDepth())
    spatial_dims = [input_dim]
    for kernel, stride in zip(resolved_kernels, resolved_strides):
        output_dim = ((spatial_dims[-1] - kernel) // stride) + 1
        if output_dim < 1:
            raise_validation_error(
                "CNNAutoencoderPreset",
                "Convolutional layers reduce the spectrum below one feature.",
            )
        spatial_dims.append(output_dim)

    common = {
        "latent_dim": latent_dim,
        "channels": resolved_channels,
        "kernels": resolved_kernels,
        "strides": resolved_strides,
        "spatial_dims": spatial_dims,
    }
    resolved_output_normalization = (
        {"type": "none", "parameters": {}}
        if output_normalization is None
        else dict(output_normalization)
    )
    return {
        "encoder": {
            "strategy": "CNNEncoder",
            "params": {"input_dim": input_dim, **common},
        },
        "decoder": {
            "strategy": "CNNDecoder",
            "params": {
                **common,
                "output_activation": {"type": "softplus", "parameters": {}},
                "output_normalization": resolved_output_normalization,
            },
        },
    }


def _positive_dimensions(name: str, dimensions: Sequence[int]) -> list[int]:
    """Validate one ordered sequence of positive dimensions."""
    if isinstance(dimensions, (str, bytes)) or not isinstance(dimensions, Sequence):
        raise_validation_error("CNNAutoencoderPreset", f"{name} must be a sequence.")
    resolved = list(dimensions)
    if not resolved or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in resolved
    ):
        raise_validation_error(
            "CNNAutoencoderPreset",
            f"{name} must contain positive integers.",
        )
    return resolved
