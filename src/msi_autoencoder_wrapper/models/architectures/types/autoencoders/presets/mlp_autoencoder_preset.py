"""Configurable deterministic MLP autoencoder preset for MSI spectra."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from ....architectures_manager import ArchitecturesManager
from ....utils.hidden_normalization import resolve_hidden_normalization
from ......utils.exceptions import raise_validation_error

if TYPE_CHECKING:
    from ......core.mixins.active_context.active_context_mixin import ActiveContextProxy


@ArchitecturesManager.register_preset("autoencoder", "MLPAutoencoder")
def get_mlp_autoencoder_preset(
    active_context: ActiveContextProxy,
    latent_dim: int,
    encoder_hidden_dims: Sequence[int] = (512,),
    decoder_hidden_dims: Sequence[int] | None = None,
    batch_normalization: bool | None = None,
    normalization: str | None = None,
    output_activation: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a deterministic fully connected autoencoder configuration.

    The dimension lists exclude the latent and output projections. Omitting
    ``decoder_hidden_dims`` creates a symmetric decoder by reversing the
    encoder dimensions; an explicit list permits asymmetric experiments.

    :param active_context: Active context providing the binned feature count.
    :type active_context: ActiveContextProxy
    :param latent_dim: Latent representation width.
    :type latent_dim: int
    :param encoder_hidden_dims: Ordered encoder hidden-layer widths.
    :type encoder_hidden_dims: Sequence[int]
    :param decoder_hidden_dims: Ordered decoder hidden-layer widths. By default,
        use the reversed encoder dimensions.
    :type decoder_hidden_dims: Sequence[int] | None
    :param batch_normalization: Deprecated compatibility flag. Use
        ``normalization`` in new configurations.
    :type batch_normalization: bool | None
    :param normalization: Hidden-feature normalization: ``layer`` (default),
        ``batch``, or ``none``.
    :type normalization: str | None
    :param output_activation: Decoder output activation; defaults to Softplus.
    :type output_activation: Mapping[str, Any] | None
    :return: Architecture-manager component configuration.
    :rtype: dict[str, Any]
    :raises ValidationError: If either dimension list is empty or invalid.
    """
    del kwargs
    resolved_encoder_dims = _validate_hidden_dims(
        "encoder_hidden_dims",
        encoder_hidden_dims,
    )
    resolved_decoder_dims = (
        list(reversed(resolved_encoder_dims))
        if decoder_hidden_dims is None
        else _validate_hidden_dims("decoder_hidden_dims", decoder_hidden_dims)
    )
    resolved_normalization = resolve_hidden_normalization(
        normalization,
        batch_normalization,
        context_name="MLPAutoencoderPreset",
    )

    input_dim = int(active_context.binner.GetXAxisDepth())
    activation = dict(
        output_activation
        or {"type": "softplus", "parameters": {}}
    )
    return {
        "encoder": {
            "strategy": "MLPEncoder",
            "params": {
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "hidden_dims": resolved_encoder_dims,
                "normalization": resolved_normalization,
            },
        },
        "decoder": {
            "strategy": "MLPDecoder",
            "params": {
                "latent_dim": latent_dim,
                "output_dim": input_dim,
                "hidden_dims": resolved_decoder_dims,
                "output_activation": activation,
                "normalization": resolved_normalization,
            },
        },
    }


def _validate_hidden_dims(name: str, dimensions: Sequence[int]) -> list[int]:
    """Validate and copy one preset hidden-dimension sequence."""
    if (
        isinstance(dimensions, (str, bytes))
        or not isinstance(dimensions, Sequence)
        or not dimensions
    ):
        raise_validation_error(
            "MLPAutoencoderPreset",
            f"{name} must be a non-empty sequence of positive integers.",
        )
    resolved = list(dimensions)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in resolved
    ):
        raise_validation_error(
            "MLPAutoencoderPreset",
            f"{name} must contain only positive integers.",
        )
    return resolved
