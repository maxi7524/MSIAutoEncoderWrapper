"""
Preset configuration blueprint executing a dynamic, data-driven gradual dimension reduction sequence.
"""

from __future__ import annotations
import copy
import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ....architectures_manager import ArchitecturesManager
from ....utils.presets_utils import estimate_max_peak_width
from ......utils.logger import get_custom_logger
if TYPE_CHECKING:
    from ......core.mixins.active_context.active_context_mixin import ActiveContextProxy

# Logger initialization
logger = get_custom_logger(__name__)


@ArchitecturesManager.register_preset("autoencoder", "GradualReduction")
def get_gradual_reduction_preset(
    active_context: ActiveContextProxy, 
    latent_dim: int, 
    user_hyperparameters: Optional[Dict[str, Any]] = None, 
    projection_dim: int = 128,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Dynamically suggests compatible neural structures based on raw peak widths metrics.

    Estimates peak envelope widths to configure matching initial convolutional fields,
    iteratively scaling hidden depths until layers dimensions compress to fit bottlenecks.

    :param latent_dim: Core dimension sizing assigned to the target bottleneck space.
    :type latent_dim: int
    :param user_hyperparameters: Manual overrides configuration maps to bypass automated heuristics.
    :type user_hyperparameters: Optional[Dict[str, Any]]
    :param projection_dim: Sizing assigned to the contrastive metrics evaluation projector head.
    :type projection_dim: int
    :return: Completed nested components setup configuration blueprint map.
    :rtype: Dict[str, Any]
    """ 
    # Context data resolution
    ## Safely query mass axis feature layout boundaries from the provided dataset object
    input_dim = active_context.binner.GetXAxisDepth()
    logger.info("Preset builder initiating hyperparameter execution pipeline analysis.")

    # Override checkpoint pass
    if user_hyperparameters is not None:
        logger.info("User configuration footprint detected. Bypassing automated estimation routines.")
        params = copy.deepcopy(user_hyperparameters)
        params["input_dim"] = input_dim
        params["latent_dim"] = latent_dim
        return params

    # Peak envelope width statistical extraction
    auto_kernel_1 = estimate_max_peak_width(active_context, sample_size=10000)
    logger.debug("Automated peak configuration analysis selected baseline field kernel size: %s", auto_kernel_1)

    # Core structural block definitions
    channels = [1, 64, 32, 16, 8]
    kernels = [auto_kernel_1, 7, 5, 3]
    strides = [3, 4, 4, 3]

    # Dynamic iterative downsampling resolution loops
    current_stride_prod = int(np.prod(strides))
    current_out_dim = input_dim // current_stride_prod
    target_conv_out = max(latent_dim * 2, 512)

    # Dynamic layers growth expansion checking conditions
    while current_out_dim > target_conv_out and len(channels) < 6:
        new_stride = 2
        strides.append(new_stride)
        channels.append(max(channels[-1] // 2, 8))
        kernels.append(3)
        current_stride_prod *= new_stride
        current_out_dim = input_dim // current_stride_prod

    # Formulate synchronized spatial dimensions trace arrays to secure decoder transpositions symmetry
    spatial_dims: List[int] = [input_dim]
    running_dim = input_dim
    for i in range(len(kernels)):
        running_dim = ((running_dim - kernels[i]) // strides[i]) + 1
        spatial_dims.append(running_dim)

    logger.info("Gradual Reduction layout synthesis finalized. Mapped feature width roadmaps: %s", spatial_dims)

    # Structural component packing framework blueprints
    return {
        "encoder": {
            "strategy": "CNNEncoder",
            "params": {
                "input_dim": input_dim, "latent_dim": latent_dim, "channels": channels,
                "kernels": kernels, "strides": strides, "spatial_dims": spatial_dims
            }
        },
        "decoder": {
            "strategy": "CNNDecoder",
            "params": {
                "latent_dim": latent_dim, "channels": channels, "kernels": kernels,
                "strides": strides, "spatial_dims": spatial_dims,
                "output_activation": {"type": "softplus", "parameters": {}}
            }
        },
        "projector": {
            "strategy": "LinearProjector",
            "params": {
                "latent_dim": latent_dim, "projection_dim": projection_dim
            }
        }
    }
