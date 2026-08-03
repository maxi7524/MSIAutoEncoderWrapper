"""Autoencoder implementation registration package."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

from ...architectures_manager import ArchitecturesManager
from .decoders.base_decoder import MSIBaseDecoder
from .encoders.base_encoder import MSIBaseEncoder
from .heads.base_head import MSIBaseHead
from .projectors.base_projector import MSIBaseProjector

ArchitecturesManager.register_component_category(
    "autoencoder", "encoder", MSIBaseEncoder
)
ArchitecturesManager.register_component_category(
    "autoencoder", "decoder", MSIBaseDecoder
)
ArchitecturesManager.register_component_category(
    "autoencoder", "projector", MSIBaseProjector
)
ArchitecturesManager.register_component_category(
    "autoencoder", "head", MSIBaseHead
)

discover_modules(__name__)
