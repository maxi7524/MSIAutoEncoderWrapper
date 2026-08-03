"""Architecture implementation registration package."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

from .architectures_manager import ArchitecturesManager
from .base_architecture import MSIBaseMasterArchitecture

discover_modules(__name__, excluded_parts={"schema"})

from .types.autoencoders.decoders.cnn_decoder import CNNDecoder
from .types.autoencoders.decoders.linear_spectrum_decoder import LinearSpectrumDecoder
from .types.autoencoders.encoders.cnn_encoder import CNNEncoder
from .types.autoencoders.encoders.variational_linear_encoder import VariationalLinearEncoder
from .types.autoencoders.heads.linear_classification_head import LinearClassificationHead
from .types.autoencoders.projectors.linear_projector import LinearProjector

__all__ = [
    "ArchitecturesManager",
    "CNNDecoder",
    "CNNEncoder",
    "LinearClassificationHead",
    "LinearProjector",
    "LinearSpectrumDecoder",
    "MSIBaseMasterArchitecture",
    "VariationalLinearEncoder",
]
