from .base import IMSBaseAutoencoderArchitecture
from .ContrastiveAutoencoderSkrajny import ContrastiveAutoencoderSkrajny
from .ContrastiveAutoencoderMax_InverseDim import ContrastiveAutoencoderMax_InverseDim

# TODO list of import
ARCHITECTURES_REGISTRY = {
    "ContrastiveAutoencoderSkrajny": ContrastiveAutoencoderSkrajny,
    "ContrastiveAutoencoderMax_InverseDim": ContrastiveAutoencoderMax_InverseDim
    }