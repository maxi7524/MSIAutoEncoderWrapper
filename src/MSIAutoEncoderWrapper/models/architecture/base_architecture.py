from typing import Optional, Any
import torch
import torch.nn as nn

# library imports
## logger
from ...utils.logger import get_custom_logger
## architecture imports
from .encoder_strategies.base_encoder import MSIBaseEncoder
from .decoder_strategies.base_decoder import MSIBaseDecoder
from .projectors_strategies.base_projector import MSIBaseProjector
from .heads_strategies.base_head import MSIBaseHead

logger = get_custom_logger(__name__)

class MSIBaseAutoencoderArchitecture(nn.Module):
    """
    Unified architectural backbone coordinating the complete MSI data flow.
    
    This class orchestrates structural transformations across independent Encoders, 
    Decoders, Projectors, and multi-task Heads. It outputs a standardized output 
    dictionary, removing rigid topology constraints and supporting variable loss computations.
    """

    def __init__(
        self,
        encoder: MSIBaseEncoder,
        decoder: Optional[MSIBaseDecoder] = None,
        projector: Optional[MSIBaseProjector] = None,
        heads: Optional[dict[str, MSIBaseHead]] = None
    ) -> None:
        """
        Aggregates modular network blocks into a single execution graph container.

        :param encoder: Subclass implementing MSIBaseEncoder to generate latent embeddings.
        :type encoder: MSIBaseEncoder
        :param decoder: Optional subclass implementing MSIBaseDecoder to execute spectrum reconstructions.
        :type decoder:編MSIBaseDecoder, optional
        :param projector: Optional subclass implementing MSIBaseProjector for contrastive alignment.
        :type projector: MSIBaseProjector, optional
        :param heads: Optional dictionary mapping task identifiers to concrete MSIBaseHead components.
        :type heads: dict[str, MSIBaseHead], optional
        """
        super().__init__()

        # Graph component injection mappings
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector
        
        # Deploy parallel multi-task heads as a secure PyTorch ModuleDict container
        self.heads = nn.ModuleDict(heads if heads is not None else {})

        # State property mapping for initialization caching
        self._config: dict[str, Any] = {}

    def GetConfig(self) -> dict[str, Any]:
        """
        Retrieves the complete architectural layout definitions for project serialization.

        :return: Blueprint configurations dictionary specifying all subcomponent maps.
        :rtype: dict
        """
        return self._config

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Direct access interface driving input vectors into the latent-space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Direct access interface reconstructing grid-x-axis profiles from the latent-space."""
        if self.decoder is None:
            raise RuntimeError("Spectral reconstruction requested but no Decoder component was initialized.")
        return self.decoder(z)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Direct access interface projecting latent points into contrastive fields."""
        if self.projector is None:
            raise RuntimeError("Contrastive projection requested but no Projector head was initialized.")
        return self.projector(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Executes a dynamic forward pass across all initialized execution endpoints.

        The execution output dictionary isolates downstream evaluation vectors, 
        safely bypassing uninitialized nodes (e.g., skipping reconstruction steps 
        if no decoder is defined).

        :param x: Spectral density input batch tensor matching grid-x-axis length profiles.
        :type x: torch.Tensor
        :return: Dictionary holding active tensor computations mapped by feature names:
                 - ``"latent_space"``: [Batch, Latent_Dim] (Always present)
                 - ``"reconstruction"``: [Batch, Grid_Depth] (Conditional)
                 - ``"projection"``: [Batch, Projection_Dim] (Conditional)
                 - ``"head_<name>"``: Task-specific evaluation outputs (Conditional)
        :rtype: dict[str, torch.Tensor]
        """
        outputs: dict[str, torch.Tensor] = {}

        # 1. Mandatory execution pass: compute structural latent-space embeddings
        z = self.encode(x)
        outputs["latent_space"] = z

        # 2. Conditional reconstruction pass: recreate spectrum footprints on grid-x-axis
        if self.decoder is not None:
            outputs["reconstruction"] = self.decode(z)

        # 3. Conditional representation pass: rzutowanie dla uczenia kontrastowego
        if self.projector is not None:
            outputs["projection"] = self.project(z)

        # 4. Auxiliary sub-graph execution loops
        for head_name, head_module in self.heads.items():
            outputs[f"head_{head_name}"] = head_module(z)

        return outputs

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Toggles structural parameter gradients to freeze the main feature extraction layers.

        :param freeze: Boolean flag activating layer parameter locking, defaults to True.
        :type freeze: bool, optional
        """
        # Layer status enforcement iteration loops
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        for param in self.decoder.parameters():
            param.requires_grad = not freeze
        
        logger.info(f"Model backbone feature layers status updated. Parameter freezing set to: {freeze}")