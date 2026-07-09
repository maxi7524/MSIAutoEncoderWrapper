"""
Unified functional master graph coordinating complete multi-task MSI data flows.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from ...utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


class MSIBaseArchitecture(nn.Module):
    """
    Unified architectural backbone coordinating data transformations across structural sub-modules.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        projector: Optional[nn.Module] = None,
        heads: Optional[Dict[str, nn.Module]] = None
    ) -> None:
        """
        Aggregates individual sub-graphs into a singular multi-task processing container.
        """
        super().__init__()
        
        # Core sub-graph state registration
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector
        self.heads = nn.ModuleDict(heads or {})
        self._config: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Executes parallelized forward mapping operations over separate architectural branches.

        :param x: Input spectral profiles aligned to regular master grid arrays. Shape: [Batch, Features].
        :type x: torch.Tensor
        :return: Standardized storage dictionary containing execution tensors mapping outputs.
        :rtype: Dict[str, torch.Tensor]
        """
        outputs: Dict[str, torch.Tensor] = {}

        # Forward execution sequence
        ## 1. Extract bottleneck latent-space representation tensor coordinates
        z = self.encoder(x)
        outputs["latent_space"] = z

        ## 2. Conditional spatial spectrum structural reconstruction pass
        if self.decoder is not None:
            outputs["reconstruction"] = self.decoder(z)

        ## 3. Conditional high-dimensional contrastive metric space projection pass
        if self.projector is not None:
            outputs["projection"] = self.projector(z)

        ## 4. Multi-task auxiliary downstream tasks head processing pass
        for head_name, head_module in self.heads.items():
            outputs[f"head_{head_name}"] = head_module(z)

        return outputs

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Locks parameter weights gradients across foundational encoder and decoder blocks.
        """
        # Tracking parameter locks loops
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
            
        if self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = not freeze
                
        logger.info("Backbone baseline processing networks gradient status updated. Freezing set to: %s", freeze)