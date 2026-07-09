"""
Unified functional master graph coordinating complete multi-task MSI autoencoder data flows.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from ...architectures_manager import ArchitecturesManager
from .....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


@ArchitecturesManager.register_model_type("autoencoder")
class MSIBaseAutoencoderArchitecture(nn.Module):
    """
    Symmetric architectural backbone coordinating data transformations across autoencoder blocks.
    """

    def __init__(self, resolved_components: Dict[str, nn.Module], **kwargs: Any) -> None:
        """
        Aggregates resolved sub-graphs into a singular multi-task processing autoencoder container.

        :param resolved_components: Map containing instantiated submodules resolved by the architecture manager.
        :type resolved_components: Dict[str, torch.nn.Module]
        """
        super().__init__()
        
        # Sub-graph state registration loops
        ## Dynamically assign structural tracking fields from the resolved component database mapping
        self.encoder = resolved_components.get("encoder")
        self.decoder = resolved_components.get("decoder")
        self.projector = resolved_components.get("projector")
        
        ### Assign fallback empty structural container if multi-task heads are omitted from configuration blueprints
        self.heads = resolved_components.get("heads", nn.ModuleDict({}))
        self._config: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Executes parallelized forward mapping operations over separate autoencoder branches.

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
        if self.heads:
            for head_name, head_module in self.heads.items():
                outputs[f"head_{head_name}"] = head_module(z)

        return outputs

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Locks parameter weights gradients across foundational encoder and decoder blocks.

        :param freeze: Boolean flag activating layer parameter locking, defaults to True.
        :type freeze: bool
        """
        # Tracking parameter locks loops
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
            
        if self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = not freeze
                
        logger.info("Autoencoder backbone baseline network blocks gradient status updated. Freezing set to: %s", freeze)