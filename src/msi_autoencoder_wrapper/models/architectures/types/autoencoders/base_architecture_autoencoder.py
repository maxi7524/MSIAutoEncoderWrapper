"""
Unified functional master graph coordinating complete multi-task MSI autoencoder data flows.
"""

from typing import Dict, Any
import torch
import torch.nn as nn

from ...architectures_manager import ArchitecturesManager
from ...base_architecture import MSIBaseMasterArchitecture
from .....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


@ArchitecturesManager.register_model_type("autoencoder")
class MSIBaseAutoencoderArchitecture(MSIBaseMasterArchitecture):
    """
    Symmetric architectural backbone coordinating data transformations across autoencoder blocks.
    """

    def __init__(self, resolved_components: Dict[str, nn.Module], **kwargs: Any) -> None:
        """
        Aggregates resolved sub-graphs into a singular multi-task processing autoencoder container.

        :param resolved_components: Map containing instantiated submodules resolved by the architecture manager.
        :type resolved_components: Dict[str, nn.Module]
        :param kwargs: Arbitrary parameter footprints preserved for downstream strategy instantiation.
        """
        # Parent initialization pass
        ## Execute base setup to initialize internal parameters storage ledgers
        super().__init__(resolved_components=resolved_components, **kwargs)
        
        # Sub-graph state registration loops
        ## Dynamically assign structural tracking fields from the resolved component database mapping
        self.encoder = resolved_components.get("encoder")
        self.decoder = resolved_components.get("decoder")
        self.projector = resolved_components.get("projector")
        
        ### Assign fallback empty structural container if multi-task heads are omitted from configuration blueprints
        heads_dict = resolved_components.get("heads", {})
        if not heads_dict and resolved_components.get("head") is not None:
            heads_dict = {"molecule": resolved_components["head"]}
        if isinstance(heads_dict, dict):
            self.heads = nn.ModuleDict(heads_dict)
        else:
            self.heads = nn.ModuleDict()
            
        logger.info("MSIBaseAutoencoderArchitecture master network graph successfully assembled.")

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Executes symmetric forward mapping operations over separate autoencoder architectural branches.

        :param x: Input spectral profiles aligned to regular master grid arrays. Shape: [Batch, Features].
        :type x: torch.Tensor
        :return: Standardized storage dictionary containing execution tensors mapping outputs.
        :rtype: Dict[str, torch.Tensor]
        """
        # Heading 1 (Mathematical Forward Graph Evaluation Trace)
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
        # Heading 1 (Gradient Control Flow Management)
        ## Iteratively switch requires_grad status across core encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
            
        ## Iteratively switch requires_grad status across core decoder parameters if instantiated
        if self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = not freeze
                
        logger.info("Autoencoder backbone baseline network blocks gradient status updated. Freezing set to: %s", freeze)
