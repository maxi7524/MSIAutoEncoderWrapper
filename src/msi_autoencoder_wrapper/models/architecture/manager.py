from typing import Type, Dict, Any, Union, Optional
import torch.nn as nn

# Library imports
## logger
from ...utils.logger import get_custom_logger
## abstract classes
from .base_architecture import MSIBaseAutoencoderArchitecture
from .encoder_strategies.base_encoder import MSIBaseEncoder
from .decoder_strategies.base_decoder import MSIBaseDecoder
from .projectors_strategies.base_projector import MSIBaseProjector
from .heads_strategies.base_head import MSIBaseHead

# Synchronized logger initialization
logger = get_custom_logger(__name__)


class ArchitectureManager:
    """
    Central orchestration registry and runtime validation engine for MSI model topologies.

    This manager acts as an automated factory that dynamically resolves separate structural
    components from isolated registries. It strictly enforces type constraints to ensure
    mathematical compatibility across grid-x-axis input dimensions and latent-space interfaces.
    """

    # ---------------------------------------
    # Registry 
    # ---------------------------------------

    # Global isolated subcomponent registries mapping string tokens to class definitions
    _ENCODER_REGISTRY: Dict[str, Type[MSIBaseEncoder]] = {}
    _DECODER_REGISTRY: Dict[str, Type[MSIBaseDecoder]] = {}
    _PROJECTOR_REGISTRY: Dict[str, Type[MSIBaseProjector]] = {}
    _HEAD_REGISTRY: Dict[str, Type[MSIBaseHead]] = {}

    @classmethod
    def register_encoder(cls, name: str) -> Any:
        """
        Decorator factory to register a concrete MSIBaseEncoder implementation strategy.

        :param name: Unique lookup token identifier string for the encoder variant.
        :type name: str
        :return: Inner decorator closure wrapping the target class.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseEncoder]) -> Type[MSIBaseEncoder]:
            if not issubclass(subclass, MSIBaseEncoder):
                raise TypeError(f"Class '{subclass.__name__}' must inherit from MSIBaseEncoder.")
            cls._ENCODER_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def register_decoder(cls, name: str) -> Any:
        """
        Decorator factory to register a concrete MSIBaseDecoder implementation strategy.

        :param name: Unique lookup token identifier string for the decoder variant.
        :type name: str
        :return: Inner decorator closure wrapping the target class.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseDecoder]) -> Type[MSIBaseDecoder]:
            if not issubclass(subclass, MSIBaseDecoder):
                raise TypeError(f"Class '{subclass.__name__}' must inherit from MSIBaseDecoder.")
            cls._DECODER_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def register_projector(cls, name: str) -> Any:
        """
        Decorator factory to register a concrete MSIBaseProjector implementation strategy.

        :param name: Unique lookup token identifier string for the projector variant.
        :type name: str
        :return: Inner decorator closure wrapping the target class.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseProjector]) -> Type[MSIBaseProjector]:
            if not issubclass(subclass, MSIBaseProjector):
                raise TypeError(f"Class '{subclass.__name__}' must inherit from MSIBaseProjector.")
            cls._PROJECTOR_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def register_head(cls, name: str) -> Any:
        """
        Decorator factory to register a concrete MSIBaseHead implementation strategy.

        :param name: Unique lookup token identifier string for the auxiliary head variant.
        :type name: str
        :return: Inner decorator closure wrapping the target class.
        :rtype: Callable
        """
        def decorator(subclass: Type[MSIBaseHead]) -> Type[MSIBaseHead]:
            if not issubclass(subclass, MSIBaseHead):
                raise TypeError(f"Class '{subclass.__name__}' must inherit from MSIBaseHead.")
            cls._HEAD_REGISTRY[name] = subclass
            return subclass
        return decorator
    
    # ---------------------------------------
    # Main functionality 
    # ---------------------------------------

    @classmethod
    def build_architecture(
        cls,
        encoder_setup: Union[dict[str, Any], MSIBaseEncoder],
        decoder_setup: Optional[Union[dict[str, Any], MSIBaseDecoder]] = None,
        projector_setup: Optional[Union[dict[str, Any], MSIBaseProjector]] = None,
        heads_setup: Optional[dict[str, Union[dict[str, Any], MSIBaseHead]]] = None
    ) -> MSIBaseAutoencoderArchitecture:
        """
        Validates, instantiates, and compiles distinct subcomponents into a coherent model architecture.

        This factory method supports both object-based injection (pre-instantiated components) 
        and text-based dictionary configuration schemes (ideal for JSON schematic recovery).

        :param encoder_setup: Encoder configuration map or an explicit pre-built instance.
        :type encoder_setup: dict or MSIBaseEncoder
        :param decoder_setup: Decoder configuration map or an explicit pre-built instance, optional.
        :type decoder_setup: dict or MSIBaseDecoder, optional
        :param projector_setup: Contrastive projector head map or pre-built instance, optional.
        :type projector_setup: dict or MSIBaseProjector, optional
        :param heads_setup: Auxiliary downstream multi-task head definitions map, optional.
        :type heads_setup: dict, optional
        :return: Assembled and structural-type validated master autoencoder network graph.
        :rtype: MSIBaseAutoencoderArchitecture
        :raises TypeError: If any component instance violates contract type checks.
        :raises KeyError: If a requested registration key token is missing from the cache.
        """
        # 1. Resolve and validate the Encoder component node
        if isinstance(encoder_setup, dict):
            enc_type = encoder_setup["type"]
            if enc_type not in cls._ENCODER_REGISTRY:
                raise KeyError(f"Encoder token '{enc_type}' not found within registries.")
            encoder_instance = cls._ENCODER_REGISTRY[enc_type](**encoder_setup.get("params", {}))
        else:
            encoder_instance = encoder_setup

        if not isinstance(encoder_instance, MSIBaseEncoder):
            raise TypeError("Compiled Encoder component fails contract check against MSIBaseEncoder.")

        # 2. Conditional resolution and validation of the Decoder component node
        decoder_instance = None
        if decoder_setup is not None:
            if isinstance(decoder_setup, dict):
                dec_type = decoder_setup["type"]
                if dec_type not in cls._DECODER_REGISTRY:
                    raise KeyError(f"Decoder token '{dec_type}' not found within registries.")
                decoder_instance = cls._DECODER_REGISTRY[dec_type](**decoder_setup.get("params", {}))
            else:
                decoder_instance = decoder_setup

            if not isinstance(decoder_instance, MSIBaseDecoder):
                raise TypeError("Compiled Decoder component fails contract check against MSIBaseDecoder.")

        # 3. Conditional resolution and validation of the Projector component node
        projector_instance = None
        if projector_setup is not None:
            if isinstance(projector_setup, dict):
                proj_type = projector_setup["type"]
                if proj_type not in cls._PROJECTOR_REGISTRY:
                    raise KeyError(f"Projector token '{proj_type}' not found within registries.")
                projector_instance = cls._PROJECTOR_REGISTRY[proj_type](**projector_setup.get("params", {}))
            else:
                projector_instance = projector_setup

            if not isinstance(projector_instance, MSIBaseProjector):
                raise TypeError("Compiled Projector component fails contract check against MSIBaseProjector.")

        # 4. Conditional resolution and validation of auxiliary multi-task heads dictionary map
        resolved_heads: dict[str, MSIBaseHead] = {}
        if heads_setup is not None:
            for head_key, head_val in heads_setup.items():
                if isinstance(head_val, dict):
                    h_type = head_val["type"]
                    if h_type not in cls._HEAD_REGISTRY:
                        raise KeyError(f"Auxiliary Head token '{h_type}' not found within registries.")
                    head_instance = cls._HEAD_REGISTRY[h_type](**head_val.get("params", {}))
                else:
                    head_instance = head_val

                if not isinstance(head_instance, MSIBaseHead):
                    raise TypeError(f"Auxiliary Head '{head_key}' fails contract check against MSIBaseHead.")
                resolved_heads[head_key] = head_instance

        # 5. Compile and pack the integrated structural network container graph
        compiled_model = MSIBaseAutoencoderArchitecture(
            encoder=encoder_instance,
            decoder=decoder_instance,
            projector=projector_instance,
            heads=resolved_heads
        )

        # 6. Capture structural configuration metadata snapshots for future serialization pipelines
        compiled_model._config = {
            "encoder": encoder_setup if isinstance(encoder_setup, dict) else encoder_setup.GetConfig(),
            "decoder": decoder_setup if isinstance(decoder_setup, dict) or decoder_setup is None else decoder_setup.GetConfig(),
            "projector": projector_setup if isinstance(projector_setup, dict) or projector_setup is None else projector_setup.GetConfig(),
            "heads": {k: (v if isinstance(v, dict) else v.GetConfig()) for k, v in heads_setup.items()} if heads_setup is not None else None
        }

        logger.info("MSI Network pipeline components built, type-verified, and aggregated into master topology container.")
        return compiled_model