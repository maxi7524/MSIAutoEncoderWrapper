from typing import Dict, Any, Union, Type
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR

# Purely relative imports within the package hierarchy
from ..utils.logger import get_custom_logger
from .criterions.criterions_manager import CriterionsManager, CompositeLoss
from .engine.base_trainer import MSIPyTorchTrainer
from ..models.architecture.base_architecture import MSIBaseAutoencoderArchitecture

# Synchronized telemetry logger initialization
logger = get_custom_logger(__name__)


class TrainingManager:
    """
    Central orchestration and runtime construction engine for MSI model training workflows.

    This manager interprets text-based configuration blueprints (such as JSON setups)
    to dynamically initialize loss functions, PyTorch optimizers, learning rate schedulers,
    and aggregate them into a fully operational MSIPyTorchTrainer instance.
    """

    # Local registry maps for standard PyTorch optimization components
    _OPTIMIZER_REGISTRY: Dict[str, Type[optim.Optimizer]] = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop
    }

    _SCHEDULER_REGISTRY: Dict[str, Any] = {
        "CosineAnnealingLR": CosineAnnealingLR,
        "StepLR": StepLR
    }

    @classmethod
    def build_optimizer(cls, name: str, model_parameters: Any, **kwargs: Any) -> optim.Optimizer:
        """
        Factory method to resolve and instantiate native PyTorch optimizers.

        :param name: Lookup string identifier for the optimizer algorithm (e.g., 'AdamW').
        :type name: str
        :param model_parameters: Iterable parameters graph extracted from the active nn.Module.
        :type model_parameters: Any
        :param kwargs: Arbitrary optimization parameters passed to the constructor (lr, weight_decay).
        :return: Configured instance of a PyTorch optimizer.
        :rtype: torch.optim.Optimizer
        :raises KeyError: If the requested optimizer token is absent from the registry.
        """
        if name not in cls._OPTIMIZER_REGISTRY:
            raise KeyError(f"Optimizer '{name}' not found. Available: {list(cls._OPTIMIZER_REGISTRY.keys())}")
        return cls._OPTIMIZER_REGISTRY[name](model_parameters, **kwargs)

    @classmethod
    def build_scheduler(cls, name: str, optimizer: optim.Optimizer, **kwargs: Any) -> Any:
        """
        Factory method to resolve and instantiate learning rate adjustment schedulers.

        :param name: Lookup string identifier for the scheduler variant (e.g., 'CosineAnnealingLR').
        :type name: str
        :param optimizer: Active initialized optimizer bound to the scheduler framework.
        :type optimizer: torch.optim.Optimizer
        :param kwargs: Strategy parameters passed directly to the scheduler configuration steps.
        :return: Initialized learning rate scheduler instance.
        :rtype: Any
        :raises KeyError: If the requested scheduler token is unknown.
        """
        if name not in cls._SCHEDULER_REGISTRY:
            raise KeyError(f"Scheduler '{name}' not found. Available: {list(cls._SCHEDULER_REGISTRY.keys())}")
        return cls._SCHEDULER_REGISTRY[name](optimizer, **kwargs)

    @classmethod
    def compile_trainer(
        cls,
        training_config: dict[str, Any],
        model: MSIBaseAutoencoderArchitecture,
        device: torch.device
    ) -> tuple[MSIPyTorchTrainer, optim.Optimizer, Any, CompositeLoss]:
        """
        Parses configuration definitions to assemble the integrated training execution engine.

        Example training_config layout:
        ------------------------------
        {
            "criterions": {
                "MSELoss": {"weight": 1.0, "params": {}},
                "InfoNCELoss": {"weight": 0.5, "params": {"temperature": 0.07}}
            },
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 0.001, "weight_decay": 1e-4}
            },
            "scheduler": {
                "type": "CosineAnnealingLR",
                "params": {"T_max": 100}
            },
            "patience": 10
        }

        :param training_config: Configuration maps detailing target components settings.
        :type training_config: dict
        :param model: Instantiated type-verified architecture conforming to MSIBaseAutoencoderArchitecture.
        :type model: MSIBaseAutoencoderArchitecture
        :param device: Hardware context allocation pointer (CPU or active CUDA core).
        :type device: torch.device
        :return: A tuple containing (trainer, optimizer, scheduler, composite_loss).
        :rtype: tuple
        """
        logger.info("Parsing training configuration to build runtime optimization pipeline.")

        # 1. Compile the combined mathematical loss functions using CriterionsManager
        criterion_setup = training_config["criterions"]
        composite_loss = CriterionsManager.build_composite_loss(criterion_setup)

        # 2. Extract active model weights parameters and build the target Optimizer
        opt_setup = training_config["optimizer"]
        optimizer = cls.build_optimizer(
            name=opt_setup["type"],
            model_parameters=model.parameters(),
            **opt_setup.get("params", {})
        )

        # 3. Bind the active Optimizer instance to build the target Scheduler pipeline step
        sched_setup = training_config["scheduler"]
        scheduler = cls.build_scheduler(
            name=sched_setup["type"],
            optimizer=optimizer,
            **sched_setup.get("params", {})
        )

        # 4. Initialize the core execution runtime trainer engine loop holder
        patience = training_config.get("patience", 10)
        trainer = MSIPyTorchTrainer(
            model=model,
            criterion=composite_loss,
            device=device,
            patience_limit=patience
        )

        logger.info("Training execution engine stack compiled and validated successfully.")
        return trainer, optimizer, scheduler, composite_loss