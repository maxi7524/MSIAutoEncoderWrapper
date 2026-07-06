import time
from typing import Callable, Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader

# Purely relative imports within the package framework
from ...utils.logger import get_custom_logger
from ...models.architecture.base_architecture import MSIBaseAutoencoderArchitecture
from ..criterions.criterions_manager import CompositeLoss

logger = get_custom_logger(__name__)


class MSIPyTorchTrainer:
    """
    High-performance engine executing execution loops for MSI network architectures.

    This trainer handles hardware tensor routing, execution metrics logging,
    pre-computation lifecycle hooks, and early stopping patience validation checkpoints.
    """

    def __init__(
        self,
        model: MSIBaseAutoencoderArchitecture,
        criterion: CompositeLoss,
        device: torch.device,
        patience_limit: int = 10
    ) -> None:
        """
        Constructs the training execution engine tracker.

        :param model: Active neural network backbone conforming to MSIBaseAutoencoderArchitecture.
        :type model: MSIBaseAutoencoderArchitecture
        :param criterion: Compiled weighted aggregate loss container module.
        :type criterion: CompositeLoss
        :param device: Allocation pointer forcing calculations on CPU or active CUDA hardware nodes.
        :type device: torch.device
        :param patience_limit: Iteration ceiling boundary allowed before triggering Early Stopping, defaults to 10.
        :type patience_limit: int
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.patience_limit = patience_limit

        self.best_loss = float("inf")
        self.patience_counter = 0

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        save_callback: Callable[[Dict[str, float], bool], None]
    ) -> list[dict[str, float]]:
        """
        Runs the full multi-epoch optimization pipeline.

        :param dataloader: Initialized PyTorch DataLoader serving structured matrix batches.
        :type dataloader: torch.utils.data.DataLoader
        :param epochs: Total integer boundary defining the max loop iterations space.
        :type epochs: int
        :param optimizer: Configured optimization algorithm managing backpropagation weight adjustments.
        :type optimizer: torch.optim.Optimizer
        :param scheduler: Active learning rate adjustment scheduler strategy.
        :type scheduler: Any
        :param save_callback: Executable closure checkpoint triggered to persist parameters on disk.
        :type save_callback: Callable
        :return: Training history tracker database log ledger.
        :rtype: list[dict[str, float]]
        """
        history_ledger: list[dict[str, float]] = []

        # 1. Execute required structural setup actions across loss functions hooks
        logger.info("Initiating global pre-computation sequences via criterion setups.")
        self.criterion.REQUIRED_SETUP(dataloader.dataset)

        total_pixels = len(dataloader.dataset)
        start_time = time.time()

        # 2. Primary epoch execution engine loop
        for epoch in range(epochs):
            self.model.train()
            epoch_metrics: Dict[str, float] = {}
            processed_pixels = 0
            last_log_percent = -1

            for batch_idx, batch_data in enumerate(dataloader):
                # Unpack and route input data array matrices to target execution hardware
                spatial_indices, spectra_tensors = batch_data
                spatial_indices = spatial_indices.to(self.device)
                spectra_tensors = spectra_tensors.to(self.device)
                device_batch = (spatial_indices, spectra_tensors)

                optimizer.zero_grad()

                # Dynamic forward-pass optimization block based on loss requirements
                ## Pass information configuration constraints down to prevent decoding calculations
                model_outputs = self.model.forward_optimized(
                    spectra_tensors,
                    requires_reconstruction=self.criterion.requires_reconstruction,
                    requires_projection=self.criterion.requires_projection
                )

                # Compute compound loss sum values and retrieve metric logs
                loss, batch_logs = self.criterion(model_outputs, device_batch)

                # Execute backpropagation graph updates
                loss.backward()
                optimizer.step()

                # Track metric accumulation variables across active batch runs
                processed_pixels += len(spatial_indices)
                for log_key, log_val in batch_logs.items():
                    epoch_metrics[log_key] = epoch_metrics.get(log_key, 0.0) + log_val

                # Metrics status updates printout tracking sequence
                current_percent = (processed_pixels / total_pixels) * 100
                if int(current_percent) % 10 == 0 and int(current_percent) != last_log_percent:
                    elapsed = time.time() - start_time
                    eta = (elapsed / max(1, processed_pixels)) * (total_pixels - processed_pixels)
                    print(
                        f"Epoch [{epoch+1:03d}/{epochs:03d}] | Progress: {current_percent:3.0f}% | "
                        f"Batch Loss: {loss.item():.4f} | ETA: {eta/60:.1f} min"
                    )
                    last_log_percent = int(current_percent)

            # Adjust optimizer rate profiles using selected scheduling strategy blueprints
            scheduler.step()

            # Compile epoch statistical calculation summary summaries maps
            avg_metrics = {k: v / len(dataloader) for k, v in epoch_metrics.items()}
            avg_metrics["epoch"] = float(epoch + 1)
            current_loss = avg_metrics["total_loss"]
            history_ledger.append(avg_metrics)

            # Early stopping and serialization validation gate
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
                save_callback(avg_metrics, True)  # Save checkpoint as the new best topology match
            else:
                self.patience_counter += 1
                save_callback(avg_metrics, False) # Save checkpoint as standard tracking step update

            print(
                f"Summary Epoch {epoch+1:03d} | Mean Loss: {current_loss:.4f} | "
                f"Patience Check: {self.patience_counter}/{self.patience_limit}"
            )

            if self.patience_counter >= self.patience_limit:
                logger.info(f"Early Stopping checkpoint triggered. Halting optimization process at epoch {epoch+1}.")
                break

        return history_ledger