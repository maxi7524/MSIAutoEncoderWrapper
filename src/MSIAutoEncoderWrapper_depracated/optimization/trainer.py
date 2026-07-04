import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Callable

# import abstract classes
from ..architectures import MSIBaseAutoencoderArchitecture
from ..criterions import MSIABaseAutoEncoderCriterion

def train_model(
        model: MSIBaseAutoencoderArchitecture, 
        dataloader: torch.utils.data.DataLoader, 
        criterion: MSIABaseAutoEncoderCriterion, 
        device: torch.device, 
        epochs: int, 
        lr: float, 
        patience_limit: int,
        #TODO we could move it also here but we would need provide our mangaer class give +- 
        save_callback: Callable
    ):
    """
    General-purpose training engine for Ion Mobility Spectrometry (MSI) Autoencoders.

    This function orchestrates the training process, handling pre-computation tasks, 
    the optimization loop, and early stopping logic. It is designed to be 
    architecture-agnostic by delegating forward logic to the ``Criterion``.

    **Execution Workflow:**
    1. **Pre-computation**: Executes ``REQUIRED_SETUP`` tasks defined in the 
       Criterion (e.g., building the Noise Peak Bank) using the dataset.
    2. **Optimization**: Initializes the Adam optimizer with weight decay 
       and a Cosine Annealing learning rate scheduler[cite: 8].
    3. **Training Loop**:
        * Iterates through batches, providing the full ``model`` and ``dataloader`` 
          context to the Criterion's ``__call__`` method[cite: 8].
        * Performs backpropagation and updates weights based on the loss components 
          returned by the Criterion.
    4. **Monitoring**: Tracks metrics, logs progress (ETA, loss), and triggers 
       the ``save_callback`` for persistence and best-model tracking[cite: 8].

    :param model: The neural network architecture (must inherit from MSIBaseAutoencoder).
    :type model: MSIBaseAutoencoderArchitecture
    :param dataloader: PyTorch DataLoader providing batches from an MSIPyTorchDataset.
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: The loss function object containing forward logic and setup tasks.
    :type criterion: MSIABaseAutoEncoderCriterion
    :param device: Hardware target ('cuda' or 'cpu').
    :type device: torch.device
    :param epochs: Maximum number of training iterations.
    :type epochs: int
    :param lr: Initial learning rate.
    :type lr: float
    :param patience_limit: Number of epochs to wait for improvement before early stopping.
    :type patience_limit: int
    :param save_callback: Function called after each epoch to save weights and metrics.
    :type save_callback: Callable[[dict, bool], None]
    """
    # Execute REQUIRED_SETUP tasks defined in the Criterion
    if hasattr(criterion, 'REQUIRED_SETUP'):
        print("[Trainer] Executing REQUIRED_SETUP tasks...")
        for task in criterion.REQUIRED_SETUP:
            func = task['func']
            args = task.get('args', {})
            # Setup functions operate on the dataset (function precalculate object from original data)
            func(dataloader.dataset, **args)

    # Initialization of Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    t_max = epochs // 10 if epochs >= 10 else 1
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    best_loss = float('inf')
    patience = 0
    total_pixels = len(dataloader.dataset)

    for epoch in range(epochs):
        # set epoch
        ## set train mode
        model.train()
        ## add row values
        epoch_metrics = {} # Flexible dictionary to store various loss components
        
        # progress tracking for the current epoch
        processed_pixels = 0
        last_log_percent = 0
        start_time = time.time()

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # go through batches
        for i, batch_data in enumerate(dataloader):
            optimizer.zero_grad()
            
            ## Use Criterion to handle logic, augmentation, and model calls
            loss, loss_dict = criterion(
                batch_idx=i, 
                batch_data=batch_data, 
                model=model, 
                dataloader=dataloader, 
                device=device
            )
            
            loss.backward()
            optimizer.step()
            
            ## Aggregate metrics dynamically based on criterion output
            for k, v in loss_dict.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

            ## Update progress tracking
            processed_pixels += len(batch_data[1]) # batch_data is (indices, spectra)
            current_percent = (processed_pixels / total_pixels) * 100

            ## Log progress every ~5% or at the end
            if current_percent - last_log_percent >= 5 or processed_pixels == total_pixels:
                elapsed = time.time() - start_time
                ### Calculate remaining time based on current speed
                eta = (elapsed / processed_pixels) * (total_pixels - processed_pixels)
                print(f"[{current_percent:3.0f}%] {processed_pixels}/{total_pixels} | "
                      f"Loss: {loss.item():.4f} | ETA: {eta/60:.1f} min")
                last_log_percent = current_percent
        
        # update model
        scheduler.step()

        # Summary and Callbacks
        ## epoch mean results
        avg_metrics = {k: v / len(dataloader) for k, v in epoch_metrics.items()}
        avg_metrics['epoch'] = epoch + 1
        current_loss = avg_metrics['total_loss']

        ## logging 
        if current_loss < best_loss:
            best_loss = current_loss
            patience = 0
            ### Run save callback for keeping iteration | overwrite best model
            save_callback(avg_metrics, is_best=True)
        else:
            patience += 1
            ### Run save callback for keeping iteration
            save_callback(avg_metrics, is_best=False)
            
        print(f"Summary Epoch {epoch+1} | Mean Loss: {current_loss:.4f} | Patience: {patience}/{patience_limit}")

        ## Early stopping 
        if patience > patience_limit:
            print(f"[Optimization] Early stopping triggered at epoch {epoch+1}.")
            break

    return avg_metrics # Returns the last metrics for history logging