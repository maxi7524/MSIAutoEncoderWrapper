"""
ims_contrastive_model/optimization.py
---------------------------------------
Helper functions to dynamically adapt network architecture 
and execute the training loops.
"""

# python
import copy

# numerical
import numpy as np

# torch 
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# IMS 
import m2aia as m2
from .architecture import ContrastiveAutoencoder
from .dataloader import IMSPyTorchDataset

# ---------------------
# train loop
# ---------------------

def train_loop_ims_contrastive_model(
        model: ContrastiveAutoencoder, 
        dataloader: DataLoader, 
        criterion,  # it is forward from autoencoder 
        device, 
        epochs: int, 
        lr: float, 
        patience_limit: int,
        save_callback: callable # function
        ):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    t_max = epochs // 10 if epochs >= 10 else 1
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    best_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        # set epoch
        ## set train mode
        model.train()
        ## add row values 
        epoch_acc = {
            'contrastive_loss': 0.0, 'std_loss': 0.0, 'mean_loss': 0.0, 
            'decoder_loss': 0.0, 'total_loss': 0.0
        }

        # go through batches
        for batch in dataloader:
            ## batch = (batch, intensities ~ resampled)
            x1 = batch.to(device)
            x2 = apply_noise(x1)
            
            optimizer.zero_grad()
            z1, x1_hat = model(x1)
            z2, x2_hat = model(x2)
            
            loss, loss_dict = criterion(
                emb_i=z1, 
                emb_j=z2, 
                encoder_inputs=torch.cat([x1, x2]), 
                decoder_outputs=torch.cat([x1_hat, x2_hat])
            )
            
            loss.backward()
            optimizer.step()
            
            for k in epoch_acc.keys():
                epoch_acc[k] += loss_dict[k]
        
        # update model
        scheduler.step()

        ## epoch mean results
        avg_metrics = {k: v / len(dataloader) for k, v in epoch_acc.items()}
        avg_metrics['epoch'] = epoch + 1

        ## logging 
        current_loss = avg_metrics['total_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            patience = 0
            ### Run save callback for keeping iteration | overwrite best model
            save_callback(avg_metrics, is_best=True)
        else:
            patience += 1
            ### Run save callback for keeping iteration
            save_callback(avg_metrics, is_best=False)
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {current_loss:.4f} | Patience: {patience}/{patience_limit}")

        ## Early stopping 
        if patience > patience_limit:
            print(f"[Optimization] Early stopping triggered at epoch {epoch}.")
            break






# ---------------------
# helpers
# ---------------------

def suggest_cnn_configuration(IMSLoader: IMSPyTorchDataset, latent_dim: int, hyperparameters: dict):
    #TODO - suggest configuration
    # deterministic
    input_dim = IMSLoader.img.GetNumberOfSpectra()

    # predicted | given *if hyperparameters are not none apply hyperparameters 
    
    test_params = {
        # deterministic
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        # predicted:  
        'channels': [1, 2, 4, 16, 32, 64],
        'kernels': [7, 7, 5, 5, 5],
        'strides': [2, 3, 3, 3, 3]
    }

    return test_params 

def apply_noise(vec: torch.Tensor) -> torch.Tensor:
    """Adds Gaussian noise to the input tensor."""
    noise = np.random.normal(1, 0.1, vec.shape)
    # Ensure tensor is on the same device
    noisy_vec = vec * torch.tensor(noise, dtype=torch.float32, device=vec.device)
    return noisy_vec

# ---------------------
# OLD 
# ---------------------

# OLD CODE DEPRECATED - FOR PATTERN ONLY 
def old_suggest_cnn_configs(input_dim: int, possible_kernel_sizes: list = [3, 5, 11, 15], num_layers: int = 5) -> dict:
    """
    Finds optimal convolution configurations (kernels and strides) 
    """
    configurations = []

    def calc_stride(input_size, kernel_size):
        valid_strides = []
        for stride in range(2, 8):
            if (input_size - kernel_size) % stride == 0:
                output_size = (input_size - kernel_size) / stride + 1
                if output_size > 0:
                    valid_strides.append((stride, int(output_size)))
        return valid_strides

    def dfs(layer, current_size, current_kernels, current_strides):
        if layer == num_layers:
            configurations.append({
                "kernel_sizes": current_kernels[:],
                "strides": current_strides[:],
                "encoding_dim": current_size
            })
            return

        for k in possible_kernel_sizes:
            for stride, next_size in calc_stride(current_size, k):
                current_kernels.append(k)
                current_strides.append(stride)
                dfs(layer + 1, next_size, current_kernels, current_strides)
                current_kernels.pop()
                current_strides.pop()

    dfs(0, input_dim, [], [])
    
    if not configurations:
        raise ValueError(f"No valid CNN configuration found for input_dim={input_dim}")
        
    # Return the first valid configuration
    #  TODO - OPTIMIZATION FOR CONFIGURATIONS
    return configurations



def old_train_clr_loop(model, dataloader, criterion, device, epochs: int, lr: float, patience_limit: int, history_obj):
    """
    Executes the full Contrastive Learning training loop with early stopping.
    Updates the provided history_obj with epoch metrics.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    t_max = epochs // 10 if epochs >= 10 else 1
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
    
    best_loss = float('inf')
    patience = 0

    # OBSOLETE
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        if patience > patience_limit:
            print(f"[Optimization] Early stopping triggered at epoch {epoch}.")
            break
            
        model.train()
        epoch_losses_acc = {
            'contrastive_loss': 0.0, 'std_loss': 0.0, 'mean_loss': 0.0, 
            'decoder_loss': 0.0, 'total_loss': 0.0
        }
        num_batches = 0
        
        for batch_idx, spectra in enumerate(dataloader):
            # Spectra directly from your IMSPyTorchDataset adapter
            input_1 = spectra.to(device)
            input_2 = apply_noise(input_1)
            
            optimizer.zero_grad()
            
            emb_1, decoded_1 = model(input_1)
            emb_2, decoded_2 = model(input_2)
            
            loss, loss_dict = criterion(
                emb_i=emb_1, 
                emb_j=emb_2, 
                encoder_inputs=torch.cat((input_1, input_2)), 
                decoder_outputs=torch.cat((decoded_1, decoded_2))
            )
            
            loss.backward()
            optimizer.step()
            
            # Update accumulators
            for k, v in loss_dict.items():
                epoch_losses_acc[k] += v
            num_batches += 1
            
        scheduler.step()
        
        if num_batches == 0:
            print("[Optimization] Warning: No valid batches processed. Check dataset or batch size.")
            break
            
        # Logowanie do obiektu TrainingHistory (zgodne z API)
        epoch_avg_loss = epoch_losses_acc['total_loss'] / num_batches
        for k, v in epoch_losses_acc.items():
            history_obj.add_metric(k, v / num_batches)
            
        # Early stopping logic
        if best_loss > epoch_avg_loss:
            best_loss = epoch_avg_loss
            patience = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            
        print(f"Epoch [{epoch+1}/{epochs}] - Total Loss: {epoch_avg_loss:.4f} (Patience: {patience}/{patience_limit})")
        
    # Loading best model
    print(f"[Optimization] Restoring best model weights (Loss: {best_loss:.4f}).")
    model.load_state_dict(best_weights)

    return model