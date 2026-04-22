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
    '''
    Quick explanation of loops 
    '''
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
            # TODO add here some visualization of training 
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
    '''
    Do not write here anything TO IMPLEMENT: 
    '''
    #TODO - suggest configuration
    # deterministic
    input_dim = len(IMSLoader.grid)

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
    # TODO create add here those noise with given spectra 
    noise = np.random.normal(1, 0.1, vec.shape)
    # Ensure tensor is on the same device
    noisy_vec = vec * torch.tensor(noise, dtype=torch.float32, device=vec.device)
    return noisy_vec