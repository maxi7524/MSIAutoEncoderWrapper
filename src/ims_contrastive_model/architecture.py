"""
models/msi_segmentation/architecture.py
---------------------------------------
PyTorch neural network definitions for Contrastive Autoencoder.
"""

from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------
# Helpers
# ---------------------

class ReshapeLayer(nn.Module):
    """Layer to reshape tensors within nn.Sequential."""
    def __init__(self, vec_shape):
        super().__init__()
        self.vec_shape = vec_shape

    def forward(self, x):
        return x.view([x.shape[0]] + self.vec_shape)
    

# ---------------------
# Encoder & decoder
# ---------------------

class Encoder(nn.Module):
    """
    Sub-module responsible for compressing MSI spectra into latent space.

    It returns normalized value. 
    """
    def __init__(self, input_dim, latent_dim, channels, kernels, strides):
        super().__init__()
        self.layers = nn.ModuleList()
        # TODO maybe change name
        self.spatial_dims = [input_dim]

        # CNN
        ## input dimension for CNN
        current_dim = input_dim
        ## recurrent construction for CNN
        for i in range(len(kernels)):
            ### channels dim update
            current_dim = (current_dim - kernels[i]) // strides[i] + 1
            self.spatial_dims.append(current_dim)

            ### BLock:  Conv1d -> LayerNorm -> ReLU
            self.layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=channels[i], 
                    out_channels=channels[i+1], 
                    kernel_size=kernels[i], 
                    stride=strides[i]
                    ),
                nn.LayerNorm(current_dim),
                nn.ReLU()
            ))
            
        ## projection
        self.flatten = nn.Flatten()
        self.projector = nn.Sequential(
            nn.Linear(current_dim * channels[-1], latent_dim),
            nn.LayerNorm(latent_dim)
        )
        ## latent dim
        self.spatial_dims.append(latent_dim)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1) # x: (Batch, Features) -> (Batch, 1, Features)
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        z = self.projector(x)
        z_norm = F.normalize(z, p=2, dim=1)
        return z_norm
    
class Decoder(nn.Module):
    """
    Sub-module responsible for reconstructing MSI spectra from latent space.
    """
    def __init__(self, spatial_dims, latent_dim, channels, kernels, strides):
        super().__init__()
        self.layers = nn.ModuleList()

        # CNN
        ## Linear expansion back to spatial representation
        self.initial_expansion = nn.Sequential(
            nn.Linear(latent_dim, spatial_dims[-2] * channels[-1]),
            nn.LayerNorm(spatial_dims[-2] * channels[-1]),
            ReshapeLayer([channels[-1], spatial_dims[-2]])
        ) 
        ## Transposed convolutions
        for i in range(len(kernels) - 1, -1, -1):
            in_dim = spatial_dims[i+1]
            target_dim = spatial_dims[i]
            
            # Calculate output_padding to perfectly match original dimensions
            out_pad = target_dim - ((in_dim - 1) * strides[i] + kernels[i])
            
            block = nn.Sequential(
                nn.ConvTranspose1d(channels[i+1], channels[i], kernels[i], 
                                   stride=strides[i], output_padding=out_pad),
                # We apply Norm and Activation to all but the very last layer
                nn.LayerNorm(target_dim) if i > 0 else nn.Identity(),
                nn.ReLU() if i > 0 else nn.Identity()
            )
            self.layers.append(block)
        
    def forward(self, x):
        x = self.initial_expansion(x)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1)



# ---------------------
# Autoencoder
# ---------------------

class ContrastiveAutoencoder(nn.Module):
    """
    Main Autoencoder class, merges Encoder and Decoder together.
    """
    
    def __init__(self, input_dim, latent_dim, channels, kernels, strides):
        super().__init__()

        # Sub modules
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            channels=channels,
            kernels=kernels,
            strides=strides)
        
        self.decoder = Decoder(
            # here we use spatial from encoder
            spatial_dims=self.encoder.spatial_dims,
            latent_dim=latent_dim,
            channels=channels,
            kernels=kernels,
            strides=strides)
        
    def forward(self, x):
        # encode 
        z_norm = self.encoder(x)
        # decode
        x_hat = self.decoder(z_norm)
        return z_norm, x_hat
        
class ContrastiveLoss(nn.Module):
    """
    Detailed Breakdown of Regularization Losses:

    1. Variance Regularization (std_loss):
       Calculates the standard deviation of features across the batch dimension. 
       Loss = Mean((Std(representations, dim=0) - 1)**2).
       This ensures that each latent dimension remains active and carries unique information.

    2. Mean Regularization (mean_loss):
       Loss = Mean(Mean(representations, dim=1)**2).
       This prevents the latent space from drifting too far from the origin, ensuring 
       a zero-centered distribution which aids decoder stability.
    """
    def __init__(self, device, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, emb_i, emb_j, encoder_inputs, decoder_outputs):
        # dynamically download current batch size and adjust size (last batch have have fewer pixels)
        actual_batch_size = emb_i.size(0)

        # contrastive loss
        ## cosine similitaries (sim)
        representations = torch.cat([emb_i, emb_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        ## extract positive values
        sim_ij = torch.diag(similarity_matrix, actual_batch_size)
        sim_ji = torch.diag(similarity_matrix, -actual_batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)

        ## mask self similarity (diagonal)
        mask = (~torch.eye(actual_batch_size * 2, actual_batch_size * 2, dtype=torch.bool, device=self.device)).float()
        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        ## contrastive loss
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        contrastive_loss = torch.sum(all_losses) / (2 * actual_batch_size)

        ## regularization loss
        std_loss = torch.mean((torch.std(representations, dim=0) - 1) ** 2 + 1e-6)  # for 0 error
        mean_loss = torch.mean(torch.mean(representations, dim=1) ** 2)

        # MSE loss - reconstruction
        decoder_loss = torch.mean(torch.sqrt(torch.mean((encoder_inputs - decoder_outputs) ** 2, dim=1)))

        # total loss
        total_loss = contrastive_loss * 1e-2 + std_loss * 1e-3 + mean_loss * 1e-3 + decoder_loss

        # loss results
        loss_dict = {
            'contrastive_loss': contrastive_loss.item(),
            'std_loss': std_loss.item(),
            'mean_loss': mean_loss.item(),
            'decoder_loss': decoder_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict