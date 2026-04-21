"""
models/msi_segmentation/architecture.py
---------------------------------------
PyTorch neural network definitions for Contrastive Autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReshapeLayer(nn.Module):
    def __init__(self, vec_shape):
        super(ReshapeLayer, self).__init__()
        self.vec_shape = vec_shape

    def forward(self, x):
        return x.view([x.shape[0]] + self.vec_shape)

class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels, kernels, strides):
        super().__init__()
        self.activation = nn.ReLU()
        self.net = nn.ModuleList( )
        self.dims = [input_dim]  # To store the dimensions at each layer

        current_dim = input_dim

        # Sequence of blocks: Convolutional Layer, Normalization Layer, ReLU
        for i in range(len(kernels)):
            kernel_size = kernels[i]
            hidden_dim = channels[i]
            next_hidden_dim = channels[i+1]
            stride = strides[i]
            
            self.net.append(nn.Conv1d(hidden_dim, next_hidden_dim, kernel_size, stride=stride))
            new_dim = (current_dim - kernel_size) // stride + 1
            self.net.append(nn.LayerNorm(new_dim))
            self.net.append(self.activation)
            current_dim = new_dim
            self.dims.append(current_dim)

        # Final output block: Linear Layer followed by Normalization Layer
        self.net.append(nn.Flatten())
        self.net.append(nn.Linear(current_dim * channels[-1], latent_dim))
        self.net.append(nn.LayerNorm(latent_dim))

        self.dims.append(latent_dim)

        self.decoder = self._get_decoder(self.dims, kernels, channels, strides)

    def _get_decoder(self, dims, kernel_sizes, hidden_dims, strides):
        activation = nn.ReLU()
        net = nn.ModuleList()

        # Initial block: Linear layer, Normalization, Transposed Convolution
        net.append(nn.Linear(dims[-1], dims[-2] * hidden_dims[-1]))
        net.append(nn.LayerNorm(dims[-2] * hidden_dims[-1]))
        net.append(ReshapeLayer([hidden_dims[-1], dims[-2]]))

        # TODO - sizes problem after implementation of last battch
        kernel_size = kernel_sizes[-1]
        stride = strides[-1]
        in_dim = dims[-2]
        target_dim = dims[-3]
        
        # Dynamically calculate rest
        out_pad = target_dim - ((in_dim - 1) * stride + kernel_size)
        net.append(nn.ConvTranspose1d(hidden_dims[-1], hidden_dims[-2], kernel_size, stride=stride, output_padding=out_pad))

        # Series of blocks: Normalization, ReLU activation, Transposed Convolution
        for i in range(len(kernel_sizes) - 1, 0, -1):
            current_dim = dims[i]
            net.append(nn.LayerNorm(current_dim))
            net.append(activation)

            kernel_size = kernel_sizes[i-1]
            stride = strides[i-1]
            in_dim = dims[i]
            target_dim = dims[i-1]

            # Dynamically calculate rest
            out_pad = target_dim - ((in_dim - 1) * stride + kernel_size)

            net.append(nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i-1], kernel_size, stride=stride, output_padding=out_pad))

        return net

    def forward(self, x):
        x = x.unsqueeze(1)
        for module in self.net:
            x = module(x)

        # Normalize embeddings to unit sphere
        x = F.normalize(x, p=2, dim=1)
        emb = x

        for module in self.decoder:
            x = module(x)
            
        x = x.squeeze(1)
        return emb, x


class ContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=2):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, emb_i, emb_j, encoder_inputs, decoder_outputs):
        # dynamically download current batch size and adjust size (last batch have have fewer pixels)
        actual_batch_size = emb_i.size(0)

        representations = torch.cat([emb_i, emb_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        # diagonals
        sim_ij = torch.diag(similarity_matrix, actual_batch_size)
        sim_ji = torch.diag(similarity_matrix, -actual_batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)

        # dynamically generated mask
        mask = (~torch.eye(actual_batch_size * 2, actual_batch_size * 2, dtype=torch.bool, device=self.device)).float()
        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        contrastive_loss = torch.sum(all_losses) / (2 * actual_batch_size)

        # Mean & Std Loss
        std_loss = torch.mean((torch.std(representations, dim=0) - 1) ** 2)
        mean_loss = torch.mean(torch.mean(representations, dim=1) ** 2)

        # MSE Loss
        decoder_loss = torch.mean(torch.sqrt(torch.mean((encoder_inputs - decoder_outputs) ** 2, dim=1)))

        # Combine the losses
        total_loss = contrastive_loss * 1e-2 + std_loss * 1e-3 + mean_loss * 1e-3 + decoder_loss

        # Zwracamy słownik z wartościami do logowania, zamiast modyfikować go w miejscu
        loss_dict = {
            'contrastive_loss': contrastive_loss.item(),
            'std_loss': std_loss.item(),
            'mean_loss': mean_loss.item(),
            'decoder_loss': decoder_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict