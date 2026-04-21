"""
models/msi_segmentation/model.py
--------------------------------
Main class implementing the Contrastive MSI Segmentation Model.
Ties together the architecture, adapter, and training loop.
"""

# basic python
from pathlib import Path
from typing import Union, Optional

# numerical libraries
import numpy as np

# torch
import torch
from torch.utils.data import DataLoader

# local modules
from .architecture import ContrastiveAutoencoder, ContrastiveLoss
from .optimization import suggest_cnn_configs, train_loop_ims_contrastive_model
from .dataloader import IMSPyTorchDataset

# IMS library 
import m2aia as m2

class ContrastiveSegmenter(BasePyTorchModel):
    def __init__(self, loader: IMSLoader = None, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3, patience: int = 5, **kwargs):
        
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, **kwargs)

        # redundant - left for readability 
        self.loader = loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        
        self.network = None
        self.history = None
        # TODO - would like to provide final size and some "complexity parameter" (maybe) so we obtain fr certain reduction task best possible networks.
        # Pass parameters to super() so they are stored in self.hyperparameters for saving/loading
        self.encoding_dim = None
        
        # TODO - add preload model feature

    def _build_network(self):
        """
        Reconstructs the PyTorch network architecture prior to loading weights.
        Uses the parameters already stored in self.hyperparameters.
        """
        self.network = ContrastiveAutoencoder(
            input_dim=self.hyperparameters.get('input_dim'),
            kernels=self.hyperparameters.get('kernel_sizes'),
            latent_dim=self.hyperparameters.get('encoding_dim'),
            channels=self.hyperparameters.get('hidden_dims'),
            strides=self.hyperparameters.get('strides')
        )
        self.to_optimal_device()

    def get_optimal_model_params(self):
        return suggest_cnn_configs(self.loader.shape[2])

    def fit(self , params: Optional[dict] = None, **kwargs) -> TrainingHistory:
        """Trains the contrastive encoder on the given IMSLoader data."""
        self.history = TrainingHistory()
        loader = self.loader

        # 1. Determine execution strategy
        strategy = kwargs.get('strategy', loader._execution_params['strategy'])
        
        # 2. Prepare PyTorch Dataset & DataLoader
        dataset = IMSPyTorchDataset(loader, strategy=strategy)
        # TODO We set drop_last=True to avoid error connected with with mask in ContrastiveLoss
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 3. Architecture Optimization
        input_dim = loader.shape[2] 
        if params:
            config = params
        else:
            print(f"[Model] Optimizing CNN for input dimension: {input_dim}")
            config = suggest_cnn_configs(input_dim)
            # TODO - implement optimal configs settings
            config = config[-1] 
        print(f"[Model] Selected configuration: {config}")

        # 4. Save hyperparameters to reconstruct network
        self.hyperparameters.update({
            'input_dim': input_dim,
            'encoding_dim': config['encoding_dim'],
            'kernel_sizes': config['kernel_sizes'],
            'hidden_dims': config['hidden_dims'],
            'strides': config['strides']
        })

        # 5. Initialize Network
        self._build_network()

        # 6. Initialize Loss Function
        criterion = ContrastiveLoss(device=self.device)

        # 7. Training loop  
        print("[Model] Starting training...")
        
        self.network = train_loop_ims_contrastive_model(
            model=self.network,
            dataloader=dataloader,
            criterion=criterion,
            device=self.device,
            epochs=self.epochs,
            lr=self.lr,
            patience_limit=self.patience,
            history_obj=self.history
        )
            
        self.is_fitted = True
        return self.history

    def transform(self, loader: IMSLoader, **kwargs) -> np.ndarray:
        """Encodes full IMS data into the compressed latent space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling transform().")
            
        strategy = kwargs.get('strategy', loader._execution_params['strategy'])
        dataset = IMSPyTorchDataset(loader, strategy=strategy)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.network.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for spectra in dataloader:
                spectra = spectra.to(self.device)
                # Zgodnie z nową architekturą, model zwraca (embedding, reconstruction)
                emb, _ = self.network(spectra)
                all_embeddings.append(emb.cpu().numpy())
                
        flat_embeddings = np.vstack(all_embeddings)
        
        h, w, _ = loader.shape
        volume = np.zeros((h, w, self.hyperparameters['encoding_dim']), dtype=np.float32)
        
        rows = dataset.spatial_coords[:, 0]
        cols = dataset.spatial_coords[:, 1]
        volume[rows, cols, :] = flat_embeddings
        
        return volume

    def export_latent_to_imzml(self, loader: Optional[IMSLoader] = None,output_path: Optional[str] = None):
        """
        Exports the latent space as an optimized imzML / ibd dataset.
        Allows loading the dimensionality reduction results back into a new IMSLoader.
        """
        if loader is None:
            if self.loader is None:
                raise ValueError("IMSLoader instance is required for export.")
            loader = self.loader

        if output_path is None:
            base_dir = Path(self.loader.experiment_dir) / self.loader.sample_dir
            output_path = base_dir / f"{self.loader.dataset_name}.segmented.imzML"
        else:
            output_path = Path(output_path)

        print(f"[Export] Generating latent space representation (dim={self.hyperparameters['encoding_dim']})...")
        volume = self.transform(loader) 
        
        valid_mask = self.loader._grid_map != -1
        rows, cols = np.where(valid_mask)

        # The m/z axis is replaced by latent feature indices
        mz_array = np.arange(self.hyperparameters['encoding_dim'], dtype=np.float64)

        print(f"[Export] Saving latent space data to: {output_path}...")
        
        with ImzMLWriter(str(output_path), polarity='positive') as writer:
            for r, c in zip(rows, cols):
                intensity_array = volume[r, c, :].astype(np.float32)
                # Note: imzML spatial coordinates are 1-indexed. Z coordinate is typically 1 for 2D MSI.
                writer.addSpectrum(mz_array, intensity_array, tuple((c + 1, r + 1, 1)))

        print(f"[Export] Successfully saved .imzML and .ibd to {output_path.parent}")