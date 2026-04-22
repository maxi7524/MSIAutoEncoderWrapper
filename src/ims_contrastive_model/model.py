"""
ims_contrastive_model/model.py
--------------------------------
Main class implementing the Contrastive MSI Segmentation Model.
Ties together the architecture, adapter, and training loop.
"""

# basic python
from pathlib import Path
from typing import Union, Optional
import json
import copy

# numerical libraries
import numpy as np
import pandas as pd

# torch
from sympy import hyperexpand
import torch
from torch.utils.data import DataLoader

# local modules
from .architecture import ContrastiveAutoencoder, ContrastiveLoss
from .optimization import suggest_cnn_configuration, train_loop_ims_contrastive_model
from .dataloader import IMSPyTorchDataset
from .utils import IMSModelVisualizer

# IMS library 
import m2aia as m2

# TODO - by default we assume that we provide parameters for image 

class IMSContrastiveModel(IMSModelVisualizer):
    def __init__(self, 
                # obligatory
                ## 
                IMSLoader: IMSPyTorchDataset, 
                latent_dim: int,
                # train
                ## parameters
                epochs: int = 10, 
                batch_size: int = 256, 
                lr: float = 1e-3,
                patience_limit: int = 5,
                ## hyperparameters
                hyperparameters = None # here put dict # TODO 
            
            ):
        # inherit
        super().__init__()
        # attributes
        ## model
        self._history = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## model configuratin
        self._IMSLoader = IMSLoader
        self._latent_dim = latent_dim
        self._hyperparameters = hyperparameters
        ## train attributes
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._patience_limit = patience_limit
        
        # dynamic 
        ## architecture construction
        # TODO - configure optimal params (optimization module call)  - moved to fit
        # self._hyperparameters = suggest_cnn_configuration(IMSLoader=IMSLoader, latent_dim=latent_dim, hyperparameters=hyperparameters)
        # self._model = ContrastiveAutoencoder(**self._hyperparameters).to(self._device)
        self._criterion = ContrastiveLoss(self._device)

    # ---------------------
    # model essentials 
    # ---------------------

    def fit(self, save_dir: str | Path):
        # create instance of model
        if self._model is None:
            print('[fit]: Initialization of new model ... ')
            self._hyperparameters = suggest_cnn_configuration(self.IMSLoader, self._latent_dim, self._hyperparameters)
            self._model = ContrastiveAutoencoder(**self._hyperparameters).to(self._device)
        else:
            print('[fit]: Continue training on loaded model ...')

        save_dir = Path(save_dir)
        train_loader = DataLoader(
            self.IMSLoader, 
            batch_size=self._batch_size, 
            # pin_memory=True,         # TODO test - nothing changed
            # shuffle=True
            )

        def save_callback(metrics, is_best):
            save_dir.mkdir(parents=True, exist_ok=True)
            self._history.append(metrics)
            # Newest state
            self.save(save_dir, filename="model_latest.pt")
            # If this is the best model we write it as our final
            if is_best:
                self.save(save_dir, filename="model_weights.pt")
            print('[save]: Model is saved')

        train_loop_ims_contrastive_model(
            model=self.model,
            dataloader=train_loader, 
            criterion=self._criterion,
            device=self._device, 
            epochs=self._epochs, 
            lr=self._lr, 
            patience_limit=self._patience_limit,
            save_callback=save_callback

        )

    
    def transform(self):
        # TODO encode full image to latent space and return img x latent space 
        # initialization
        ## change model execution
        self.model.eval()
        ## create loader
        loader = DataLoader(self.IMSLoader, batch_size=self._batch_size, shuffle=False)
        embeddings = []

        print("[Model] Transforming image to latent space...")
        with torch.no_grad(): # ram saving
            for batch in loader:
                z_norm, _ = self._model(batch.to(self._device))
                embeddings.append(z_norm.cpu().numpy())

            return np.concatenate(embeddings, axis=0)



    
    # ---------------------
    # helpers
    # ---------------------

    def save(self, path: str | Path = None, filename: str = "model_weights.pt"):
        """Saves model weights and training configuration."""
        # obtain paths
        if path is None:
            path = Path(str(self.IMSLoader.data_path) + '_model')
        else:
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # save
        ## model wages
        torch.save(self.model.state_dict(), path / filename)
        ## training history
        if self._history:
            df = pd.DataFrame(self._history)
            df.to_csv(path / "training_history.csv", index=False)
        ## save hyperparameters
        config = {
            "latent_dim": int(self._latent_dim),
            "mz_grid_size": len(self.IMSLoader.grid),
            "hyperparameters": copy.deepcopy(self._hyperparameters)
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)


    def load(self, path: str | Path):
        path = Path(path)
        # load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        # initialize model
        self._model = ContrastiveAutoencoder(**config['hyperparameters']).to(self._device)


        # load wages
        self._model.load_state_dict(torch.load(path / "model_weights.pt", map_location=self._device))
        # load history
        if (path / "training_history.csv").exists():
            self._history = pd.read_csv(path / "training_history.csv").to_dict('records')
        print(f"[Load]: Model loaded from {path}")

        # update attributes
        self._latent_dim = config.get("latent_dim", self._latent_dim)
        self._hyperparameters = config['hyperparameters']

        print(f"[Load]: Model loaded successfully from {path}")

    # ---------------------
    # getters and setters
    # ---------------------

    @property
    def IMSLoader(self):
        return self._IMSLoader

    @property
    def model(self):
        return self._model
    
    @property
    def history(self):
        return self._history
