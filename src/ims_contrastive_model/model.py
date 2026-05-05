"""
ims_contrastive_model/model.py
--------------------------------
Main class implementing the Contrastive MSI Segmentation Model.
Ties together the architecture, adapter, and training loop.
"""

# basic python
from pathlib import Path
import time
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
from .utils.Binners import IMSPyTorchBinner, IMSPyTorchInverseBinner, BINNER_REGISTRY
from .utils.plots import IMSModelVisualizer
from .utils import docs as DOCS 
from .utils.LatentSpace import build_latent_grid 
# IMS library 
import m2aia as m2
from pyimzml.ImzMLWriter import ImzMLWriter 

# TODO - by default we assume that we provide parameters for image 

class IMSContrastiveModel(IMSModelVisualizer):
    def __init__(self, 
                # obligatory
                ## path for model 
                path: Path | str,
                # train
                ## parameters
                epochs: int = 10, 
                batch_size: int = 64, 
                lr: float = 1e-3,
                patience_limit: int = 5,
                *,
                ## model configuration      
                latent_dim: int = None,
                hyperparameters = None, 
                ## data configuration
                IMSLoader: IMSPyTorchDataset = None,
                Binner:  IMSPyTorchBinner = None,
                InverseBinner: IMSPyTorchInverseBinner = None,
                #TODO add: criterion, model type 
            
            ):
        '''
        path -> folder model (saving / loading )
        '''
        # inherit
        super().__init__()
        # attributes
        ## model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model will be loaded on {self._device}")
        ## model configuratin
        self._path = Path(path)
        self._latent_dim = latent_dim
        self._hyperparameters = hyperparameters
        ## data configuration
        self._IMSLoader = IMSLoader
        self._Binner = Binner
        self._InverseBinner = InverseBinner
        ## train attributes
        self._history = []
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._patience_limit = patience_limit
        ## INFO: model is initialized in`fit` method 
        self._model = None
        self._criterion = ContrastiveLoss(self._device)

    # ---------------------
    # model essentials 
    # ---------------------

    # --- training ---

    def fit(self, save_dir: str | Path = None, continue_training = False):
        
        # handlers
        self._ensure_model_initialized()
        self._ensure_loader_available()
        self._ensure_binners_ready()

        if save_dir is None:
            save_dir = self.path
        else:
            save_dir = Path(save_dir)

        train_loader = DataLoader(
            self.IMSLoader, 
            batch_size=self._batch_size, 
            # # TODO SET PARAMS: here you can adjust parameters settings
            pin_memory= True,      
            num_workers = 2,   
            # shuffle=True
            )

        def save_callback(metrics, is_best):
            save_dir.mkdir(parents=True, exist_ok=True)
            self._history.append(metrics)
            # Newest state
            self.save(save_dir, filename="model_latest.pt")
            # If this is the best model we write it as our final
            if is_best:
                print('[fit]: New best model!')
                self.save(save_dir, filename="model_weights.pt")
            print('[fit]: Model is saved')

        # load latest model weights (not best) - if training was interrupted 
        if continue_training:
            self._model = torch.load(self.path / 'model_latest.pt')
            print('[fit]: Continue training on latest model ...')

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

    # --- autoencoder operations ---

    def encode(self, x: torch.Tensor):
        '''
        Methdods that encodes given batch_of_pixels x spectra 
        '''
        self._ensure_model_initialized()
        self._ensure_binners_ready()
        self._ensure_trained()
        self._ensure_loader_available()

        self.model.eval()
        x_tensor = self._prepare_input(x)
        with torch.no_grad():
            z_norm = self.model.encoder(x_tensor.to(self._device))
        return z_norm.cpu().numpy()

    def decode(self, z: torch.Tensor, grid_xs=False):
        '''
        Method that decode information from given transformed pixel 

        :params grid_xs if True returns decode value in grid-ys coordinates 
        '''
        # handlers
        self._ensure_model_initialized()
        self._ensure_binners_ready()
        self._ensure_trained()

        self.model.eval()
        z_tensor = self._prepare_input(z)
        with torch.no_grad():
            #  we use decoder 
            x_hat = self.model.decoder(z_tensor.to(self._device))

        # return ys value
        ## transform tensor to cpu
        x_hat = x_hat.cpu().numpy()
        ## return grid-ys value
        if grid_xs:
            return x_hat
        ## return ys value
        return self.InverseBinner(x_hat)
            

    def transform(self):
        '''
        Method which transforms whole image to latent space and return image_shape x latent_dim
        '''
        # handlers
        self._ensure_model_initialized()
        self._ensure_binners_ready()
        self._ensure_trained()
        self._ensure_loader_available()

        ## change model execution
        self.model.eval()
        ## create loader
        loader = DataLoader(
            self.IMSLoader, 
            batch_size=self._batch_size, 
            shuffle=False,
            # loader settings
            pin_memory=True,
            num_workers = 2
        )
        embeddings = []

        print("[Model] Encoding image to latent space...")
        with torch.no_grad(): # ram saving
            for batch in loader:
                embeddings.append(self.encode(batch))

            print("[Model] Done encoding image.")
            return np.concatenate(embeddings, axis=0)

    def get_latent_grid(self, embeddings=None, coordinates=None, compressed_path=None):

        # case: give path to existing image
        if compressed_path:
            print(f"[Model] Loading latent space from {compressed_path}...")
            data = np.load(compressed_path, allow_pickle=True)
            embeddings = data['embeddings']
            coordinates = data['coordinates']

        # case: not existing latent space 
        elif embeddings is None or coordinates is None:
            print("[Model] Transforming current image to latent space...")
            embeddings = self.transform() # Returns (N, C)
            
            # Get internal 0-indexed positions directly from m2aia
            I = self.IMSLoader.img
            coordinates = np.array([I.GetSpectrumPosition(i) for i in range(I.GetNumberOfSpectra())])

        return build_latent_grid(embeddings, coordinates)
        
    # --- compression & reconstruction ---

    def compress_to_file(self, output_path: str | Path):
        """
        Transforms the MSI image into latent space and saves it as a 
        compressed archive along with necessary metadata for future reconstruction.

        Args:
            output_path (str | Path): Path to save the compressed (.npz) file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # handlers
        self._ensure_model_initialized()
        self._ensure_loader_available()
        self._ensure_trained()

        # transforming image to latent space
        latent_embeddings = self.transform()

        # img metadata
        I = self.IMSLoader.img
        ## obtain metadata from m2aia
        metadata = I.GetMetaData()
        n_spectra = I.GetNumberOfSpectra()

        ## obtaining spectrum -> position configuration (we do not keep order of spectras)
        ### allocate memory for spectra
        coords = np.zeros((n_spectra, 3), dtype=np.int32)
        for i in range(n_spectra):
            p = I.GetSpectrumPosition(i)
            ### IMPORTANT:
            #### GetSpectrumPosition returns 0 indexed array WHILE imzML works on 1 indexed labels
            #### We save it in 0 index format to ensure similar indexing as in `GetSpectrumPosition` 
            #### In load case we need to add 1 to indexes to avoid errors
            coords[i] = [int(p[0]), int(p[1]), int(p[2])]

        # saving all data 
        np.savez_compressed(
            output_path,
            embeddings = latent_embeddings,
            metadata = json.dumps(metadata),
            coordinates = np.array(coords)
        )
        print(f"[Model] Compression complete. Saved to: {output_path}")


    def reconstruct_from_file(self, compressed_img_path: str | Path, output_imzml: str | Path):
        """
        Reconstructs an imzML/ibd file pair from a compressed latent space file.
        Uses the model's decoder and inverse binner to recover spectra.

        Args:
            compressed_path (str | Path): Path to the .npz file created by .compress().
            output_imzml (str | Path): Path to save the reconstructed .imzML file.
        """
        # handlers
        self._ensure_model_initialized()
        self._ensure_binners_ready()
        self._ensure_trained()

        # params
        compressed_img_path = Path(compressed_img_path)
        output_imzml = Path(output_imzml)

        print(f"[Model] Reconstructing image from {compressed_img_path}...")

        # load compressed data
        data = np.load(compressed_img_path, allow_pickle=True)
        embeddings = data['embeddings']
        #TODO - wymyślić jako można to dodać 
        metadata = json.loads(str(data['metadata']))
        coordinates = data['coordinates']


        # TODO - przetestować
        # 
        ## 
        latent_ds = latent_ds = torch.utils.data.TensorDataset(torch.from_numpy(embeddings).float())
        latent_loader = torch.utils.data.DataLoader(
            latent_ds, 
            batch_size=self._batch_size, 
            shuffle=False
        )

        print(f"[Model] Reconstructing {len(embeddings)} spectra using batch processing...")

        # create imsML
        ## initialization of m2aia
        with ImzMLWriter(
            str(output_imzml),
            intensity_dtype=np.float32,
            mz_dtype=np.float32
        ) as w:
            
            total_pixels = len(embeddings)
            processed_pixels = 0
            start_time = time.time()
            last_log_percent = 0

            self.model.eval()
            ## decoding 
            with torch.no_grad():
                for i, (z_batch,) in enumerate(latent_loader):
                    ### Decode entire batch on GPU and move to CPU
                    x_hat_batch = self.model.decoder(z_batch.to(self._device)).cpu().numpy()
                    
                    ### Slice coordinates corresponding to the current batch
                    start_idx = i * self._batch_size
                    batch_coords = coordinates[start_idx : start_idx + len(x_hat_batch)]

                    ### Zip decoded spectra with coordinates for iterative writing
                    for grid_ys, coords in zip(x_hat_batch, batch_coords):
                        #### Filter zeros and map to original m/z axis
                        xs, ys = self.InverseBinner(grid_ys)

                        #### Reshape coordinates 
                        #### IMPORTANT - 0 indexes brake img shape
                        x_c = int(coords[0]) + 1
                        y_c = int(coords[1]) + 1
                        z_c = (int(coords[2]) + 1) if len(coords) > 2 else 1
                        #### Write spectrum to file (requires 1D arrays and coordinate tuple)
                        w.addSpectrum(xs, ys, (x_c, y_c, z_c))

                    ### LOGS 

                    #### Update progress tracking
                    processed_pixels += len(x_hat_batch)
                    current_percent = (processed_pixels / total_pixels) * 100

                    #### Log progress every ~5% or at the end
                    if current_percent - last_log_percent >= 5 or processed_pixels == total_pixels:
                        elapsed_time = time.time() - start_time
                        # Calculate remaining time based on current speed
                        remaining_time = (elapsed_time / processed_pixels) * (total_pixels - processed_pixels)
                        
                        print(f"[{current_percent:3.0f}%] {processed_pixels}/{total_pixels} pixels processed | "
                              f"ETA: {remaining_time/60:.1f} min")
                        last_log_percent = current_percent

        print(f"[Model] Reconstruction complete. Output saved to: {output_imzml}")
        
    # ---------------------
    # save & load & initialization model 
    # ---------------------

    def define_model(self,
            ## model configuration      
            latent_dim: int,
            IMSLoader: IMSPyTorchDataset,
            InverseBinner: IMSPyTorchInverseBinner,
            ## data configuration
            hyperparameters = None, 
        ):
        # adding attributes needed to create configuration, 
        self._latent_dim = latent_dim
        self._IMSLoader = IMSLoader
        self._Binner = IMSLoader.Binner
        self._InverseBinner = InverseBinner
        self._hyperparameters = hyperparameters

        if self._model is None:
            print('[IMSContrastiveModel]: Initialization of new model ... ')
            self._hyperparameters = suggest_cnn_configuration(self.IMSLoader, self._latent_dim, self._hyperparameters)
            self._model = ContrastiveAutoencoder(**self._hyperparameters).to(self._device)
            #TODO - should write parameters we obtain
        
        else: 
            out_txt_warning = '''Model is already initialized. For safety reason you can not define it again.
            
            Create new instance of class.'''
            #TODO - should write parameters/architecture of current model 

            print(out_txt_warning)
            pass

    def save(self, path: str | Path = None, filename: str = "model_weights.pt"):
        """Saves model weights and training configuration."""
        # local path 
        if path is None:
            path = self.path
        else:
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # save
        ## model wages
        torch.save(self.model.state_dict(), path / filename)
        ## training history
        if self.history:
            df = pd.DataFrame(self.history)
            df.to_csv(path / "training_history.csv", index=False)
        ## save hyperparameters
        config = {
            # TODO - zrobić testy które sprawdzają łądowanie tego 
            "latent_dim": int(self._latent_dim),
            # added binner to allow reconstruction
            "binner_type": self.Binner.__class__.__name__,
            "binner_config": self.Binner.GetConfig(),
            "inverse_binner_type": self.InverseBinner.__class__.__name__,
            "inverse_binner_config": self.InverseBinner.GetConfig(), 
            "hyperparameters": copy.deepcopy(self._hyperparameters)
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)


    def load(self, best_model: bool = True):
        '''
        Loads trained model
        '''
        load_path = self.path

        # Load JSON configuration
        config_file = load_path / 'config.json'
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_file}")

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Binners 

        ## Helper function
        def reconstruct_binner(b_type, b_config):
            if b_type in BINNER_REGISTRY:
                # It loads binner from implemented binners in BINNER_REGISTRY
                return BINNER_REGISTRY[b_type](**b_config)
            print(f"[Load Warning]: Binner type '{b_type}' not in registry. Manual setup required.")
            return None
        
        ## Reconstruction
        ### binner 
        self._Binner = reconstruct_binner(config['binner_type'], config['binner_config'])
        ### inverse (need to add binner first)
        inv_config = config['inverse_binner_config']
        inv_config['Binner'] = self.Binner
        self._InverseBinner = reconstruct_binner(config['inverse_binner_type'], inv_config)

        # Model initialization 
    
        ## architecture
        self._latent_dim = config["latent_dim"]
        self._hyperparameters = config["hyperparameters"]
        self._model = ContrastiveAutoencoder(**self._hyperparameters).to(self._device)
        
        ## weights
        ### choose best model
        if best_model:
            weights_path = load_path / "model_weights.pt"
        ### choose recent model
        else:
            weights_path = load_path / "model_latest.pt"
        ### Load weights
        if weights_path.exists():
            self._model.load_state_dict(torch.load(weights_path, map_location=self._device))
            self._is_trained = True

        ## History
        history_path = load_path / "training_history.csv"
        if history_path.exists():
            self._history = pd.read_csv(history_path).to_dict('records')

        print(f"[Load]: Model loaded successfully from {load_path}")


    # ---------------------
    # helpers
    # ---------------------

    def _prepare_input(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Internal helper to ensure input is a float32 torch.Tensor on the correct device.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        # Ensure correct type and move to device
        return data.float().to(self._device)
    
    # --- Handlers: Not initialized model

    def _ensure_model_initialized(self):
        if self._model is None:
            raise RuntimeError(
                f"Model is not initialized. Call .define_model() or .load('{self.path}') first."
            )

    def _ensure_loader_available(self):
        if self._IMSLoader is None:
            raise ValueError(
                "IMSLoader is missing. You cannot perform operations requiring raw MSI data (like .fit() or .transform())."
            )

    def _ensure_trained(self):
        if not self._history:
            print("[Warning]: Operating on an untrained model. Results may be noise.")

    def _ensure_binners_ready(self):
        if self._Binner is None or self._InverseBinner is None:
            raise ValueError(
                "Binner or InverseBinner is not configured. Reconstruction is impossible."
            )


    # ---------------------
    # getters and setters
    # ---------------------

    
    
    @property
    def path(self):
        return self._path

    # #TODO - to nie powinno być edytowalne - to zależy jak obsługujemy całość 
    # @path.setter
    # def path(self, path: Path | str):
    #     self._path = Path(path)

    @property
    def IMSLoader(self):
        return self._IMSLoader

    @property
    def model(self):
        if self._model is None:
            print(f'''Model is not initialized.
                  
                  Initialize it using:
                  - `define_model` method which creates new model
                  - `load` method which loads preexisting model from {self.path} folder''')
            return None
        else:
            return copy.deepcopy(self._model)

    @property
    def Binner(self):
        return self._Binner

    @property
    def InverseBinner(self):
        return self._InverseBinner

    @InverseBinner.setter
    def InverseBinner(self, InverseBinner: IMSPyTorchInverseBinner):
        self._InverseBinner = InverseBinner
    
    @property
    def history(self):
        return self._history

IMSContrastiveModel.__doc__ = DOCS.IMSContrastiveModel_DOC
IMSContrastiveModel.__init__.__doc__ = DOCS.IMSContrastiveModel_init_DOC
IMSContrastiveModel.fit.__doc__ = DOCS.IMSContrastiveModel_fit_DOC
IMSContrastiveModel.transform.__doc__ = DOCS.IMSContrastiveModel_transform_DOC
IMSContrastiveModel.encode.__doc__ = DOCS.IMSContrastiveModel_encode_DOC
IMSContrastiveModel.decode.__doc__ = DOCS.IMSContrastiveModel_decode_DOC

