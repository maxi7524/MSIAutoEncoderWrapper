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
# important: updated depracated function
np.fromstring = np.frombuffer
import pandas as pd

# torch
import torch
from torch.utils.data import DataLoader


# local modules
## data
from .dataset import IMSPyTorchDataset
## model configuration
from .architectures import ARCHITECTURES_REGISTRY, IMSBaseAutoencoderArchitecture
from .criterions import CRITERIONS_REGISTRY,IMSABaseAutoEncoderCriterion
from .optimization.trainer import train_model
from .utils.Binners import IMSPyTorchBinner, IMSPyTorchInverseBinner, BINNER_REGISTRY
## helpers
from .utils.LatentSpace import build_latent_grid 
from .utils.plots import IMSModelVisualizer
from .utils import docs as DOCS 

# IMS libraries
## main 
import m2aia as m2
## writer
from pyimzml.ImzMLWriter import ImzMLWriter 

# TODO - by default we assume that we provide parameters for image 

class IMSAutoEncoder(IMSModelVisualizer):
    def __init__(self, 
                # obligatory
                ## path for model 
                path: Path | str,
                device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                # train
                ## parameters
                epochs: int = 10, 
                batch_size: int = 64, 
                lr: float = 1e-3,
                patience_limit: int = 5,
                *,
                # model configuration
                ## model type
                architecture: IMSBaseAutoencoderArchitecture = None,                
                criterion: IMSABaseAutoEncoderCriterion = None,
                ## model configuration      
                config: dict = None,

                # data configuration
                ## loader / dataset
                IMSDataset: IMSPyTorchDataset = None,
                ## binner 
                Binner:  IMSPyTorchBinner = None,
                InverseBinner: IMSPyTorchInverseBinner = None,
            
            ):
        '''
        path -> folder model (saving / loading )
        '''
        # inherit
        super().__init__()

        # attributes
        self._path = Path(path)
        self._device = device
        print(f"Model will be loaded on {self._device}")
        
        # model
        ## model configuratin
        self._architecture = architecture
        self._criterion = criterion

        ## data configuration
        self._IMSDataset = IMSDataset
        self._Binner = Binner
        self._InverseBinner = InverseBinner

        ## train attributes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience_limit = patience_limit

        ## configuration storage
        self._config = {
            "Architecture": {},
            "Criterion": {},
            "Binner": {},
            "InverseBinner": {}
        }
        ### update config
        if config:
            self._config.update(config)
        ### training history
        self._history = []

        
    # ---------------------
    # Model definition
    # ---------------------

    # ONE 

    # --- Configure model ---

    # TODO two concepts:
    # 1: as it is *loading classes via methods* 
    # 2: First init object then pass it 
    # TODO - trzeba zrobić tak żeby obie konwencje były obsługiwane 

    def SetArchitecture(self, ArchitectureClass: IMSBaseAutoencoderArchitecture, latent_dim: int, user_hyperparameters: dict = None):

        self._ensure_loader_available()

        # Handling existing architecture (not update - can broke model)
        if self._architecture is not None:
            out_txt_warning = '''Architecture is already initialized. For safety reason you can not define it again.'''
            print(out_txt_warning)
            return None

        hyperparameters = ArchitectureClass.SetHyperparameters(
            IMSDataset=self.IMSDataset, 
            latent_dim=latent_dim, 
            user_hyperparameters=user_hyperparameters, initialize_model=False)

        # update config
        self._config['Architecture'] = {
            "name": ArchitectureClass.__name__,
            ## For convenience (it is also in hyperparameters)
            "latent_dim": latent_dim,
            "hyperparameters": hyperparameters
        }

        self._architecture = ArchitectureClass(
            **hyperparameters
        ).to(self.device)

        # Logger
        print(f"[Manager] Architecture: {self._config['Architecture']['name']} initialized.")


    def SetCriterion(self, CriterionClass, crit_params):
        
        self._config["Criterion"] = {
            "name": CriterionClass.__name__,
            "params": crit_params
        }
        self._criterion = CriterionClass(**crit_params)
        self._criterion.device = self.device

        # Logger
        print(f"[Manager] Criterion: {self._config['Criterion']['name']} initialized.")

    # --- Configure loader ---

    # THOSE SHOULD BE InverseBinner.Setter, and SetInverseBinner, should working like: give class name and their params ???
    def SetInverseBinner(self, InverseBinner):
        self._InverseBinner = InverseBinner
        self._config["InverseBinner"] = {
            "name": self.InverseBinner.__class__.__name__,
            "params": self.InverseBinner.GetConfig()
        }



    def SetIMSDataset(self, IMSDataset: IMSPyTorchDataset):
        
        # Adding dataset and save Binner Option
        if not self._config['Binner']:
            self._IMSDataset = IMSDataset
            self._Binner = IMSDataset.Binner
            self._config['Binner'] = {
                "name": self.Binner.__class__.__name__,
                "params": self.Binner.GetConfig()
            }
            # Logger
            print(f"[Manager] Binner: {self._config['Binner']['name']} initialized.")

        else: 
            #TODO SHOULD CHECK DIMENSION - IF IMAGE CAN BE TRANSFERED *min_mz, max_mz should be same ...
            self._ensure_untrained()
            self._IMSDataset = IMSDataset

            print(f"[Manager] Binner already exists {self._config['Binner']['name']} initialized.")


    
    # ability to change criterion (it does not change model structure behaves) 
        
    # ---------------------
    # I/O operations 
    # ---------------------        

    def save(self, path: str | Path = None, filename: str = "model_weights.pt"):
        """Saves model weights and training configuration."""
        # Paths
        path = Path(path) if path else self.path
        path.mkdir(parents=True, exist_ok=True)

        # save
        ## model wages
        torch.save(self._architecture.state_dict(), path / filename)

        ## training history
        if self.history:
            df = pd.DataFrame(self.history)
            df.to_csv(path / "training_history.csv", index=False)

        ## save config
        with open(path / "config.json", "w") as f:
            json.dump(self._config, f, indent=4)

        # Logger 
        print(f"[Manager] Model and configuration saved to {path}")


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

        self._config = config
        print(f"[Manager] Loading and reconstructing model from {load_path}")

        # Reconstruction
        ## Binners
        ### Binner
        self._Binner = self._reconstruct_component(
            self._config.get("Binner"), 
            BINNER_REGISTRY
        )
        ### InverseBinner
        self._InverseBinner = self._reconstruct_component(
            self._config.get("InverseBinner"), BINNER_REGISTRY, 
            Binner=self._Binner
        )

        ## Architecture
        arch_cfg = self._config.get("Architecture")
        if arch_cfg:
            ArchClass = ARCHITECTURES_REGISTRY.get(arch_cfg["name"])
            if ArchClass:
                ### Instantiate architecture using saved hyperparameters
                self._architecture = ArchClass(**arch_cfg["hyperparameters"]).to(self._device)
                print(f"[OK] Architecture reconstructed: {arch_cfg['name']}")
            else:
                print(f"[FAILED] Architecture {arch_cfg['name']} not in registry.")

        ## Criterion
        crit_cfg = self._config.get("Criterion")
        if crit_cfg:
            CritClass = CRITERIONS_REGISTRY.get(crit_cfg["name"])
            if CritClass:
                ### Instantiate criterion using saved params
                self._criterion = CritClass(**crit_cfg["params"])
                print(f"[OK] Criterion reconstructed: {crit_cfg['name']}")
            else:
                print(f"[FAILED] Criterion {crit_cfg['name']} not in registry.")
                
        ## Weights
        ### choose best model
        if best_model:
            weights_name = "model_weights.pt"
        ### choose recent model
        else:
            weights_name = "model_latest.pt"
        weights_path = load_path / weights_name
        ### Load weights
        if weights_path.exists():
            self._architecture.load_state_dict(torch.load(weights_path, map_location=self._device))
            self._is_trained = True
            print(f"[OK] Weights loaded: {weights_name}")

        ## History
        history_path = load_path / "training_history.csv"
        if history_path.exists():
            self._history = pd.read_csv(history_path).to_dict('records')
            print(f"[OK] History loaded ({len(self._history)} epochs)")

        print("--- Model Load Complete ---\n")

    def _reconstruct_component(self, comp_config: dict, registry: dict, **extra_params):
        """
        Internal helper to instantiate a class from a registry based on 
        the provided configuration dictionary[cite: 21].
        """
        if not comp_config:
            return None
            
        name = comp_config.get("name")
        # Handles potential naming differences in config keys (params vs conf)
        params = comp_config.get("params", {}).copy()
        params.update(extra_params)
        
        comp_class = registry.get(name)
        if comp_class:
            instance = comp_class(**params)
            print(f"[OK] Component reconstructed: {name}")
            return instance
        
        print(f"[FAILED] Component {name} not found in registry.")
        return None
        


    # ---------------------
    # Training & Inference
    # ---------------------

    # --- training ---

    def fit(
            self, 
            save_dir: str | Path = None, 
            continue_training = False,
            train_model_config: dict = None,
            TorchLoader_config: dict = None, 
            ):
        
        # handlers
        self._ensure_model_initialized()
        self._ensure_loader_available()
        self._ensure_binners_ready()

        # outdir 
        save_dir = Path(save_dir) if save_dir else self._path
        
        # train_model
        ## Defaults settings
        train_model_params = {'epochs': 10, 'lr': 1e-3, 'patience_limit': 5}
        if train_model_config:
            train_model_params.update(train_model_config)

        # DataLoader
        ## Default settings
        TorchLoader_params = {
            'batch_size': self.batch_size, 
            'pin_memory': True,
            'num_workers': 2,
            'shuffle': True
            # itd.
        }
        ## Update settings
        if TorchLoader_config:
            TorchLoader_params.update(TorchLoader_config)
        ## Define loader
        TorchLoader = DataLoader(
            self.IMSDataset, 
            **TorchLoader_params   
            )

        ## Define helper function for saving data
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
            weights_path = self.path / 'model_latest.pt'
            self._architecture.load_state_dict(torch.load(weights_path, map_location=self._device))
            print('[fit]: Continue training on latest weights...')


        train_model(
            ## Model configuration
            model=self._architecture,
            dataloader=TorchLoader,
            criterion=self.criterion,
            device=self.device,
            ## train model settings
            save_callback=save_callback,
            **train_model_params
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

        self._architecture.eval()
        x_tensor = self._prepare_input(x)
        with torch.no_grad():
            z_norm = self._architecture.encoder(x_tensor.to(self._device))
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

        self._architecture.eval()
        z_tensor = self._prepare_input(z)
        with torch.no_grad():
            #  we use decoder 
            x_hat = self._architecture.decoder(z_tensor.to(self._device))

        # return ys value
        ## transform tensor to cpu
        x_hat = x_hat.cpu().numpy()
        ## return grid-ys value
        if grid_xs:
            return x_hat
        ## return ys value
        return self.InverseBinner(x_hat)
            

    def transform(self, TorchLoader_config: dict = None):
        '''
        Method which transforms whole image to latent space and return image_shape x latent_dim
        '''
        # handlers
        self._ensure_model_initialized()
        self._ensure_binners_ready()
        self._ensure_trained()
        self._ensure_loader_available()

        ## change model execution
        self._architecture.eval()
        ## create loader
        ### Default settings
        TorchLoader_params = {
            'batch_size': self.batch_size, 
            'pin_memory': True,
            'num_workers': 2,
            'shuffle': False
            # itd.
        }
        ### Update settings
        if TorchLoader_config:
            TorchLoader_params.update(TorchLoader_config)
        ### Define loader
        TorchLoader = DataLoader(
            self.IMSDataset, 
            **TorchLoader_params   
            )
        embeddings = []

        print("[Model] Encoding image to latent space...")
        with torch.no_grad(): # ram saving
            for spatial_idx, batch in TorchLoader:
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
            I = self.IMSDataset.img
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
        I = self.IMSDataset.img
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
            batch_size=self.batch_size, 
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

            self._architecture.eval()
            ## decoding 
            with torch.no_grad():
                for i, (z_batch,) in enumerate(latent_loader):
                    ### Decode entire batch on GPU and move to CPU
                    x_hat_batch = self._architecture.decoder(z_batch.to(self._device)).cpu().numpy()
                    
                    ### Slice coordinates corresponding to the current batch
                    start_idx = i * self.batch_size
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
    # Utilities
    # ---------------------

    # --- Helpers: ---

    def _prepare_input(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Internal helper to ensure input is a float32 torch.Tensor on the correct device.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        # Ensure correct type and move to device
        return data.float().to(self._device)
    
    # --- Handlers: Not initialized handlers --- 

    def _ensure_model_initialized(self):
        if self._architecture is None:
            raise RuntimeError(
                f"Model is not initialized. Call .define_model() or .load('{self.path}') first."
            )

    def _ensure_loader_available(self):
        if self._IMSDataset is None:
            raise ValueError(
                "IMSDataset is missing. You cannot perform operations requiring raw MSI data (like .fit() or .transform())."
            )

    def _ensure_trained(self):
        if not self._history:
            print("[Warning]: Operating on an untrained model. Results may be noise.")

    def _ensure_untrained(self):
        if self._history:
            print("[Warning]: Operating on an trained model. IMSDataset, must have set same binner.")

    def _ensure_binners_ready(self):
        if self._Binner is None or self._InverseBinner is None:
            raise ValueError(
                "Binner or InverseBinner is not configured. Reconstruction is impossible."
            )


    # ---------------------
    # getters and setters
    # ---------------------

    # --- obligatory ---

    @property
    def path(self):
        return self._path
    
    @property
    def device(self):
        return self._device

    # --- model type ---

    @property
    def architecture(self):
        return self._architecture
    
    @property
    def criterion(self):
        return self._criterion
    
     # --- model configuration ---
    
    @property
    def latent_dim(self):
        return self._latent_dim
    
    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def Model(self):
        #TODO trzeba to zmienc 
        if self._architecture is None:
            print(f'''Model is not initialized.
                  
                  Initialize it using:
                  - `define_model` method which creates new model
                  - `load` method which loads preexisting model from {self.path} folder''')
            return None
        else:
            return copy.deepcopy(self._architecture)

    # --- data configuration ---

    @property
    def IMSDataset(self):
        return self._IMSDataset
    
    @property
    def Binner(self):
        return self._Binner

    @property
    def InverseBinner(self):
        return self._InverseBinner
    
    @InverseBinner.setter
    def InverseBinner(self, InverseBinner: IMSPyTorchInverseBinner):
        self._InverseBinner = InverseBinner

    # --- train ---

    # TODO
    ## ADD BATCH SIZE HERE - it should calculate available memory and model size with all weights and print if there is risk of getting memory limit - 

    # --- other ---
    
    @property
    def history(self):
        return copy.deepcopy(self._history)

IMSAutoEncoder.__doc__ = DOCS.IMSContrastiveModel_DOC
IMSAutoEncoder.__init__.__doc__ = DOCS.IMSContrastiveModel_init_DOC
IMSAutoEncoder.fit.__doc__ = DOCS.IMSContrastiveModel_fit_DOC
IMSAutoEncoder.transform.__doc__ = DOCS.IMSContrastiveModel_transform_DOC
IMSAutoEncoder.encode.__doc__ = DOCS.IMSContrastiveModel_encode_DOC
IMSAutoEncoder.decode.__doc__ = DOCS.IMSContrastiveModel_decode_DOC

