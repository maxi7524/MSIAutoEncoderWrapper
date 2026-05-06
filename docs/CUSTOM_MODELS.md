# Creating Custom Models & Components

This guide explains how to extend `MSIAutoEncoderWrapper` by implementing custom architectures, loss functions (criterions), and binning strategies.

## General Concept

The library is built on **Abstract Base Classes (ABCs)**. To ensure that your custom component works seamlessly with the automated loading/saving system and the trainer, you must:
1. Inherit from the appropriate base class.
2. Implement all required abstract methods.
3. **Register** your class in the corresponding `__init__.py` file (or registry dictionary).

### The Registry System
Registration allows the `MSIAutoEncoder` manager to find and initialize your classes by name (e.g., from a config file). After defining a new class, add it to the registry in these locations:
* **Architectures**: `src/MSIAutoEncoderWrapper/architectures/__init__.py` -> `ARCHITECTURES_REGISTRY`
* **Criterions**: `src/MSIAutoEncoderWrapper/criterions/__init__.py` -> `CRITERIONS_REGISTRY`
* **Binners**: `src/MSIAutoEncoderWrapper/utils/Binners.py` -> `BINNER_REGISTRY`

### Automated Validation (Coming Soon)
We are developing a `pytest` suite to validate custom components. It will automatically check if your class:
* Correctly implements the interface.
* Handles device placement (CPU/GPU).
* Returns expected tensor shapes.

***

## Custom ABCs

### Custom Architectures
**Base Class**: `MSIBaseAutoencoderArchitecture`
**Location**: `src/MSIAutoEncoderWrapper/architectures/base.py`

Every model must function as an Autoencoder.

#### Requirements:
* **`encode(x)`**: Must return the latent representation $z$.
* **`decode(z)`**: Must reconstruct the spectrum from latent space.
* **`forward(x)`**: Usually returns a tuple `(latent, reconstruction)`.
* **`SetHyperparameters(...)`**: A static method that analyzes the `MSIDataset` to suggest optimal kernel sizes or layers before the model is initialized.

#### Example

```python
import torch
import torch.nn as nn
from .base import MSIBaseAutoencoderArchitecture

# We define separate nn.Module classes for the Encoder and Decoder 
# to allow PyTorch to efficiently optimize the gradient flow.

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, z):
        return self.layers(z)

class MyCustomModel(MSIBaseAutoencoderArchitecture):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # We assign modules as attributes of the main class to 
        # ensure all parameters are registered for the optimizer.
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Main flow using the modules defined above
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return z, x_reconstructed

    @staticmethod
    def SetHyperparameters(MSIDataset, latent_dim, user_hyperparameters=None, initialize_model=True):
        # Logic for automatic hyperparameter selection before initialization
        params = {"input_dim": MSIDataset.GetGridXAxisDepth(), "latent_dim": latent_dim}
        return MyCustomModel(**params) if initialize_model else params
```

***

### Custom Criterions
**Base Class**: `MSIABaseAutoEncoderCriterion`
**Location**: `src/MSIAutoEncoderWrapper/criterions/base.py`

The Criterion defines the training **logic**, you . Unlike standard PyTorch loss functions, it has access to the full context: the model, the dataloader, and spatial metadata.

#### Key Features:
* **Pre-training Setup**: Use the `REQUIRED_SETUP` list to define methods (e.g., `build_peak_bank`) that must run once before the training loop starts.
* **Spatial Awareness**: The `forward` method receives `batch_data` containing spectral indices. You can use `dataloader.dataset.img.GetSpectrumPosition(idx)` to incorporate spatial relationships into your loss.
* **Return Format**: Must return a tuple: `(total_loss_tensor, metrics_dict)`.


#### Example

Important: This is not just a loss function; it is a definition of how the cost 
should be calculated and the logic for comparing spectra (e.g., considering noise or neighborhood).

```python
import torch.nn.functional as F
from .base import MSIABaseAutoEncoderCriterion

class MyComparisonCriterion(MSIABaseAutoEncoderCriterion):

    # precalculation 
    self.REQUIRED_SETUP = [
            {
                'func': self.precompute_peak_bank, 
                'args': {'max_peaks_per_spectrum': max_peaks_per_spectrum}
            }
        ]

    def build_reference_bank(self, dataset):
        # Method called automatically before the training loop.
        # Can be used to pre-calculate features for the entire dataset.
        print("Pre-calculating reference bank...")

    def forward(self, batch_idx, batch_data, model, dataloader, device):
        indices, spectra = batch_data # spectra shape: [Batch, M/Z]
        
        # 1. Define how the model processes the data
        z, x_hat = model(spectra)
        
        # 2. Define how results are compared (e.g., MSE + custom logic)
        recon_loss = F.mse_loss(x_hat, spectra)
        
        # Spatial comparison logic can be added here 
        # by utilizing dataloader.dataset.img
        
        total_loss = recon_loss
        return total_loss, {"total_loss": total_loss.item(), "recon": recon_loss.item()}
```

***

### 3. Binners & Inverse Binners
**Location**: `src/MSIAutoEncoderWrapper/utils/Binners.py`

#### Binners
Binners project irregular raw MSI spectra onto a fixed m/z grid. This is crucial because CNNs require spatially consistent features (the same index must always mean the same m/z).
* **Requirement**: Must implement `__call__(xs, ys)` returning the binned intensities.
* **Tip**: Pay attention to the binning resolution—too coarse loses information, too fine creates sparse tensors which brings noise.



#### Inverse Binners
Used during reconstruction (e.g., exporting to `imzML`).
* **Requirement**: They must ensure the output m/z range matches the original image.
* **Consistency**: Ensure the `GetGridXAxis` results are consistent with the visualization tools to prevent coordinate shifting.

***

## Important Considerations
1. **Device Handling**: Always use the `device` parameter passed to the methods. Do not hardcode `.cuda()`.
2. **Memory**: MSI data is large. When pre-computing (like in `ContrastiveCriterion`), use `float16` or limit the number of samples to avoid OOM (Out Of Memory) errors.
3. **Registration Check**: If your model is not found during `load()`, check if the string name in the registry matches your class name exactly.