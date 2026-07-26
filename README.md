
## MSIAutoEncoderWrapper

## Introduction

`MSIAutoEncoderWrapper` is a Python library for training and deploying Autoencoder architectures on **Mass Spectrometry Imaging (MSI)** data.

### Key Features
* **Seamless Data Integration**: Direct loading of `imzML` files via M²aia or pyimzML.
* **Standardized Pipeline**: Automatic handling of spectral binning, normalization, and PyTorch dataset creation.
* **Model Agnostic**: Easily plug in any PyTorch Autoencoder architecture.
* **Integrated Visualization**: Mixin classes for plotting training history, latent space maps, and reconstructions.
* **Automated I/O**: Portable JSON model configuration, model weights, and latent `imzML`/`ibd` representations.


### Research Context
This library reimplements and extends the work from:
> **"Contrastive Learning for Unsupervised Feature Extraction in Mass Spectrometry Imaging"** ([Anal. Chem. 2024, 96, 21, 8560–8571](https://pubs.acs.org/doi/10.1021/acs.analchem.4c06913)).

The original implementation was refactored to use `m2aia` for improved data handling and performance.



# Usage

## Installation

### 1. Environment Setup
We recommend using [Mamba](https://mamba.readthedocs.io/en/latest/) or [Conda](https://docs.conda.io/en/latest/) to manage dependencies. A pre-configured `environment.yml` file is provided.


```bash
# To use `m2aia` you need to install system libraries:
sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev libgomp1 libopenslide-dev
```

```bash
# Create the environment
conda env create -f scripts/environment/msi_env.yml

# Activate the environment
conda activate msi_env

# Install torch (~3 GB) (adding it to .yml drastically slows process)
# Select the CUDA toolkit version compatible with the installed NVIDIA driver.
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. Library Installation
Currently, the library is in development mode. Install it in editable mode:

```bash
# From the project root directory
pip install -e .
```

## Tutorials
The tutorials are ordered so that each notebook builds on the contracts introduced
by the previous one:

1. **[Workspace and model artifacts](assets/notebooks/tutorials/01_workspace_and_models.ipynb)**
   — create and customize a workspace, install the future example bundle, and
   load, save, and export portable models.
2. **[Readers, binners, and coordinates](assets/notebooks/tutorials/02_readers_binners_and_coordinates.ipynb)**
   — configure data components, compare M²aia and pyimzML backends, and select
   spectra and spatial slices in XY or matrix order.
3. **[Model configuration and training](assets/notebooks/tutorials/03_model_configuration_and_training.ipynb)**
   — use registries and presets, configure categorized criteria, estimate
   RAM/VRAM/disk requirements, and train the current single-image pipeline.
4. **[Autoencoder and latent space](assets/notebooks/tutorials/04_autoencoder_and_latent_space.ipynb)**
   — distinguish loaded and image-local models, encode/decode data, write latent
   imzML, slice latent images, and monitor memory.
5. **[Multi-image models — TODO](assets/notebooks/tutorials/05_multi_image_models_todo.ipynb)**
   — records the planned scope without presenting unimplemented APIs as usable.

## Creating Custom Models
Users can implement their own architectures and loss functions (criterions) by subclassing the base modules. For detailed instructions on how to integrate your own PyTorch models into the wrapper, please refer to:
* **[Development Guide: Custom Models](docs/CUSTOM_MODELS.md)**
* **[Training Criteria](docs/CRITERIONS.md)** — criterion categories, lifecycle
  hooks, configuration, and the Masserstein reconstruction objective.
* **[External Dataset Pipeline](docs/DATASET_PIPELINE.md)** — METASPACE
  discovery, catalog-only and download modes, annotation readers, and imzML
  merge provenance.

## Feedback & Support
If you have questions, suggestions, or find any bugs, please feel free to open an issue or [contact me directly by mail](mailto:mb.strozyk@student.uw.edu.pl).

<!-- Later mayebe :): contact the maintainers at [your-email@domain.com]. -->


## Bibliography
If you use this library in your research, please cite:
* **m2aia**: Cordes, J., et al. "M2aia-Interactive, Mobile, and Memory-Efficient Analysis of MSI Data." *Journal of Open Source Software*.
    * *Note: We highly encourage using (and citing) m2aia as it serves as the foundational engine for data handling in this project.*
* **Skrajny et al.**: [Link to original paper](https://pubs.acs.org/doi/10.1021/acs.analchem.4c06913) (Note: While m2aia was not explicitly cited in the original Skrajny paper due to implementation differences at that time, it is the core engine of this refactored library).
