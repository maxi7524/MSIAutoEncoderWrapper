
## IMSAutoEncoderWrapper

## Introduction

`IMSAutoEncoderWrapper` is a Python library for training and deploying Autoencoder architectures on **Mass Spectrometry Imaging (MSI)** data.

### Key Features
* **Seamless Data Integration**: Direct loading of `imzML` files via `m2aia`.
* **Standardized Pipeline**: Automatic handling of spectral binning, normalization, and PyTorch dataset creation.
* **Model Agnostic**: Easily plug in any PyTorch Autoencoder architecture.
* **Integrated Visualization**: Mixin classes for plotting training history, latent space maps, and reconstructions.
* **Automated I/O**: Simplified saving and loading of model weights and compressed latent representations.

<!-- #TODO - zastanawiam się czy to dawać
## Current Project Status
* Developing **Contrastive Learning** models for feature extraction.
* Optimizing Autoencoders for **lossy compression** of MSI datasets.
* Building predictive models (classification/segmentation) based on the latent space. -->

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
conda env create -f scripts/environment/ims_env.yml

# Activate the environment
conda activate ims_env

# Install torch (~3 GB) (adding it to .yml drastically slows process)
# TODO - adjust `pytorch-cuda`  
micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. Library Installation
Currently, the library is in development mode. Install it in editable mode:

```bash
# From the project root directory
pip install -e .
```

## Tutorials
Detailed guides can be found in the `notebooks/tutorials` directory:

1.  **[Tutorial 1: Compression](notebooks/tutorials/tutorial1_compression.ipynb)**
    * Initialize and train an `IMSContrastiveModel`.
    * Analyze training losses and visualize reconstruction quality.
    * Compress a full `imzML` image into a latent space saved in `npz` format.
    * Demonstration of loading the latent space independently of the raw data.

2.  **[Tutorial 2: Reconstruction & Visualization](notebooks/tutorials/tutorial2_decompression.ipynb)**
    * Load a pre-trained model and a compressed latent space file.
    * Reconstruct the latent space back into the original spectral domain.
    * Export results back to `imzML` format.
    * Visualization of latent components.

## Creating Custom Models
Users can implement their own architectures and loss functions (criterions) by subclassing the base modules. For detailed instructions on how to integrate your own PyTorch models into the wrapper, please refer to:
* **[Development Guide: Custom Models](docs/CUSTOM_MODELS.md)**

## Feedback & Support
If you have questions, suggestions, or find any bugs, please feel free to open an issue or [contact me directly by mail](mailto:mb.strozyk@student.uw.edu.pl).

<!-- Later mayebe :): contact the maintainers at [your-email@domain.com]. -->


## Bibliography
If you use this library in your research, please cite:
* **m2aia**: Cordes, J., et al. "M2aia-Interactive, Mobile, and Memory-Efficient Analysis of MSI Data." *Journal of Open Source Software*.
    * *Note: We highly encourage using (and citing) m2aia as it serves as the foundational engine for data handling in this project.*
* **Skrajny et al.**: [Link to original paper](https://pubs.acs.org/doi/10.1021/acs.analchem.4c06913) (Note: While m2aia was not explicitly cited in the original Skrajny paper due to implementation differences at that time, it is the core engine of this refactored library).
