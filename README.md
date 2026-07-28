
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

Here we provide commands to set up an environment with one of the following
configurations:

* `cpu` — installs the CPU-only PyTorch build.
* `cu118` — installs the PyTorch build for CUDA 11.8.

### System packages

To use the imzML readers, install the system libraries required by `m2aia` and
OpenSlide. The following command supports Debian and Ubuntu systems:


```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev libgomp1 libopenslide-dev
```

<!-- TODO: Add a tested cross-platform system dependency installation script. -->

### Environment

Run the commands from the project root. Choose exactly one environment manager
and one configuration.

#### uv manager (suggested)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then
create and synchronize the project environment:

```bash
# CPU
uv sync --extra cpu
```

or:

```bash
# CUDA 11.8
uv sync --extra cu118
```

The environment is stored in `.venv`. Pass the selected configuration to
`uv run` to keep the environment synchronized while executing commands:

```bash
# CPU
uv run --extra cpu pytest
uv run --extra cpu python

# CUDA 11.8
uv run --extra cu118 pytest
uv run --extra cu118 python
```

#### Conda managers

The same setup works with Conda, Mamba, or Micromamba. The examples below use
`conda`; replace it consistently with `mamba` or `micromamba` when needed.

```bash
# Create and activate the environment
conda create --name msi_env python=3.12 pip -y
conda activate msi_env

# CPU
python -m pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[cpu]"
```

For CUDA 11.8, replace the two installation commands with:

```bash
python -m pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ".[cu118]"
```

#### venv manager

Create a standard Python virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU
python -m pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[cpu]"
```

For CUDA 11.8, replace the two installation commands with:

```bash
python -m pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ".[cu118]"
```

Verify the selected PyTorch build:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
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
