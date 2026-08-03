
# MSIAutoEncoderWrapper
[Documentation](https://maxi7524.github.io/MSIAutoEncoderWrapper/)

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


## Installation

Here we provide commands to set up an environment with one of the following
configurations:

* `cpu` — installs the CPU-only PyTorch build (Linux and Windows).
* `mps` — installs the CPU-only PyTorch build (macOS, both mps and other CPUs).
   > REMARK: 
   > It is also compatible with other CPUs, we discern this option because of lack libraries for m2aia. 

* `cu118` — installs the PyTorch build for CUDA 11.8.

### System packages

To use the `m2aia` imzML readers, install the system libraries and
OpenSlide. The following command supports Debian and Ubuntu systems:


```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev libgomp1 libopenslide-dev
```

> REMARK:
> We also implemented `PyImzMLReader` for compatibility. 

### Environment

Run the commands from the project root. Choose exactly one environment manager
and one configuration.

#### uv manager (suggested)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then
create and synchronize the project environment:

To install all basic packages run:

```bash
uv sync 
```

To install / change TORCH version run: 

```bash
# CPU: Linux or Windows, includes m2aia package and loader
uv sync --extra cpu
```

or:

```bash
# CUDA 11.8: Linux or Windows with CUDA 11.8; includes m2aia
uv sync --extra cu118
```

or:
```bash
# MPS: macOS with Apple Metal Performance Shaders, excludes m2aia
uv sync --extra mps
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

# MPS 
uv run --extra mps pytest
uv run --extra mps python
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

For macOS with MPS, install the standard PyPI build of PyTorch:

```bash
python -m pip install -e ".[mps]"
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

For macOS with MPS, install the standard PyPI build of PyTorch:

```bash
python -m pip install -e ".[mps]"
```

Verify the selected PyTorch build:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Tutorials

The executable tutorials are available in the Sphinx documentation:

- **Workspace and contexts:** [open tutorials](https://maxi7524.github.io/MSIAutoEncoderWrapper/tutorials/workspace-and-contexts/)
- **Autoencoders:** [open tutorials](https://maxi7524.github.io/MSIAutoEncoderWrapper/tutorials/autoencoder/)
- **Cohort models:** [open tutorials](https://maxi7524.github.io/MSIAutoEncoderWrapper/tutorials/cohort-models/)
- **Dataset management:** [open tutorials](https://maxi7524.github.io/MSIAutoEncoderWrapper/tutorials/dataset-management/)
- **CLI and configuration:** [open tutorials](https://maxi7524.github.io/MSIAutoEncoderWrapper/tutorials/cli-and-configuration/)

See the [tutorial index](https://maxi7524.github.io/MSIAutoEncoderWrapper/tutorials/)
for the recommended order and workspace preparation.

## Documentation

The complete documentation is available at:

https://maxi7524.github.io/MSIAutoEncoderWrapper/


## Feedback & Support
If you have questions, suggestions, or find any bugs, please feel free to open an issue or [contact me directly by mail](mailto:mb.strozyk@student.uw.edu.pl).

<!-- Later mayebe :) contact the maintainers at [your-email@domain.com]. -->


## Bibliography
If you use this library in your research, please cite:
* **m2aia**: Cordes, J., et al. "M2aia-Interactive, Mobile, and Memory-Efficient Analysis of MSI Data." *Journal of Open Source Software*.
    * *Note: We highly encourage using (and citing) m2aia as it serves as the foundational engine for data handling in this project.*
* **Skrajny et al.**: [Link to original paper](https://pubs.acs.org/doi/10.1021/acs.analchem.4c06913) (Note: While m2aia was not explicitly cited in the original Skrajny paper due to implementation differences at that time, it is the core engine of this refactored library).
