# Use the official Python 3.12 slim image as base
FROM --platform=linux/amd64 python:3.12-slim

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for M2aia, Qt5, and patchelf
# Added 'patchelf' which is often required by M2aia to resolve internal library paths
RUN apt-get update && apt-get install -y --no-install-recommends \
    # basic libraries
    git \
    tree \
    htop \
    vim \
    curl \
    # m2aia dependencies
    libglu1-mesa-dev \
    libtiff5-dev \
    qtbase5-dev \
    libqt5svg5-dev \
    libqt5opengl5-dev \
    libqt5xmlpatterns5-dev \
    qtwebengine5-dev \
    qttools5-dev \
    libqt5charts5-dev \
    libqt5x11extras5-dev \
    libopenslide-dev \
    wget \
    tar \
    build-essential \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/
ENV PATH="/uv/bin:${PATH}"

# Set working directory inside the container
WORKDIR /app

# Set environment variables for M2aia Engine
# Pointing LD_LIBRARY_PATH directly to the site-packages location
ENV PYM2AIA_VERSION_TAG=0.5.10
ENV M2AIA_BIN_DIR=/usr/local/lib/python3.12/site-packages/m2aia/bin
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/m2aia/bin

# Step 1: Install standard libraries from PyPI
RUN uv pip install --system \
    ipykernel \
    tabulate \
    pandas \
    numpy \
    polars \
    matplotlib \
    seaborn \
    scipy \
    statsmodels \
    scikit-learn \
    m2aia

# Step 2: Install PyTorch (CPU version)
RUN uv pip install --system \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Step 3: Extract binaries directly into the site-packages directory
# CHANGE: Added 'mkdir -p' for the internal m2aia/bin folder to ensure it exists before copying
COPY M2aia-2025.07.00-linux-x86_64.tar.gz /tmp/
RUN mkdir -p /tmp/m2aia_bin && \
    tar -xzf /tmp/M2aia-2025.07.00-linux-x86_64.tar.gz -C /tmp/m2aia_bin --strip-components=1 && \
    mkdir -p /usr/local/lib/python3.12/site-packages/m2aia/bin && \
    cp -rp /tmp/m2aia_bin/bin/* /usr/local/lib/python3.12/site-packages/m2aia/bin/ && \
    rm -rf /tmp/m2aia_bin /tmp/M2aia-2025.07.00-linux-x86_64.tar.gz

# Register the kernel for Jupyter
RUN python3 -m ipykernel install --user --name m2aia_env --display-name "Python 3.12 (M2aia_Docker)"

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/m2aia/bin:$LD_LIBRARY_PATH

# Expose port for Jupyter
EXPOSE 8888

CMD ["bash"]