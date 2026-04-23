# Usunięto sztywne platform=linux/amd64 # TODO - to jest problmeatyczne albo nie ładuje plików albo nie mogę tego odczytać 
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instalacja zależności
RUN apt-get update && apt-get install -y --no-install-recommends \
    git tree htop vim curl \
    libgl1 libopengl0 libgomp1 libfontconfig1 libdbus-1-3 \
    libxcursor1 libxinerama1 libxrandr2 libxi6 libxfixes3 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
    libtiff6 libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

# Instalacja uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/
ENV PATH="/uv/bin:${PATH}"

WORKDIR /app

# Konfiguracja M2aia
ENV PYM2AIA_VERSION_TAG=0.5.10
ENV M2AIA_BIN_DIR=/usr/local/lib/python3.12/site-packages/m2aia/bin
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/m2aia/bin

# Instalacja paczek (uv automatycznie dobierze odpowiednie binarne wersje lub emulację)
RUN uv pip install --system \
    ipykernel \
    notebook \
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

# PyTorch - dla Maca (Apple Silicon) i tak najlepiej używać CPU wewnątrz kontenera amd64
RUN uv pip install --system \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Kopiowanie i wypakowanie binariów M2aia
COPY M2aia-2025.07.00-linux-x86_64.tar.gz /tmp/
RUN mkdir -p /usr/local/lib/python3.12/site-packages/m2aia/bin && \
    tar -xzf /tmp/M2aia-2025.07.00-linux-x86_64.tar.gz -C /tmp/ --strip-components=1 && \
    cp -rp /tmp/bin/* /usr/local/lib/python3.12/site-packages/m2aia/bin/ || true && \
    rm -rf /tmp/M2aia-2025.07.00-linux-x86_64.tar.gz

# WAŻNE: Rejestracja kernela globalnie (--name i --display-name)
RUN python3 -m ipykernel install --sys-prefix --name m2aia_env --display-name "M2aia (Docker)"

EXPOSE 8888
CMD ["bash"]