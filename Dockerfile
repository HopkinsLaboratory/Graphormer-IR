# Use an NVIDIA CUDA runtime image that provides the necessary libraries (e.g. cuBLAS)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent tzdata from prompting during installation and set timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set CUDA identifier for PyG wheels; adjust if needed (e.g. to cu121)
ENV CUDA=cu118

# Install system packages (including wget, git, build-essential, and ca-certificates)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH to include conda and create a conda environment
ENV PATH="/opt/conda/bin:$PATH"
RUN conda create -n graphormer-ir python=3.9.18 -y && \
    /opt/conda/envs/graphormer-ir/bin/pip install --upgrade pip

# Make the conda environment the default by prepending its bin directory to PATH
ENV PATH="/opt/conda/envs/graphormer-ir/bin:$PATH"
ENV CONDA_DEFAULT_ENV=graphormer-ir

WORKDIR /workspace

# Cache-busting ARG: update this value manually when you want to re-download the repository
ARG CACHEBUST=1
# Remove any existing clone (if present) and clone the latest repository
RUN rm -rf Graphormer-RT && git clone git@github.com:HopkinsLaboratory/Graphormer-IR.git

# Install fairseq (switch to the fairseq subdirectory)
WORKDIR /workspace/Graphormer-RT/fairseq

# Downgrade pip so that omegaconf versions used by fairseq are accepted
RUN pip install --upgrade "pip<24.1"

# Install fairseq from the current repository in editable mode (this builds its extensions)
RUN pip install --editable ./

# Install dependencies

# Install DGL using the appropriate wheel URL (for torch-2.1 and cu118)
RUN pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html

# Install torch-geometric; here we pin a version that works with our setup (e.g. 2.4.0)
RUN pip install torch-geometric==2.4.0

# Install torch-scatter and torch-sparse wheels built for torch-2.1.0+cu118
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html

# Install additional dependencies
RUN pip install ogb==1.3.2
RUN pip install rdkit-pypi==2022.9.5
RUN pip install matplotlib
# RUN pip install googledrivedownloader==0.4
RUN pip install numpy==1.23
RUN pip install dgllife==0.3.2
RUN pip install mordred==1.2.0
RUN pip install torchaudio==2.1.0
RUN pip install rdkit==2023.9.1
RUN pip install rdkit-pypi==2022.9.5
# RUN pip install dgl==1.1.3

# Set working directory back to the repository root
WORKDIR /workspace/Graphormer-IR

# Set the default command to launch an interactive bash shell (the conda environment is already active)
CMD ["/bin/bash"]
