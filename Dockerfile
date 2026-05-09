# --------------------------------------------------------------------------
# RoSE SeisBench tutorial image.
#
# Build:
#   docker build -t rose:cpu .
#
# Run (mount the dataset at /data; tutorials read ROSE_DATA_DIR=/data):
#   docker run --rm -it \
#       -v $(pwd)/data/rose:/data:ro \
#       rose:cpu python 01_load_and_browse.py
#
# For GPU training, build with the CUDA target:
#   docker build --build-arg TORCH_VARIANT=cuda -t rose:cuda .
#   docker run --rm -it --gpus all -v $(pwd)/data/rose:/data:ro rose:cuda
# --------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# System libs needed by obspy/h5py/matplotlib (libgomp for sklearn-ish numerical libs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        libhdf5-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single static binary).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install Python deps via uv into the system interpreter, with the chosen torch
# variant. Default is CPU; flip TORCH_VARIANT=cuda for the CUDA index.
ARG TORCH_VARIANT=cpu
COPY pyproject.toml requirements.txt README.md ./
COPY rose ./rose

RUN if [ "${TORCH_VARIANT}" = "cuda" ]; then \
        uv pip install --system -e ".[cuda]" ; \
    else \
        uv pip install --system -e ".[cpu]" ; \
    fi

# Copy docs/examples for self-contained images. The published `data/` directory
# (~35 GB) is intentionally excluded by .dockerignore — mount it at /data.
COPY docs ./docs
COPY examples ./examples

# Sanity check that the package and key heavy deps import.
RUN python -c "import rose, seisbench, obspy, torch; \
print('rose OK; seisbench', seisbench.__version__, '| torch', torch.__version__)"

# Conventional mount point for the published dataset; tutorials read this via
# the ROSE_DATA_DIR env var.
ENV ROSE_DATA_DIR=/data

# Default working dir for users running tutorials.
WORKDIR /app/examples
CMD ["python", "01_load_and_browse.py"]
