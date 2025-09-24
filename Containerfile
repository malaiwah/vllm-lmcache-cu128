# syntax=docker/dockerfile:1.7

FROM docker.io/nvidia/cuda@${CUDA_BUILD_DIGEST} AS build

ARG JOBS=10
ENV JOBS=${JOBS}

# Version pins
ENV PYTHON_VERSION=3.12
ENV VLLM_COMMIT=8938774c79f185035bc3de5f19cfc7abaa242a5a
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128
# 12.8.1-cudnn-devel-ubuntu22.04 (24.04 exists)
# original was FROM docker.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS build
ENV CUDA_BUILD_DIGEST=sha256:2a015be069bda4de48d677b6e3f271a2794560c7d788a39a18ecf218cae0751d
# 12.8.1-cudnn-runtime-ubuntu22.04 (24.04 exists as well)
ENV CUDA_RUNTIME_DIGEST=sha256:05de765c12d993316f770e8e4396b9516afe38b7c52189bce2d5b64ef812db58

# toolchain-wide limits
#ENV MAKEFLAGS="-j${JOBS}"
#ENV CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
#ENV NINJAFLAGS="-j${JOBS}"
#ENV NINJA_NUM_CORES="${JOBS}"
#ENV MAX_JOBS="${JOBS}"
#ENV CMAKE_CUDA_FLAGS="--threads=${JOBS}"
#ENV NVCC_THREADS="${JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
#ENV TORCH_CUDA_ARCH_LIST="8.9+PTX;12.0+PTX"
#ENV TRITON_CUDA_ARCH_LIST="89;120"
#ENV FLASHINFER_CUDA_ARCHS="89;120"
#ENV CUDA_ARCH_LIST="8.9+PTX;12.0+PTX"

ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV TRITON_CUDA_ARCH_LIST="120"
ENV FLASHINFER_CUDA_ARCHS="120"
ENV CUDA_ARCH_LIST="12.0+PTX"

ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV UV_PYTHON_PREFER_PREBUILT=1
ENV UV_LINK_MODE=copy
ENV CC=ccache
ENV CXX=ccache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      git build-essential curl ca-certificates pkg-config python3 python3-pip ninja-build ccache \
    && apt-get clean

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv venv --python ${PYTHON_VERSION} /opt/venv

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python -U pip wheel setuptools

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python --pre torch torchvision \
      --index-url ${TORCH_INDEX_URL}

WORKDIR /opt/app
COPY requirements.txt /tmp/requirements.txt
RUN git clone https://github.com/vllm-project/vllm.git && cd vllm && git config advice.detachedHead false && git checkout ${VLLM_COMMIT}
WORKDIR /opt/app/vllm

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python --verbose .

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt --no-binary lmcache --no-binary flashinfer-python --force-reinstall lmcache --force-reinstall flashinfer-python

# Optional: verify dependency health (non-fatal)
RUN /opt/venv/bin/python -m pip check || true

RUN printf "import sys, torch, vllm, numpy as np, numba, llvmlite, setuptools, lmcache\nprint('Python:', sys.version.split()[0])\nprint('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)\nprint('vLLM:', vllm.__version__)\nprint('NumPy:', np.__version__)\nprint('Numba:', numba.__version__)\nprint('LLVMLite:', llvmlite.__version__)\nprint('Setuptools:', setuptools.__version__)\nprint('LMCache:', lmcache.__version__)\n" | /opt/venv/bin/python -

RUN /opt/venv/bin/python -m pip freeze > /opt/venv/requirements.freeze.txt

FROM docker.io/nvidia/cuda@${CUDA_RUNTIME_DIGEST} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV CC=gcc
ENV CXX=g++

# add a compiler for Triton/TorchInductor JIT (small, safe)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cuda-nvcc-12-8 \
      libcurand-dev-12-8 \
    && apt-get clean

# add the toolkit for fp4/fp8 <cublasLt.h>
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
      cuda-toolkit-12-8=12.8.0-1 \
      libcublas-12-8=12.8.4.1-1 \
    && apt-get clean

# bring both the venv AND uv’s installed Python
COPY --from=build /opt/venv /opt/venv
COPY --from=build /root/.local /root/.local

WORKDIR /srv
VOLUME ["/root/.cache/huggingface"]

LABEL org.opencontainers.image.title="vLLM + LMCache (Ada/Blackwell, cu128, UV 3.12)"
LABEL org.opencontainers.image.source="https://github.com/malaiwah/vllm-lmcache-cu128"
LABEL org.opencontainers.image.description="vLLM built for RTX 40/50 (sm_89/120) with LMCache & FlashInfer."
LABEL org.opencontainers.image.vllm_commit="${VLLM_COMMIT}"
LABEL org.opencontainers.image.cuda_build_digest="${CUDA_BUILD_DIGEST}"
LABEL org.opencontainers.image.cuda_runtime_digest="${CUDA_RUNTIME_DIGEST}"

EXPOSE 8000

ENTRYPOINT ["/opt/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server"]
