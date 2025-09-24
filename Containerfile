# syntax=docker/dockerfile:1.7

FROM docker.io/nvidia/cuda@sha256:2a015be069bda4de48d677b6e3f271a2794560c7d788a39a18ecf218cae0751d AS build

ARG JOBS=10
ENV JOBS=${JOBS}

# toolchain-wide limits
ENV MAKEFLAGS="-j${JOBS}"
ENV CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
ENV NINJAFLAGS="-j${JOBS}"
ENV NINJA_NUM_CORES="${JOBS}"
ENV MAX_JOBS="${JOBS}"
ENV CMAKE_CUDA_FLAGS="--threads=${JOBS}"
ENV NVCC_THREADS="${JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX;12.0+PTX"
ENV TRITON_CUDA_ARCH_LIST="89;120"
ENV FLASHINFER_CUDA_ARCHS="89;120"
ENV CUDA_ARCH_LIST="8.9+PTX;12.0+PTX"

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

RUN uv venv --python 3.12 /opt/venv

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python -U pip wheel setuptools

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python --pre torch torchvision \
      --index-url https://download.pytorch.org/whl/nightly/cu128

WORKDIR /opt/app
COPY requirements.txt /tmp/requirements.txt
RUN git clone https://github.com/vllm-project/vllm.git && cd vllm && git config advice.detachedHead false && git checkout 8938774c79f185035bc3de5f19cfc7abaa242a5a
WORKDIR /opt/app/vllm

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python .

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt --no-binary lmcache --no-binary flashinfer-python --force-reinstall lmcache --force-reinstall flashinfer-python

# Optional: verify dependency health (non-fatal)
RUN /opt/venv/bin/python -m pip check || true

RUN printf "import sys, torch, vllm\nprint('Python:', sys.version.split()[0])\nprint('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)\nprint('vLLM:', vllm.__version__)\n" | /opt/venv/bin/python -

RUN /opt/venv/bin/python -m pip freeze > /opt/venv/requirements.freeze.txt

FROM docker.io/nvidia/cuda@sha256:05de765c12d993316f770e8e4396b9516afe38b7c52189bce2d5b64ef812db58 AS runtime

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

# bring both the venv AND uv’s installed Python
COPY --from=build /opt/venv /opt/venv
COPY --from=build /root/.local /root/.local

WORKDIR /srv
VOLUME ["/root/.cache/huggingface"]

LABEL org.opencontainers.image.title="vLLM + LMCache (Blackwell/cu128, UV 3.12)"
LABEL org.opencontainers.image.source="https://github.com/malaiwah/vllm-lmcache-cu128"
LABEL org.opencontainers.image.description="vLLM built for RTX 50 (sm_120) with LMCache & FlashInfer."

EXPOSE 8000

ENTRYPOINT ["/opt/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server"]
