# syntax=docker/dockerfile:1.7

FROM docker.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV UV_PYTHON_PREFER_PREBUILT=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      git build-essential curl ca-certificates pkg-config python3 python3-pip ninja-build \
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
RUN git clone --depth=1 https://github.com/vllm-project/vllm.git
WORKDIR /opt/app/vllm

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python .

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python -U lmcache && \
    uv pip install --python /opt/venv/bin/python --no-binary lmcache --force-reinstall "lmcache==0.3.6"

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python -U flashinfer-python && \
    uv pip install --python /opt/venv/bin/python --no-binary flashinfer-python --force-reinstall flashinfer-python

# Torch warns because it wants the new meta-package.
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python nvidia-ml-py

# Pin versions so vLLM+Numba are compatible
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python \
      "numpy==2.2.2" "numba==0.61.2" "llvmlite==0.44.0" "setuptools==79.0.0"

# Optional: verify dependency health (non-fatal)
RUN /opt/venv/bin/python -m pip check || true

RUN printf "import sys, torch, vllm\nprint('Python:', sys.version.split()[0])\nprint('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)\nprint('vLLM:', vllm.__version__)\n" | /opt/venv/bin/python -

RUN /opt/venv/bin/python -m pip freeze > /opt/venv/requirements.freeze.txt

FROM docker.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS runtime

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
