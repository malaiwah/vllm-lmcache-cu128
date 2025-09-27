# syntax=docker/dockerfile:1.7

ARG CUDA_BUILD_DIGEST=sha256:468c101db63b1fd84b05dd082f8bda87326c86ff5f7356b5e5aa37f9b8585ca5
ARG CUDA_RUNTIME_DIGEST=sha256:2189eb90b6f7a93003344a5e9d45aeed7cd6158bffb41d9fbe8b1b1a624533af

FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS build
# Update packages, except the pinned ones
RUN apt-get update && apt-get dist-upgrade -y && apt-get clean

ARG JOBS=16
ENV JOBS=${JOBS}
ENV CUDA_BUILD_DIGEST=${CUDA_BUILD_DIGEST}
ENV CUDA_RUNTIME_DIGEST=${CUDA_RUNTIME_DIGEST}

# Version pins
ENV PYTHON_VERSION=3.12
ENV VLLM_COMMIT=8938774c79f185035bc3de5f19cfc7abaa242a5a
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128

# Limits to keep memory usage under control
ENV CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
ENV MAX_JOBS="${JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX;12.0+PTX"
ENV TRITON_CUDA_ARCH_LIST="89;120"
ENV FLASHINFER_CUDA_ARCHS="89;120"
ENV CUDA_ARCH_LIST="8.9+PTX;12.0+PTX"

ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV UV_PYTHON_PREFER_PREBUILT=1
ENV UV_LINK_MODE=copy

# Tell CMake to launch compilers via sccache
ENV CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      git build-essential curl ca-certificates pkg-config python3 python3-pip python3-dev ninja-build sccache cmake \
    && apt-get clean

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv venv --python ${PYTHON_VERSION} --managed-python /opt/venv

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python -U pip wheel setuptools

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python --pre torch torchvision \
      --index-url ${TORCH_INDEX_URL}

WORKDIR /opt/app
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt

RUN git clone https://github.com/vllm-project/vllm.git && cd vllm && git config advice.detachedHead false && git checkout ${VLLM_COMMIT}
WORKDIR /opt/app/vllm

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python --no-build-isolation --verbose .

WORKDIR /opt/app
# Force recompile for Blackwell support
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python --no-binary lmcache --force-reinstall lmcache && \
    uv pip install --python /opt/venv/bin/python --no-binary flashinfer-python --force-reinstall flashinfer-python

# Pin versions so vLLM+Numba are compatible
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python \
      "numpy==2.2.2" "numba==0.61.2" "llvmlite==0.44.0" "setuptools==79.0.0"

# Verify dependency health (non-fatal)
RUN /opt/venv/bin/python -m pip check || true

RUN printf "import sys, torch, vllm, numpy as np, numba, llvmlite, setuptools\nprint('Python:', sys.version.split()[0])\nprint('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)\nprint('vLLM:', vllm.__version__)\nprint('NumPy:', np.__version__)\nprint('Numba:', numba.__version__)\nprint('LLVMLite:', llvmlite.__version__)\nprint('Setuptools:', setuptools.__version__)\n" | /opt/venv/bin/python -

RUN /opt/venv/bin/python -m pip freeze > /opt/venv/requirements.freeze.txt

FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime
# Update packages, except the pinned ones
RUN apt-get update && apt-get dist-upgrade -y && apt-get clean

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV CC=gcc
ENV CXX=g++
ENV PYTHONPATH=/opt

# add a compiler for Triton/TorchInductor JIT (small, safe)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cuda-nvcc-12-8 \
      nvidia-cuda-dev \
      libcurand-dev-12-8 \
      logrotate \
      supervisor \
      nginx \
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

# Create runtime directories for Unix domain sockets and logs
RUN mkdir -p /run/vllm /run/nginx /run/litellm /var/log/supervisor /srv/litellm

# Monkey patch, this should be in the previous stage (requirements.txt) instead -- TODO
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python \
      "litellm[proxy]"

# Copy configurations and polyglot handler
COPY polyglot/polyglot_tools_stream_handler.py /srv/litellm/polyglot_tools_stream_handler.py
COPY config/supervisord.conf /etc/supervisor/supervisord.conf
COPY config/nginx.conf /etc/nginx/nginx.conf
COPY config/logrotate.d/nginx /etc/logrotate.d/nginx
COPY config/litellm.yaml /srv/litellm/litellm.yaml

WORKDIR /srv
VOLUME ["/root/.cache/huggingface"]

LABEL org.opencontainers.image.title="vLLM + LMCache + LiteLLM Proxy (Ada/Blackwell, cu128, UV 3.12)"
LABEL org.opencontainers.image.source="https://github.com/malaiwah/vllm-lmcache-cu128"
LABEL org.opencontainers.image.description="vLLM with LMCache, FlashInfer, and LiteLLM proxy with polyglot tool call normalization for RTX 40/50-series."
LABEL org.opencontainers.image.vllm_commit="${VLLM_COMMIT}"
LABEL org.opencontainers.image.cuda_build_digest="${CUDA_BUILD_DIGEST}"
LABEL org.opencontainers.image.cuda_runtime_digest="${CUDA_RUNTIME_DIGEST}"

EXPOSE 8000

# Copy and set executable entrypoint script
COPY config/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

ENTRYPOINT ["/usr/local/bin/start.sh"]
