# syntax=docker/dockerfile:1.7

ARG CUDA_BUILD_DIGEST=sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c
ARG CUDA_RUNTIME_DIGEST=sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9
ARG TORCH_NIGHTLY_INDEX=https://download.pytorch.org/whl/nightly/cu128
ARG UV_INDEX_STRATEGY=unsafe-best-match
ARG UV_PRERELEASE=allow
ARG VLLM_COMMIT=3758757377b713b6acc997d0ac2c5dd49c332278
ARG XFORMERS_COMMIT=5d4b92a5e5a9c6c6d4878283f47d82e17995b468
ARG XFORMERS_VERSION=0.0.33+5d4b92a5.d20251029
ARG XFORMERS_MAX_JOBS=16
ARG CUDA_ARCH_LIST_PTX="8.9+PTX;10.0+PTX;12.0+PTX"
ARG CUDA_ARCH_LIST_NUMERIC="89;100;120"

FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@${CUDA_BUILD_DIGEST} AS build-base
ARG TORCH_NIGHTLY_INDEX
ARG UV_INDEX_STRATEGY
ARG VLLM_COMMIT
ARG XFORMERS_COMMIT
ARG XFORMERS_VERSION
ARG UV_PRERELEASE
ARG CUDA_ARCH_LIST_PTX
ARG CUDA_ARCH_LIST_NUMERIC
# Update packages, except the pinned ones
RUN apt-get update && apt-get dist-upgrade -y && apt-get clean

#ARG JOBS=16
#ENV JOBS=${JOBS}
ENV CUDA_BUILD_DIGEST=${CUDA_BUILD_DIGEST}
ENV CUDA_RUNTIME_DIGEST=${CUDA_RUNTIME_DIGEST}

# Version pins
ENV PYTHON_VERSION=3.12
ENV VLLM_COMMIT=${VLLM_COMMIT}
ENV XFORMERS_COMMIT=${XFORMERS_COMMIT}
ENV XFORMERS_VERSION=${XFORMERS_VERSION}
ENV TORCH_INDEX_URL=${TORCH_NIGHTLY_INDEX}
ENV PIP_EXTRA_INDEX_URL=${TORCH_NIGHTLY_INDEX}
ENV UV_EXTRA_INDEX_URL=${TORCH_NIGHTLY_INDEX}
ENV UV_INDEX_STRATEGY=${UV_INDEX_STRATEGY}
ENV UV_PRERELEASE=${UV_PRERELEASE}

# Limits to keep memory usage under control
#ENV CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
#ENV MAX_JOBS="${JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST_PTX}"
ENV CUDA_ARCH_LIST="${CUDA_ARCH_LIST_PTX}"
ENV CUDAARCHS="${CUDA_ARCH_LIST_NUMERIC}"
ENV TRITON_CUDA_ARCH_LIST="${CUDA_ARCH_LIST_NUMERIC}"
ENV FLASHINFER_CUDA_ARCHS="${CUDA_ARCH_LIST_NUMERIC}"
ENV FLASH_ATTENTION_CUDA_ARCHS="${CUDA_ARCH_LIST_NUMERIC}"
ENV FLASH_ATTENTION_FORCE_BUILD=1

ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV UV_PYTHON_PREFER_PREBUILT=1
ENV UV_LINK_MODE=copy

# Tell CMake to launch compilers via sccache
ENV CMAKE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache"
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

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

FROM build-base AS xformers-wheel
ARG XFORMERS_COMMIT
ARG XFORMERS_VERSION
ARG XFORMERS_MAX_JOBS
ENV MAX_JOBS=${XFORMERS_MAX_JOBS}
ENV CCACHE_DIR=/root/.cache/ccache
WORKDIR /opt/app
# Build the pinned xFormers wheel locally to satisfy vLLM's dependency.
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/ccache,sharing=locked \
    git clone https://github.com/facebookresearch/xformers.git /tmp/xformers \
    && cd /tmp/xformers \
    && git config advice.detachedHead false \
    && git checkout ${XFORMERS_COMMIT} \
    && git submodule update --init --recursive \
    && mkdir -p /opt/dist/xformers \
    && BUILD_VERSION=${XFORMERS_VERSION} /opt/venv/bin/python setup.py bdist_wheel --dist-dir /opt/dist/xformers --verbose \
    && rm -rf /tmp/xformers

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    uv pip install --python /opt/venv/bin/python /opt/dist/xformers/*.whl

WORKDIR /opt/app/vllm

FROM xformers-wheel AS build
ARG VLLM_COMMIT
ARG XFORMERS_COMMIT
ARG XFORMERS_VERSION
ARG UV_PRERELEASE

WORKDIR /opt/app/vllm

RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python --no-build-isolation --verbose .

#    uv pip install -v --no-build-isolation --no-binary=:all: lmcache==0.3.6 && \
WORKDIR /opt/app
# Force recompile for Blackwell support
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip uninstall --python /opt/venv/bin/python lmcache flashinfer-python && \
    uv pip install --python /opt/venv/bin/python --no-binary flashinfer-python --force-reinstall flashinfer-python && \
    pip -v --python /opt/venv/bin/python install --no-binary :all: --no-build-isolation lmcache==0.3.6

# Ensure the nightly numerical stack is installed consistently
RUN --mount=type=cache,target=/root/.cache/uv,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0,sharing=locked \
    --mount=type=cache,target=/root/.ccache,sharing=locked \
    uv pip install --python /opt/venv/bin/python \
      "numpy" "numba" "llvmlite" "setuptools"

# Verify dependency health (non-fatal)
RUN /opt/venv/bin/python -m pip check || true

COPY tools/check_archs.py /usr/local/bin/check_archs.py
RUN /opt/venv/bin/python /usr/local/bin/check_archs.py

RUN printf "import sys, torch, vllm, numpy as np, numba, llvmlite, setuptools\nprint('Python:', sys.version.split()[0])\nprint('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)\nprint('vLLM:', vllm.__version__)\nprint('NumPy:', np.__version__)\nprint('Numba:', numba.__version__)\nprint('LLVMLite:', llvmlite.__version__)\nprint('Setuptools:', setuptools.__version__)\n" | /opt/venv/bin/python -

RUN /opt/venv/bin/python -m pip freeze > /opt/venv/requirements.freeze.txt

FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@${CUDA_RUNTIME_DIGEST} AS runtime
ARG VLLM_COMMIT
ARG XFORMERS_COMMIT
ARG XFORMERS_VERSION
ENV VLLM_COMMIT=${VLLM_COMMIT}
ENV XFORMERS_COMMIT=${XFORMERS_COMMIT}
ENV XFORMERS_VERSION=${XFORMERS_VERSION}
# Update packages, except the pinned ones
RUN apt-get update && apt-get dist-upgrade -y && apt-get clean

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV CC=gcc
ENV CXX=g++
ENV PYTHONPATH=/opt
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
ARG TORCH_NIGHTLY_INDEX
ARG UV_INDEX_STRATEGY
ARG UV_PRERELEASE
ENV PIP_EXTRA_INDEX_URL=${TORCH_NIGHTLY_INDEX}
ENV UV_EXTRA_INDEX_URL=${TORCH_NIGHTLY_INDEX}
ENV UV_INDEX_STRATEGY=${UV_INDEX_STRATEGY}
ENV UV_PRERELEASE=${UV_PRERELEASE}

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
