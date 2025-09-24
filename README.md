![CI](https://github.com/malaiwah/vllm-lmcache-cu128/workflows/Build%20and%20Push%20Container/badge.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/malaiwah/vllm-lmcache-cu128)

# vLLM + LMCache + Yarn Setup (RTX 5070/50-series)

This repository contains instructions and a containerized setup for
running **vLLM** with **LMCache** and **Yarn** extensions on consumer
RTX GPUs (e.g., RTX 5070, RTX 4080, RTX 5090). It is optimized for
**long-context inference (128k+)** with model caching and efficient GPU
utilization.

------------------------------------------------------------------------

## 🚀 Features

-   **vLLM** for high-throughput OpenAI-compatible serving
-   **LMCache** integration for KV cache persistence (CPU/NVMe offload)
-   **Yarn** models for extended context (128k+ tokens)
-   Built with **CUDA 12.8** and **PyTorch 2.5+**
-   Uses **Podman/Docker** containers
-   Tuned for **RTX consumer GPUs (Ada SM89 and Blackwell SM120, 16GB+ VRAM)**

------------------------------------------------------------------------

## 📦 Building the Container

``` bash
# Clone your local repo and cd into it
podman build -t vllm-lmcache-cu128:uv312 -f Containerfile .
```

The container uses `uv` to manage Python environments, so you don't have
to pre-install dependencies.

Alternatively, pull the pre-built signed image from [Docker Hub](https://hub.docker.com/r/malaiwah/vllm-lmcache-cu128):

``` bash
podman pull malaiwah/vllm-lmcache-cu128:uv312
```

------------------------------------------------------------------------

## ▶️ Running

See the example run script (`run.sh`)

## 🧪 Testing

To run tests locally:

``` bash
chmod +x test.sh
./test.sh
cat test_output.txt
```

This will perform inference on the model and display the output.

------------------------------------------------------------------------

## 🔐 Verifying Signatures

The pre-built images are signed with Cosign. To verify:

1. Install Cosign: `curl -L https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64 -o cosign && chmod +x cosign && sudo mv cosign /usr/local/bin/`

2. Verify the signature:
   ``` bash
   cosign verify --key cosign.pub malaiwah/vllm-lmcache-cu128:uv312
   ```

This ensures the image integrity and authenticity.

------------------------------------------------------------------------

## ⚡ Notes

-   For **128k context**, use **Llama-3.1-Yarn** or Qwen Yarn variants
    from Hugging Face.
-   Increase context by adjusting `--max-model-len` but ensure GPU VRAM
    is sufficient.
    -   On a 16GB RTX GPU, expect \~20--22k max length without
        Yarn/LMCache tricks.\
    -   With Yarn + LMCache (CPU/NVMe offload), 128k is achievable but
        slower.
-   Recommended: use **fast NVMe SSDs** for LMCache spillover.\
-   If running dual GPUs, remember consumer RTX cards **do not have
    NVLink**. Communication is via PCIe only.
-   CUDA arch support: **Ada (SM89, 4070/4080/5070)** and **Blackwell (SM120, 50-series)** with precompiled cubins + PTX fallback for future GPUs.

------------------------------------------------------------------------

## 📚 References

-   [vLLM](https://github.com/vllm-project/vllm)
-   [LMCache](https://github.com/LmCache/lmcache)
-   [Yarn](https://github.com/jquesnelle/yarn)
-   [Hugging Face Models](https://huggingface.co/models)
