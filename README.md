![CI](https://github.com/malaiwah/vllm-lmcache-cu128/workflows/Build%20and%20Push%20Container/badge.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/malaiwah/vllm-lmcache-cu128)

# vLLM + LMCache + LiteLLM Proxy Setup (RTX 5070/50-series)

This repository contains instructions and a containerized setup for
running **vLLM** with **LMCache**, **Yarn** extensions and **LiteLLM proxy**
with polyglot tool call normalization on consumer RTX GPUs
(e.g., RTX 5070, RTX 4080, RTX 5090). It is optimized for
**long-context inference (128k+)** with model caching, efficient GPU
utilization, and compatible tool calling.

## 🏗️ Architecture

The container uses **supervisord** to orchestrate three processes over Unix domain sockets for secure internal communication:

1. **vLLM** serves inference via `/run/vllm/instruct.sock`
2. **Nginx (backend)** proxies `/vllm/instruct/v1/*` to vLLM's UDS on internal port 9000
3. **LiteLLM proxy** with polyglot callback normalizes tool calls (e.g., `<tool_use>` → `tool_calls`)
4. **Nginx (frontend)** exposes the proxy at `:8000/v1/*` externally

This stack allows models with varying tool call syntax (Hermes `<tool_call>`, Anthropic `<tool_use>`) to work seamlessly with editors like Crush.

------------------------------------------------------------------------

## 🚀 Features

-   **vLLM** for high-throughput OpenAI-compatible serving
-   **LMCache** integration for KV cache persistence (CPU/NVMe offload)
-   **Yarn** models for extended context (128k+ tokens)
-   **LiteLLM proxy** with polyglot streaming handler for tool call normalization
-   Built with **CUDA 12.8** and **PyTorch 2.5+**
-   Uses **Podman/Docker** containers with UDS-only internal communication
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

**Important**: To persist logs, add volume mounts in `run.sh` or your podman run command:
```bash
-v /host/path/logs/supervisor:/var/log/supervisor:Z \
-v /host/path/logs/nginx:/var/log/nginx:Z
```
Replace `/host/path/logs` with your preferred host directory.

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
-   **API Compatibility**: Point clients at `http://<host>:8000/v1`. Container CLI remains identical to original vLLM (passes all arguments to vLLM process).
-   **Tool Call Normalization**: Use with models like Qwen that emit `<tool_call>` or `<tool_use>` - the proxy streaming handler converts to OpenAI `tool_calls` during response.

------------------------------------------------------------------------

## 📁 Internal Paths

| Process | Protocol | Path | Notes |
|---------|----------|------|-------|
| vLLM Server | UDS | `/run/vllm/vllm.sock` | Direct IPC to avoid exposed TCP |
| Nginx Backend | TCP | `127.0.0.1:9000` | Internal only, routes `/vllm/vllm/v1/*` |
| LiteLLM Proxy | TCP | `127.0.0.1:8001` | Internal, applies polyglot callback |
| Nginx Frontend | TCP | `:8000` | External, proxies `/v1/*` to LiteLLM |

## 📊 Logging

Logs are rotated daily with compression:
- Supervisor logs: 50MB per file, 5 backups each
- Nginx logs: daily rotation, 52 weeks retention

To persist logs across container restarts, volume-mount:
```bash
-v /host/path/logs:/var/log/supervisor:Z \
-v /host/path/nginx:/var/log/nginx:Z
```

------------------------------------------------------------------------

## 📚 References

-   [vLLM](https://github.com/vllm-project/vllm)
-   [LMCache](https://github.com/LmCache/lmcache)
-   [Yarn](https://github.com/jquesnelle/yarn)
-   [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/quick_start)
-   [Hugging Face Models](https://huggingface.co/models)
