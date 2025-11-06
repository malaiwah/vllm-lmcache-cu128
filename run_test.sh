#!/bin/bash
source .env
# uv venv .. uv pip install huggingface_hub[cli]
source .venv/bin/activate

IMAGE=2be0efef9405

LOCAL_PORT=8001
CONTAINER_PORT=8000
#
# allocator tuned to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True,garbage_collection_threshold:0.7,max_split_size_mb:64"
export VLLM_USE_FLASHINFER=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

podman run --rm -it \
  --device nvidia.com/gpu=all \
  --ipc=host \
  -p ${LOCAL_PORT}:${CONTAINER_PORT} \
  --security-opt label=disable \
  -e HF_TOKEN="$HF_TOKEN" \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e VLLM_USE_FLASHINFER=0 \
  -e VLLM_ATTENTION_BACKEND=TORCH_SDPA \
  -v $PWD/vllm-logs/supervisor:/var/log/supervisor:Z \
  ${IMAGE} \
    --model zai-org/GLM-4.5-Air-FP8 \
    --served-model-name glm45-air-fp8b \
    --max-model-len 110000 \
    --gpu-memory-utilization 0.82 \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 1
