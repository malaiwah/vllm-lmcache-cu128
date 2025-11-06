#!/bin/bash
source .env

IMAGE=docker.io/vllm/vllm-openai:nightly
IMAGE=localhost/vllm:local
IMAGE=localhost/vllm:lmcache
LOCAL_PORT=8000
CONTAINER_PORT=8000
MODEL=cerebras/GLM-4.5-Air-REAP-82B-A12B-FP8
MODEL_LEN=131072
GPU_UTIL=0.97

podman run --rm -it \
  --device nvidia.com/gpu=all \
  --name vllm \
  --ipc=host \
  -p ${LOCAL_PORT}:${CONTAINER_PORT} \
  --security-opt label=disable \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TORCH_FORCE_ALLOW_TF32=1 -e NVIDIA_TF32_OVERRIDE=1 \
  -e PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128" \
  -e LMCACHE_CONFIG_FILE=/srv/lmcache.yaml \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:Z \
  -v $PWD/lmcache.yaml:/srv/lmcache.yaml:ro \
  ${IMAGE} \
  "${MODEL}" \
  --max-model-len ${MODEL_LEN} \
  --served-model-name "local/vllm" \
  --gpu-memory-utilization ${GPU_UTIL} \
  --max_num_seqs 4 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 2048 \
  --enable-auto-tool-choice --tool-call-parser glm45 --reasoning-parser glm45 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","limit_tokens":32768}' \
#  END
