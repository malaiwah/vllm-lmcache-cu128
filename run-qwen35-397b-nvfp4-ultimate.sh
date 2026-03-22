#!/usr/bin/env bash
set -euo pipefail

source .env

IMAGE="${IMAGE:-localhost/vllm-cu130-ultimate:latest}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
CONTAINER_PORT=8000
MODEL="${MODEL:-lukealonso/Qwen3.5-397B-A17B-NVFP4}"
MODEL_LEN="${MODEL_LEN:-524288}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen35-397b}"
HF_HOME="${HF_HOME:-/mnt/vault/llm/huggingface}"
CONFIGS_DIR="${CONFIGS_DIR:-/mnt/vault/llm/vllm+lmcache/configs}"
CONFIG_DIR="${CONFIG_DIR:-/mnt/vault/llm/vllm+lmcache/config}"
LMCACHE_DIR="${LMCACHE_DIR:-/mnt/fast/lmcache/qwen35-397b}"
GPU_UTIL="${GPU_UTIL:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":1}}"

HF_OVERRIDES="${HF_OVERRIDES:-{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":2.0,\"original_max_position_embeddings\":262144}}}}"
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"enable_cache_usage_details_in_response\":true}}}"

mkdir -p "${LMCACHE_DIR}"

exec podman run --rm -it \
  --name "${CONTAINER_NAME}" \
  --replace \
  --device nvidia.com/gpu=all \
  --ipc=host \
  -p "${LOCAL_PORT}:${CONTAINER_PORT}" \
  -v "${HF_HOME}:/root/.cache/huggingface:Z" \
  -v "${CONFIGS_DIR}:/configs:ro,Z" \
  -v "${CONFIG_DIR}:/config:ro,Z" \
  -v "${LMCACHE_DIR}:/root/.cache/lmcache:Z" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e OMP_NUM_THREADS=4 \
  -e NCCL_IB_DISABLE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TUNED_CONFIG_FOLDER=/configs \
  -e LMCACHE_CONFIG_FILE=/config/lmcache.yaml \
  -e LMCACHE_USE_EXPERIMENTAL=True \
  "${IMAGE}" \
    --host 0.0.0.0 \
    --port "${CONTAINER_PORT}" \
    --model "${MODEL}" \
    --served-model-name qwen35-397b \
    --hf-overrides "${HF_OVERRIDES}" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-prefix-caching \
    --trust-remote-code \
    --max_num_seqs 32 \
    --max-num-batched-tokens 4096 \
    --max-model-len "${MODEL_LEN}" \
    --speculative-config "${SPECULATIVE_CONFIG}" \
    --enable-prompt-tokens-details \
    --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
    --chat-template /configs/chat_template_qwen35.jinja \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
