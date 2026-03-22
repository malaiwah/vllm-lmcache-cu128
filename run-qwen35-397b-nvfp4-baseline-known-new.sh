#!/usr/bin/env bash
set -euo pipefail

source .env

#IMAGE=docker.io/orthozany/vllm-qwen35-mtp:latest
IMAGE=localhost/vllm-cu130-hfov:pr37443-latest
LOCAL_PORT=8000
CONTAINER_PORT=8000
#MODEL=nvidia/Qwen3.5-397B-A17B-NVFP4
MODEL=lukealonso/Qwen3.5-397B-A17B-NVFP4
#MODEL_LEN=262144
MODEL_LEN=524288
CONTAINER_NAME=vllm-qwen35-397b
HF_HOME=/mnt/vault/llm/huggingface
CONFIGS_DIR=/mnt/vault/llm/vllm+lmcache/configs
GPU_UTIL=0.90

# VLLM_CUTLASS is supposed to be better for accuracy (per-expert scaling, instead of per-layer)
# (Worker pid=124) (Worker_TP0 pid=124) INFO 03-17 13:15:45 [nvfp4.py:290] Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN'].

HF_OVERRIDES="${HF_OVERRIDES:-{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":2.0,\"original_max_position_embeddings\":262144}}}}"

exec podman run --rm -it \
  --name "$CONTAINER_NAME" \
  --replace \
  --device nvidia.com/gpu=all \
  --ipc=host \
  -p ${LOCAL_PORT}:${CONTAINER_PORT} \
  -v "$HF_HOME":/root/.cache/huggingface:Z \
  -v "$CONFIGS_DIR":/configs:ro,Z \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e OMP_NUM_THREADS=4 \
  -e NCCL_IB_DISABLE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TUNED_CONFIG_FOLDER=/configs \
  "$IMAGE" \
    --host 0.0.0.0 \
    --port ${CONTAINER_PORT} \
    --model "$MODEL" \
    --served-model-name qwen35-397b \
    --hf-overrides "${HF_OVERRIDES}" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization ${GPU_UTIL} \
    --kv-cache-dtype auto \
    --enable-prefix-caching \
    --trust-remote-code \
    --max_num_seqs 32 \
    --max-num-batched-tokens 4096 \
    --max-model-len ${MODEL_LEN} \
    --enable-prompt-tokens-details \
    --chat-template /configs/chat_template_qwen35.jinja \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
    #--speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
