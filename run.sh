#!/bin/bash
HF_TOKEN=
MODEL=Qwen/Qwen3-4B-AWQ
MODEL_LEN=32768

podman run --rm -it \
  --device nvidia.com/gpu=all \
  --ipc=host --network host \
  --security-opt label=disable \
  -e HF_TOKEN="$HF_TOKEN" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LMCACHE_CONFIG_FILE=/srv/lmcache.yaml \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:Z \
  -v $PWD/lmcache.yaml:/srv/lmcache.yaml:Z \
  malaiwah/vllm-lmcache-cu128:uv312 \
  --model "${MODEL}" \
  --gpu-memory-utilization 0.90 \
  --dtype auto \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len ${MODEL_LEN}
#  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
