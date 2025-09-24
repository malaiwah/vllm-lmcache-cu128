#!/bin/bash
HF_TOKEN=
MODEL=Qwen/Qwen3-4B-AWQ
MODEL_LEN=32768

podman run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --security-opt label=disable \
  -e HF_TOKEN="$HF_TOKEN" \
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=8 \
  -e LMCACHE_CHUNK_SIZE=256 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LMCACHE_CONFIG_FILE=/srv/lmcache.yaml \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:Z \
  -v $PWD/lmcache.yaml:/srv/lmcache.yaml:Z \
  --entrypoint /opt/venv/bin/python \
  malaiwah/vllm-lmcache-cu128:uv312 \
  -c "
from vllm import LLM
llm = LLM(model='${MODEL}', dtype='auto', gpu_memory_utilization=0.9, max_model_len=${MODEL_LEN}, kv_transfer_config={'kv_connector':'LMCacheConnectorV1','kv_role':'kv_both'})
output = llm.generate('Who are you? Describe yourself in about 100 words.')
print(output[0].outputs[0].text)
" > test_output.txt
#  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
