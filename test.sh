#!/bin/bash
HF_TOKEN=
MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
MODEL_LEN=8192

podman run --rm \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --security-opt label=disable \
  -e HF_TOKEN="$HF_TOKEN" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:Z \
  --entrypoint /opt/venv/bin/python \
  malaiwah/vllm-lmcache-cu128:uv312 \
  -c "
from vllm import LLM, SamplingParams
llm = LLM(model='${MODEL}', enforce_eager=True, dtype='auto', max_model_len=${MODEL_LEN}, kv_transfer_config={'kv_connector':'LMCacheConnectorV1','kv_role':'kv_both'})
sampling_params = SamplingParams(max_tokens=500)
output = llm.generate(['Who are you? Describe yourself in about 100 words.'], sampling_params=sampling_params)
print(output[0].outputs[0].text)
" > test_output.txt
