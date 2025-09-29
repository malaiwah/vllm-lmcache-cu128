#!/bin/bash
source .env
# uv venv .. uv pip install huggingface_hub[cli]
source .venv/bin/activate

IMAGE=docker.io/malaiwah/vllm-lmcache-cu128:uv312

# For local testing
#IMAGE=vllm-lmcache-cu128:test

LOCAL_PORT=8000

MODEL=Qwen/Qwen3-4B-Instruct-2507
#MODEL_LEN=8192

#MODEL=mistralai/Mistral-7B-Instruct-v0.3
#MODEL_LEN=2896

#Too big for 16GB
#MODEL=unsloth/mistral-7b-instruct-v0.3-bnb-4bit

MODEL=RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit
MODEL=thesven/Mistral-7B-Instruct-v0.3-GPTQ
#18992 maximum context
MODEL=microsoft/Phi-3-mini-128k-instruct
#MODEL_LEN=18000
#
#MODEL=microsoft/Phi-4-mini-reasoning
#MODEL_LEN=32000

MODEL=microsoft/Phi-4-mini-reasoning
MODEL=microsoft/Phi-4-multimodal-instruct
#MODEL_LEN=36784

#bnb library .. done .. but too large somehow
#MODEL=unsloth/Phi-3.5-mini-instruct-bnb-4bit

#32656. (bumped maximum mem to 94% but no go)
#MODEL=solidrust/Phi-3-mini-128k-instruct-AWQ
#MODEL_LEN=32000

#Input Type(s): Text, image and speech
#needs trust_remote_code (ok, official from nvidia)
#it does support 131072!
#fp4=72192
#fp8=61360
#(EngineCore_DP0 pid=189)    19 | #include <cublasLt.h>
# let's make the kv cache fp8 as well
# --kv_cache_dtype fp8 --> max_model_len 122896
#fp4 just loops output
#MODEL=nvidia/Phi-4-multimodal-instruct-FP4
MODEL=nvidia/Phi-4-multimodal-instruct-FP8
#MODEL_LEN=122896

#bleeding edge and hard to make work with library
#(APIServer pid=1) ImportError: Please install torchao>=0.10.0 via `pip install torchao>=0.10.0` to use torchao quantization.
#MODEL=pytorch/Phi-4-mini-instruct-AWQ-INT4

#also possible .. 34B maybe (no) (13B is context limited to 16k, only 9504 fits. 16k would probably fit in a 32gb VRAM card) -- no fit
#MODEL=TheBloke/CodeLlama-13B-Instruct-AWQ
#MODEL_LEN=9504

#too big, but added --cpu-offload-gb 32 and still didn't work
#MODEL=stelterlab/NextCoder-32B-AWQ

## GGUF is highly experimental -- must download separately and does not support split files
#wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
#vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
#   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
#
#mv NextCoder-14B-Q5_K_M.gguf\?download\=true NextCoder-14B-Q5_K_M.gguf
#vllm serve ./NextCoder-14B-Q5_K_M.gguf --tokenizer ${MODEL_TOKENIZER}
#MODEL_TOKENIZER=microsoft/NextCoder-14B
#MODEL_LEN=18768

#too large
#MODEL=cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit

#(APIServer pid=1) AttributeError: 'MistralCommonTokenizer' object has no attribute 'all_special_ids'. Did you mean: '_all_special_ids'?
#MODEL=cpatonn/Devstral-Small-2507-AWQ-4bit
#(APIServer pid=1) AttributeError: 'MistralCommonTokenizer' object has no attribute 'all_special_ids'. Did you mean: '_all_special_ids'?
#MODEL=cpatonn/Magistral-Small-2507-AWQ-4bit
MODEL=cpatonn/Qwen3-4B-Instruct-2507-AWQ-8bit
#MODEL_LEN=65536
#yarn/rope
#MODEL_LEN=131072

# Too large
#MODEL=cpatonn/GLM-4.5-Air-AWQ-4bit
#MODEL=cpatonn/GLM-4.5-Air-GPTQ-4bit
# no way
#MODEL=zai-org/GLM-4.5-Air-FP8

# openwebui does not collapse the <think/>
#MODEL=cpatonn/Qwen3-4B-Thinking-2507-AWQ-8bit

# Snappy and 128k on 5090
MODEL="zai-org/glm-4-9b-chat"

# works 20tk/s but 2tk/s with cpu offload .. limited context
MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
#MODEL_LEN=24384

# works but does not obey to crush tool calls
MODEL="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
MODEL_LEN=131072

#MODEL=Qwen/Qwen3-4B-Instruct-2507

#(EngineCore_DP0 pid=197) AttributeError: MoE Model GptOssForCausalLM does not support BitsAndBytes quantization yet. Ensure this model has 'get_expert_mapping' method.
#MODEL=unsloth/gpt-oss-20b-unsloth-bnb-4bit
MODEL=openai/gpt-oss-20b

#podman pull ${IMAGE} && \
podman run --rm -it \
  --device nvidia.com/gpu=all \
  --name vllm \
  --ipc=host \
  -p ${LOCAL_PORT}:${LOCAL_PORT} \
  --security-opt label=disable \
  -e HF_TOKEN="$HF_TOKEN" \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128" \
  -e LMCACHE_CONFIG_FILE=/srv/lmcache.yaml \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:Z \
  -v $HOME/.cache/triton:/root/.cache/triton \
  -v $HOME/.cache/torch/inductor:/root/.cache/torch/inductor \
  -v $HOME/.cache/flashinfer:/root/.cache/flashinfer \
  -v $PWD/lmcache.yaml:/srv/lmcache.yaml:Z \
  -v $PWD/vllm-logs/supervisor:/var/log/supervisor:Z \
  -v $PWD/vllm-logs/nginx:/var/log/nginx:Z \
  ${IMAGE} \
  --model "${MODEL}" \
  --served-model-name "vllm" \
  --dtype auto \
  --enable-auto-tool-choice --tool-call-parser openai \
  --trust-remote-code \
  --kv_cache_dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max_num_seqs 4 \
  --async-scheduling \
  --cuda-graph-sizes 2048 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --compilation-config '{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
#  END
#  --enable-auto-tool-choice --tool-call-parser hermes \
#  --max-model-len ${MODEL_LEN} \
#  --rope-scaling '{"type":"dynamic","factor":4.0,"original_max_position_embeddings":32768}' \
#  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
#  --sliding-window 16384 \
#  --cpu-offload-gb 8 \
#  --kv_cache_dtype fp8 \
#  --enforce-eager True \
#  --no-enable_prefix_caching \
#  --quantization awq \
#  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_buffer_device":"cpu","kv_buffer_size":0}' \
#  --network host
