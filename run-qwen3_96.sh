#!/bin/bash
source .env
# uv venv .. uv pip install huggingface_hub[cli]
source .venv/bin/activate

IMAGE=docker.io/malaiwah/vllm-lmcache-cu128:uv312
#IMAGE=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:latest

# For local testing
#IMAGE=vllm-lmcache-cu128:test

LOCAL_PORT=8000
CONTAINER_PORT=8000
GPU_UTIL=0.92
MODEL_LEN=131072

# Works on 16GB
#MODEL=Qwen/Qwen3-4B-Instruct-2507
#MODEL_LEN=65536

# Can't get it to work
#MODEL=mistralai/Mistral-7B-Instruct-v0.3
#MODEL_LEN=6144
#GPU_UTIL=0.94

# Works on 16GB
#MODEL=unsloth/mistral-7b-instruct-v0.3-bnb-4bit
#MODEL_LEN=65536

#MODEL=RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit
#MODEL=thesven/Mistral-7B-Instruct-v0.3-GPTQ

# Works on 16GB
#MODEL=microsoft/Phi-3-mini-128k-instruct
#MODEL_LEN=18000

#MODEL=microsoft/Phi-4-mini-reasoning
#MODEL=microsoft/Phi-4-multimodal-instruct
#MODEL_LEN=36784

# Works on 16GB
#MODEL=unsloth/Phi-3.5-mini-instruct-bnb-4bit
#MODEL_LEN=32768

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
#MODEL_LEN=32768
#MODEL=nvidia/Phi-4-multimodal-instruct-FP8
#MODEL_LEN=122896

## GGUF is highly experimental -- must download separately and does not support split files
#wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
#vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
#   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
#
#mv NextCoder-14B-Q5_K_M.gguf\?download\=true NextCoder-14B-Q5_K_M.gguf
#vllm serve ./NextCoder-14B-Q5_K_M.gguf --tokenizer ${MODEL_TOKENIZER}
#MODEL_LEN=18768

#(APIServer pid=1) AttributeError: 'MistralCommonTokenizer' object has no attribute 'all_special_ids'. Did you mean: '_all_special_ids'?
#MODEL=cpatonn/Devstral-Small-2507-AWQ-4bit
#(APIServer pid=1) AttributeError: 'MistralCommonTokenizer' object has no attribute 'all_special_ids'. Did you mean: '_all_special_ids'?
#MODEL=cpatonn/Magistral-Small-2507-AWQ-4bit
#MODEL=cpatonn/Qwen3-4B-Instruct-2507-AWQ-8bit
#MODEL_LEN=65536
#yarn/rope
#MODEL_LEN=131072

# openwebui does not collapse the <think/>
#MODEL=cpatonn/Qwen3-4B-Thinking-2507-AWQ-8bit

# Snappy and 128k on 5090
#MODEL="zai-org/glm-4-9b-chat"

# works 20tk/s but 2tk/s with cpu offload .. limited context
#MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
#MODEL_LEN=24384

# works but does not obey to crush tool calls
#MODEL="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
#MODEL_LEN=131072

#(EngineCore_DP0 pid=197) AttributeError: MoE Model GptOssForCausalLM does not support BitsAndBytes quantization yet. Ensure this model has 'get_expert_mapping' method.
#MODEL=unsloth/gpt-oss-20b-unsloth-bnb-4bit

# Very compliant (jan.ai), 85% max
#MODEL=janhq/Jan-v1-4B

#cutoff Sept 2021, no tools calling
#MODEL=huggingFaceH4/zephyr-7b-beta

#was: too large, but works at 85% max
#(EngineCore_DP0 pid=202) ERROR 09-30 00:23:07 [core.py:708] ValueError: To serve at least one request with the models's max seq len (262144), (12.00 GiB KV cachasing `max_model_len` when initializing the engine.
#MODEL=cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit
#MODEL_LEN=110000

#bleeding edge and hard to make work with library
#(APIServer pid=1) ImportError: Please install torchao>=0.10.0 via `pip install torchao>=0.10.0` to use torchao quantization.
#MODEL=pytorch/Phi-4-mini-instruct-AWQ-INT4

# cutoff sept 2021
#MODEL=microsoft/Phi-4-multimodal-instruct
#MODEL_LEN=32000

#still: too big, but added --cpu-offload-gb 32 and still didn't work
#MODEL=stelterlab/NextCoder-32B-AWQ
#MODEL_LEN=24000

#MODEL=Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ
#MODEL_LEN=32768
#NEXT: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
#maybe qwen2.5-coder:32b for coding, and qwen2.5:72b
#
#(APIServer pid=45) AttributeError: 'MistralCommonTokenizer' object has no attribute 'all_special_ids'. Did you mean: '_all_special_ids'?
#MODEL=RedHatAI/Devstral-Small-2507-FP8-Dynamic

#MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#MODEL=unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit
MODEL_LEN=65536

# MUST FIX, TOOL CALLING
#MODEL=bartowski/Qwen2.5-Coder-32B-Instruct-exl2
#MODEL=/model/models--bartowski--Qwen2.5-Coder-32B-Instruct-exl2/snapshots/d212ef3e545022aacb8cedb54197624b3eb7ebcc
#MODEL=/model/gguf/NextCoder-14B-Q8_0.gguf
#MODEL_TOKENIZER=microsoft/NextCoder-14B
# MAX 32768
#MODEL=/model/gguf/microsoft_NextCoder-32B-Q4_K_M.gguf
#MODEL_TOKENIZER=microsoft/NextCoder-32B
#MODEL_LEN=32768
#tool parser openai, 85% max
#  --compilation-config '{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
#MODEL=openai/gpt-oss-20b

#(EngineCore_DP0 pid=198) ValueError: Attempted to use an uninitialized parameter in vllm._fused_mul_mat_gguf. This error happens when you are using a `LazyModule` or explicitly manipulating `torch.nn.parameter.GGUFUninitializedParameter` objects.
#MODEL=/model/gguf/Mistral-2x24B-MOE-Magistral-2506-Devstral-2507-1.1-Coder-Reasoning-Ultimate-44B.Q3_K_M.gguf
#MODEL_TOKENIZER=DavidAU/Mistral-2x24B-MOE-Magistral-2506-Devstral-2507-1.1-Coder-Reasoning-Ultimate-44B

#(APIServer pid=46) ERROR 09-30 20:57:32 [serving_chat.py:1133] NotImplementedError: Not being used, manual parsing in serving_chat.py
#MODEL=/model/gguf/Huihui-K2-Think-abliterated.Q4_K_M.gguf
#MODEL_TOKENIZER=huihui-ai/Huihui-K2-Think-abliterated
#MODEL_LEN=2048

# still is: was Too large on RTX5090 but is correct on RTX6000 Pro
#MODEL=cpatonn/GLM-4.5-Air-AWQ-4bit
#MODEL=cpatonn/GLM-4.5-Air-AWQ-8bit
#MODEL=cpatonn/GLM-4.5-Air-GPTQ-4bit
# no way
#MODEL=zai-org/GLM-4.5-Air-FP8

#MODEL=openai/gpt-oss-120b

#MODEL=cpatonn/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
#MODEL_LEN=262144

#MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
#MODEL_LEN=131072
#GPU_UTIL=0.92

MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8
MODEL_LEN=262144

#podman pull ${IMAGE} && \
podman run --rm -it \
  --device nvidia.com/gpu=all \
--name vllm \
--ipc=host \
-p ${LOCAL_PORT}:${CONTAINER_PORT} \
--security-opt label=disable \
-e HF_TOKEN="$HF_TOKEN" \
-e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128" \
-e LMCACHE_CONFIG_FILE=/srv/lmcache.yaml \
-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
-e LMCACHE_MAX_DISK_GB=64 \
-e LMCACHE_TTL_HOURS=24 \
-v $HOME/.cache/huggingface:/root/.cache/huggingface:Z \
-v $HOME/.cache/triton:/root/.cache/triton \
-v $HOME/.cache/lmcache:/root/.cache/lmcache \
-v $HOME/.cache/torch/inductor:/root/.cache/torch/inductor \
-v $HOME/.cache/flashinfer:/root/.cache/flashinfer \
-v $PWD/lmcache.yaml:/srv/lmcache.yaml:Z \
-v $HOME/.cache/vllm:/root/.cache/vllm \
-v $PWD/vllm-logs/supervisor:/var/log/supervisor:Z \
-v $PWD/vllm-logs/nginx:/var/log/nginx:Z \
-v /home/mbelleau/.cache/huggingface/hub:/model:Z \
${IMAGE} \
--model "${MODEL}" \
--max-model-len ${MODEL_LEN} \
--served-model-name "vllm" \
--trust-remote-code \
--gpu-memory-utilization ${GPU_UTIL} \
--max_num_seqs 4 \
--max-num-batched-tokens 16384 \
--enable-auto-tool-choice --tool-call-parser hermes \
--kv_cache_dtype fp8 \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","limit_tokens":50000}' \
--rope-scaling '{"type":"dynamic","factor":4.0}' \
#  END
#--enable-auto-tool-choice --tool-call-parser openai \
#--cuda-graph-sizes 2048 \
#--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
#--no-enable-chunked-prefill \
#--rope-scaling '{"type":"dynamic","factor":4.0,"original_max_position_embeddings":32768}' \
#  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' \
#  --tokenizer ${MODEL_TOKENIZER} \
#  --dtype auto \
#  --async-scheduling \
#  --reasoning-parser openai_gptoss \
#  --enable-auto-tool-choice --tool-call-parser glm45 \
#  --enforce-eager \
#  --compilation-config '{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
#  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
#  --sliding-window 16384 \
#  --cpu-offload-gb 8 \
#  --no-enable_prefix_caching \
#  --quantization awq \
#  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_buffer_device":"cpu","kv_buffer_size":0}' \
