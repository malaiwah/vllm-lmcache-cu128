#!python

#mbelleau@aiboss:~/vllm+lmcache$ podman ps
#CONTAINER ID  IMAGE                                        COMMAND               CREATED        STATUS        PORTS                   NAMES
#d4848eadad1c  docker.io/malaiwah/vllm-lmcache-cu128:uv312  --model Qwen/Qwen...  7 seconds ago  Up 7 seconds  0.0.0.0:8000->8000/tcp  laughing_dhawan
#mbelleau@aiboss:~/vllm+lmcache$ podman exec -it laughing_dhawan /bin/bash --login
#root@d4848eadad1c:/srv# python - <<'PY'
#import torch, json
#assert torch.cuda.is_available(), "CUDA not available"
#gpus=[]
#for i in range(torch.cuda.device_count()):
#    name = torch.cuda.get_device_name(i)
#    cc   = torch.cuda.get_device_capability(i)    # (major, minor)
#    gpus.append({"index": i, "name": name, "cc": f"{cc[0]}.{cc[1]}", "tuple": cc})
#print(json.dumps(gpus, indent=2))
#PY
#/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
#  import pynvml  # type: ignore[import]
#[
#  {
#    "index": 0,
#    "name": "NVIDIA GeForce RTX 5090",
#    "cc": "12.0",
#    "tuple": [
#      12,
#      0
#    ]
#  }
#]

import torch, json
assert torch.cuda.is_available(), "CUDA not available"
gpus=[]
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cc   = torch.cuda.get_device_capability(i)    # (major, minor)
    gpus.append({"index": i, "name": name, "cc": f"{cc[0]}.{cc[1]}", "tuple": cc})
print(json.dumps(gpus, indent=2))
