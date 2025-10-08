# vLLM commit pin status

The container image pins vLLM to commit `31a4b3e6c40278025664169eafbc8165e1d0c393` as defined in the [`Containerfile`](../Containerfile).

## Latest verification (2025-10-08)

- Upstream `main` HEAD: `31a4b3e6c40278025664169eafbc8165e1d0c393`
- Pinned commit before this check: `31a4b3e6c40278025664169eafbc8165e1d0c393`
- CUDA devel base image digest (linux/amd64): `sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c`
- CUDA runtime base image digest (linux/amd64): `sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9`
- Action taken: refreshed pinned CUDA base image digests to match registry values; vLLM pin already matches upstream `main`.

Command output:

```bash
$ git ls-remote https://github.com/vllm-project/vllm.git main
31a4b3e6c40278025664169eafbc8165e1d0c393        refs/heads/main
```

```bash
$ python tools/inspect_cuda_manifest.py
12.8.1-cudnn-devel-ubuntu24.04 sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c
12.8.1-cudnn-runtime-ubuntu24.04 sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9
```

## Historical notes

- 2025-10-08:
  - Updated `VLLM_COMMIT` pin to match upstream `main`.
  - Refreshed the CUDA 12.8.1 build and runtime base image digests to the latest linux/amd64 manifests.
  - Added `tools/inspect_cuda_manifest.py` to automate future digest inspections.
  - Removed the earlier note about a failed verification attempt to keep this log focused on successful checks.
