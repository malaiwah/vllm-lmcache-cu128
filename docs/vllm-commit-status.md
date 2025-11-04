# vLLM commit pin status

The container image pins vLLM to commit `3758757377b713b6acc997d0ac2c5dd49c332278` as defined in the [`Containerfile`](../Containerfile).

## Latest verification (2025-11-03)

- Upstream `main` HEAD: `3758757377b713b6acc997d0ac2c5dd49c332278`
- Pinned commit before this check: `31a4b3e6c40278025664169eafbc8165e1d0c393`
- CUDA devel base image digest (linux/amd64): `sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c`
- CUDA runtime base image digest (linux/amd64): `sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9`
- Action taken: updated `VLLM_COMMIT` to the upstream `main` head; CUDA base image digests already match registry values.

Command output:

```bash
$ git ls-remote https://github.com/vllm-project/vllm.git main
3758757377b713b6acc997d0ac2c5dd49c332278        refs/heads/main
```

```bash
$ python tools/inspect_cuda_manifest.py
12.8.1-cudnn-devel-ubuntu24.04 sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c
12.8.1-cudnn-runtime-ubuntu24.04 sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9
```

```bash
$ python tools/check_vllm_pin.py
Pinned commit   : 3758757377b713b6acc997d0ac2c5dd49c332278
Upstream     main: 3758757377b713b6acc997d0ac2c5dd49c332278
Status          : up to date
```

## Historical notes

- 2025-10-08:
  - Updated `VLLM_COMMIT` pin to match upstream `main`.
  - Refreshed the CUDA 12.8.1 build and runtime base image digests to the latest linux/amd64 manifests.
  - Added `tools/inspect_cuda_manifest.py` to automate future digest inspections.
  - Removed the earlier note about a failed verification attempt to keep this log focused on successful checks.
- 2025-11-03:
  - Refreshed the vLLM pin to `3758757377b713b6acc997d0ac2c5dd49c332278` to match upstream `main`.
  - Verified CUDA 12.8.1 build/runtime image digests; no updates required.
  - Added `tools/check_vllm_pin.py` to automate future pin drift checks.
