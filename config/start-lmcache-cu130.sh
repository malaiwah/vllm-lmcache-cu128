#!/usr/bin/env bash
set -euo pipefail

export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/config/lmcache.yaml}"
export LMCACHE_USE_EXPERIMENTAL="${LMCACHE_USE_EXPERIMENTAL:-True}"

exec python3 -m vllm.entrypoints.openai.api_server "$@"
