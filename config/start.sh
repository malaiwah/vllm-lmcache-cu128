#!/bin/bash
set -e

# Rotate logs on startup
chmod 0644 /etc/logrotate.d/nginx || true
logrotate --force /etc/logrotate.d/nginx || true

# All arguments passed to container are vLLM arguments
VLLM_ARGS="$*"

# Inject vLLM command into supervisord config
sed -i "s|command=/opt/venv/bin/python -m vllm.entrypoints.openai.api_server|command=/opt/venv/bin/python -m vllm.entrypoints.openai.api_server ${VLLM_ARGS}|" /etc/supervisor/supervisord.conf

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
