#!/bin/bash
set -e

# Rotate logs on startup
chmod 0644 /etc/logrotate.d/nginx || true
logrotate --force /etc/logrotate.d/nginx || true

# Build the launch script that supervisord will invoke so we can preserve
# the original argument boundaries, even for JSON payloads.
VLLM_LAUNCHER=/run/vllm/vllm-launch.sh
export VLLM_LAUNCHER
mkdir -p "$(dirname "${VLLM_LAUNCHER}")"

VLLM_CMD=(/opt/venv/bin/python -m vllm.entrypoints.openai.api_server --uds /run/vllm/vllm.sock "$@")

{
  echo "#!/bin/bash"
  echo "set -e"
  printf 'exec %s\n' "$(printf '%q ' "${VLLM_CMD[@]}")"
} > "${VLLM_LAUNCHER}"

chmod 0755 "${VLLM_LAUNCHER}"

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
