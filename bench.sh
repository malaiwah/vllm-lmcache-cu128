#!/usr/bin/env bash
set -euo pipefail

URL=${URL:-http://localhost:8000/v1/chat/completions}
MODEL=${MODEL:-local/vllm}
PROMPT_FILE=${PROMPT_FILE:-prompt.txt}
MAX_TOKENS=${MAX_TOKENS:-2048}

warmup() {
  curl -s "$URL" \
    -H 'Content-Type: application/json' \
    -d '{
      "model":"'"$MODEL"'",
      "temperature":0,
      "top_p":1,
      "max_tokens":'"$MAX_TOKENS"',
      "seed":1,
      "messages":[{"role":"user","content":"'"$(cat "$PROMPT_FILE")"'"}]
    }' >/dev/null
}

run_one() {
  local body=/tmp/vllm_resp.json
  # write JSON to file, print only timing to stdout
  local t
  t=$(
    curl -s "$URL" \
      -H 'Content-Type: application/json' \
      -w '%{time_total}' \
      -o "$body" \
      -d '{
        "model":"'"$MODEL"'",
        "temperature":0,
        "top_p":1,
        "max_tokens":'"$MAX_TOKENS"',
        "seed":1,
        "messages":[{"role":"user","content":"'"$(cat "$PROMPT_FILE")"'"}]
      }'
  )
  local comp
  comp=$(jq -r '.usage.completion_tokens' "$body")
  python3 - <<PY
t=float("$t"); comp=int("$comp")
print(f"completion_tokens={comp}, time_s={t:.3f}, TPS={comp/max(t,1e-9):.2f}")
PY
}

warmup
run_one
