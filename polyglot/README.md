
# Polyglot Tools (Streaming) for LiteLLM Proxy

Normalize **Hermes** (`<tool_call>...</tool_call>`) and **Anthropic** (`<tool_use>...</tool_use>`) tool tags **during streaming** into standard **OpenAI `tool_calls` deltas**, so your client (e.g., Crush/charms) always receives a consistent tool-calling signal.

## Why

Some models (e.g., Qwen, Mixtral variants) emit different tag styles for tool calls. This proxy callback watches the streaming chunks, detects completed tool blocks, and **emits OpenAI-compatible `tool_calls`** while **stripping the raw tags** from visible content.

- Works with vLLM (OpenAI-compatible) behind LiteLLM
- Lets you keep mixed model templates without breaking your client’s tool flow

## Install

```bash
pip install litellm
```

## Files

- `polyglot_tools_stream_handler.py` – the LiteLLM callback
- `config.yaml` – your proxy config (example below)

## Configure LiteLLM Proxy

Create or edit `config.yaml`:

```yaml
model_list:
  - model_name: qwen-vllm
    litellm_params:
      model: openai/vllm
      api_base: http://127.0.0.1:8000/v1   # your vLLM OpenAI endpoint
      api_key: dummy

litellm_settings:
  # Register the callback by <python_module>.<instance_name>
  callbacks: polyglot_tools_stream_handler.proxy_handler_instance
```

Start the proxy:
```bash
litellm --config config.yaml --port 4000
```

Point your client (Crush/charms) at `http://localhost:4000/v1` using the **OpenAI Chat Completions** API.

## How it behaves (streaming)

- As tokens arrive, the handler buffers assistant `content` to detect **closed** tool blocks.
- When it sees one, it emits a synthetic **`tool_calls` delta** with the parsed `name` + full `arguments` JSON (emitted in one chunk for simplicity).
- The original XML-ish block is **removed** from the visible `content` stream.
- Non-tool chunks pass through unchanged.
- If no tool blocks are found, everything passes through unchanged.

> Note: OpenAI’s spec streams `arguments` character-by-character. This handler emits the full `arguments` once the tag closes—clients almost always accept this.

## Optional: Force/Encourage One Style

Even with this shim, you can reduce edge cases by nudging the model to use **Hermes `<tool_call>`** via your chat template and/or `tool_choice="required"` when appropriate.

## Troubleshooting

- If your client still sees raw `<tool_use>` tags, make sure the callback is actually loaded (typo-free `callbacks:` path) and you are calling the LiteLLM proxy (not vLLM directly).
- For very large `arguments`, your model may stream the tool JSON across multiple chunks **inside the same tag**. This handler waits for the **closing tag** before emitting a `tool_calls` delta, ensuring valid JSON.
- For non‑streaming requests, you can also add an `async_post_call_success_hook` that normalizes the final message—left out here since the focus is streaming.

## Security

This shim **parses** JSON inside tool tags. Do not execute anything until you’ve validated schema, types, and limits.

## References

- LiteLLM Proxy Call Hooks (streaming iterator hook): https://docs.litellm.ai/docs/proxy/call_hooks
- LiteLLM Custom Callback (registration): https://docs.litellm.ai/docs/proxy/streaming_logging
- LiteLLM Callbacks overview: https://docs.litellm.ai/docs/observability/callbacks

---

MIT License © 2025
