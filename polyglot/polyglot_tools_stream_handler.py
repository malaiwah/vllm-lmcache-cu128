
# polyglot_tools_stream_handler.py
"""
LiteLLM Proxy callback that NORMALIZES tool calls during **streaming**.

What it does
------------
- Watches streaming chunks for either <tool_call>{...}</tool_call> (Hermes) OR
  <tool_use>{...}</tool_use> (Anthropic) blocks.
- When it sees a *complete* block, it emits an OpenAI-compatible `tool_calls` delta
  so upstream clients (e.g., Crush/charms) receive standard tool calls.
- It also strips the raw XML-ish tags from the visible `content` stream.

How to use
----------
1) Put this file somewhere accessible by the LiteLLM proxy process.
2) In your proxy config (config.yaml), register this as a callback:
      litellm_settings:
        callbacks: polyglot_tools_stream_handler.proxy_handler_instance
3) Start the proxy:
      litellm --config config.yaml
"""

import re
import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponseStream

# Robust-ish regex for XML-ish tool blocks.
HERMES_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANTHRO_RE = re.compile(r"<tool_use>\s*(\{.*?\})\s*</tool_use>", re.DOTALL)

def _extract_calls_from_text(text: str) -> List[Dict[str, str]]:
    """
    Try Hermes first; if none found, try Anthropic-style.
    Return list of {"name": str, "arguments": str(JSON)}
    """
    calls: List[Dict[str, str]] = []
    for m in HERMES_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            calls.append({"name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}))})
        except Exception:
            pass
    if calls:
        return calls
    for m in ANTHRO_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            calls.append({"name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}))})
        except Exception:
            pass
    return calls

def _strip_tool_blocks(text: str) -> str:
    text = HERMES_RE.sub("", text)
    text = ANTHRO_RE.sub("", text)
    return text

def _parse_sse_data_line(line: str) -> Optional[Dict[str, Any]]:
    """If `line` looks like 'data: {...}', return the parsed JSON dict."""
    if not isinstance(line, str):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return {"[DONE]": True}
    try:
        return json.loads(payload)
    except Exception:
        return None

def _pack_sse(obj: Dict[str, Any]) -> ModelResponseStream:
    """
    Pack an object into a ModelResponseStream-like item.
    We let LiteLLM handle type wrapping; yielding a dict works in practice.
    """
    return ModelResponseStream(data=json.dumps(obj))

def _mk_tool_call_delta(call_index: int, name: str, arguments: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Build an OpenAI Chat Completions streaming delta for tool_calls.
    According to OpenAI spec, the `arguments` field is streamed as incremental strings.
    For simplicity, we emit it all at once when the tag closes.
    """
    delta = {
        "id": None,
        "object": "chat.completion.chunk",
        "created": None,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": call_index,
                            "id": f"call_{call_index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments
                            }
                        }
                    ]
                },
                "finish_reason": None
            }
        ]
    }
    return delta

def _mk_content_delta(text_piece: str, model: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": None,
        "object": "chat.completion.chunk",
        "created": None,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": text_piece},
                "finish_reason": None
            }
        ]
    }

class PolyglotToolsStreamingHandler(CustomLogger):
    """
    Streaming-aware hook: normalizes <tool_call>/<tool_use> into OpenAI tool_calls.
    """
    def __init__(self):
        super().__init__()

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Wrap the upstream async iterator.

        Strategy:
          - Buffer assistant `content` text until we detect a CLOSED tool tag.
          - When a tag closes, parse the inner JSON and emit a synthetic `tool_calls` delta.
          - Strip the tag text from the visible content.
          - Pass through all non-tool content unchanged.
        """
        buffer_text = ""          # accumulate content text (for tag detection)
        pending_plain_flush = ""  # content to flush after stripping tags
        model_name = None
        tool_calls_emitted: List[Dict[str, str]] = []

        async for item in response:
            # Item may be a ModelResponseStream with .data, or already a dict.
            raw = None
            if isinstance(item, ModelResponseStream):
                raw = item.data
            elif isinstance(item, (bytes, str)):
                raw = item
            else:
                # Try to JSON-serialize+parse; if fails, just yield
                try:
                    raw = item
                except Exception:
                    yield item
                    continue

            # Try parse as SSE "data: {...}"
            parsed = None
            if isinstance(raw, (bytes, str)):
                line = raw.decode() if isinstance(raw, bytes) else raw
                parsed = _parse_sse_data_line(line)
                if parsed is None:
                    # Non-JSON chunk (e.g., SSE keepalive). Pass through.
                    yield item
                    continue
                if "[DONE]" in parsed:
                    # Flush any remaining plain text (without tool tags)
                    if pending_plain_flush:
                        yield _pack_sse(_mk_content_delta(pending_plain_flush, model_name))
                        pending_plain_flush = ""
                    yield item  # forward the original [DONE]
                    continue
            else:
                # Already a dict-like JSON chunk
                parsed = raw if isinstance(raw, dict) else None
                if parsed is None:
                    yield item
                    continue

            # Pull out text content deltas (if present)
            try:
                if model_name is None:
                    model_name = parsed.get("model")
                choices = parsed.get("choices") or []
                if not choices:
                    yield item  # pass through unrecognized chunk
                    continue
                delta = choices[0].get("delta") or {}
                delta_text = delta.get("content")
                if delta_text:
                    buffer_text += delta_text
                    # Detect completed tool blocks
                    calls = _extract_calls_from_text(buffer_text)
                    if calls:
                        # Remove the blocks from visible text
                        visible = _strip_tool_blocks(buffer_text)
                        # Determine the new visible segment to flush
                        to_flush = visible[len(pending_plain_flush):]
                        if to_flush:
                            yield _pack_sse(_mk_content_delta(to_flush, model_name))
                            pending_plain_flush += to_flush

                        # Emit tool_calls delta(s)
                        for idx, c in enumerate(calls):
                            # Avoid duplicating on repeated detection (idempotency)
                            if c in tool_calls_emitted:
                                continue
                            tool_calls_emitted.append(c)
                            yield _pack_sse(_mk_tool_call_delta(
                                call_index=len(tool_calls_emitted)-1,
                                name=c["name"],
                                arguments=c["arguments"],
                                model=model_name
                            ))
                        # We've consumed the entire buffer into visible text + tool_calls;
                        # keep buffer_text equal to visible so future diffs are correct.
                        buffer_text = visible
                    # Do NOT forward the original item (we already emitted normalized deltas)
                    continue
                else:
                    # Forward chunks with no 'content' delta (roles, etc.)
                    yield item
                    continue
            except Exception:
                # On any parsing error, just pass through untouched
                yield item
                continue

# The instance that LiteLLM expects to import via config.yaml
proxy_handler_instance = PolyglotToolsStreamingHandler()
