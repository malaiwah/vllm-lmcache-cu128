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

import json
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

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


def _mk_tool_call_delta(call_index: int, name: str, arguments: str) -> Dict[str, Any]:
    """
    Build an OpenAI Chat Completions streaming delta for tool_calls.
    According to OpenAI spec, the `arguments` field is streamed as incremental strings.
    For simplicity, we emit it all at once when the tag closes.
    """
    return {
        "tool_calls": [
            {
                "index": call_index,
                "id": f"call_{call_index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ]
    }


def _mk_content_delta(text_piece: str) -> Dict[str, Any]:
    return {"content": text_piece}


def _make_stream_chunk(
    template: Optional[Dict[str, Any]],
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a chunk dictionary shaped like ModelResponseStream/ChatCompletion chunk.
    Falls back to synthetic metadata if no template is available.
    """
    base = template or {}
    choices = base.get("choices") or []
    first_choice = choices[0] if choices else {}

    chunk = {
        "id": base.get("id") or f"polyglot-tools-handler-{uuid4().hex}",
        "object": base.get("object") or "chat.completion.chunk",
        "created": base.get("created") or int(time.time()),
        "model": base.get("model"),
        "system_fingerprint": base.get("system_fingerprint"),
        "choices": [
            {
                "index": first_choice.get("index", 0),
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": first_choice.get("logprobs"),
            }
        ],
        "provider_specific_fields": base.get("provider_specific_fields"),
    }
    return chunk


def _pack_stream(chunk: Dict[str, Any], kind: Optional[str]) -> Any:
    """
    Serialize a chunk according to the upstream stream type we intercepted.
    """
    if kind == "model":
        return ModelResponseStream(**chunk)
    if kind == "sse":
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    return chunk


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
        buffer_text = ""
        emitted_plain_text = ""
        tool_calls_emitted: List[Dict[str, str]] = []
        last_template: Optional[Dict[str, Any]] = None
        last_kind: Optional[str] = None

        async for item in response:
            parsed: Optional[Dict[str, Any]] = None
            kind: Optional[str] = None

            if isinstance(item, ModelResponseStream):
                parsed = item.model_dump()
                kind = "model"
            elif isinstance(item, (bytes, str)):
                line = item.decode() if isinstance(item, bytes) else item
                parsed = _parse_sse_data_line(line)
                if parsed is None:
                    yield item
                    continue
                if "[DONE]" in parsed:
                    visible = _strip_tool_blocks(buffer_text)
                    remaining = visible[len(emitted_plain_text):]
                    if remaining:
                        chunk_dict = _make_stream_chunk(last_template, _mk_content_delta(remaining))
                        yield _pack_stream(chunk_dict, last_kind or kind or "sse")
                        emitted_plain_text += remaining
                    buffer_text = ""
                    emitted_plain_text = ""
                    tool_calls_emitted.clear()
                    last_template = None
                    last_kind = None
                    yield item
                    continue
                kind = "sse"
            elif isinstance(item, dict):
                parsed = item
                kind = "dict"
            else:
                yield item
                continue

            if not isinstance(parsed, dict):
                yield item
                continue

            if parsed.get("choices"):
                last_template = parsed
                last_kind = kind or last_kind

            choices = parsed.get("choices") or []
            if not choices:
                yield item
                continue

            delta = choices[0].get("delta") or {}
            delta_text = delta.get("content")

            if delta_text:
                buffer_text += delta_text
                visible = _strip_tool_blocks(buffer_text)
                new_plain = visible[len(emitted_plain_text):]
                if new_plain:
                    chunk_dict = _make_stream_chunk(parsed, _mk_content_delta(new_plain))
                    yield _pack_stream(chunk_dict, kind or last_kind or "dict")
                    emitted_plain_text += new_plain

                calls = _extract_calls_from_text(buffer_text)
                for c in calls:
                    if c in tool_calls_emitted:
                        continue
                    tool_calls_emitted.append(c)
                    call_delta = _mk_tool_call_delta(len(tool_calls_emitted) - 1, c["name"], c["arguments"])
                    chunk_dict = _make_stream_chunk(parsed, call_delta)
                    yield _pack_stream(chunk_dict, kind or last_kind or "dict")

                buffer_text = visible
                continue

            # No content text; pass through original item unchanged.
            yield item


# The instance that LiteLLM expects to import via config.yaml
proxy_handler_instance = PolyglotToolsStreamingHandler()
