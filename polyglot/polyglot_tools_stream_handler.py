
# polyglot_tools_stream_handler.py
"""
LiteLLM Proxy callbacks that NORMALIZE tool calls for both **streaming** and **non-streaming**.
This file exposes two callback instances:
  - proxy_handler_instance           -> streaming normalization (async_post_call_streaming_iterator_hook)
  - nonstream_handler_instance       -> non-streaming normalization (async_post_call_hook)

Add both to your LiteLLM config:
  litellm_settings:
    callbacks:
      - polyglot_tools_stream_handler.proxy_handler_instance
      - polyglot_tools_stream_handler.nonstream_handler_instance
"""

import json
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import uuid4

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponseStream

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Tag detection (outer blocks). We capture the inner payload of the tag.
HERMES_BLOCK = re.compile(r"<\s*tool_call\s*>(.*?)</\s*tool_call\s*>", re.DOTALL | re.IGNORECASE)
ANTHRO_BLOCK = re.compile(r"<\s*tool_use\s*>(.*?)</\s*tool_use\s*>", re.DOTALL | re.IGNORECASE)

NAME_TAG = re.compile(r"<\s*(?:name|tool_name)\s*>(.*?)</\s*(?:name|tool_name)\s*>", re.DOTALL | re.IGNORECASE)
ARGS_TAG = re.compile(r"<\s*arguments\s*>(.*?)</\s*arguments\s*>", re.DOTALL | re.IGNORECASE)

# Max buffer for assistant text before we flush visible content (prevents DoS/memory bloat)
MAX_BUFFER_BYTES = 256 * 1024  # 256 KiB


def _strip_code_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        # remove optional language after opening fence
        t = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t


def _json_fixer_mild(s: str) -> str:
    """Conservative fixes to coerce quasi-JSON into JSON."""
    t = _strip_code_fences(s)
    # Remove trailing commas before closing } or ]
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # Replace unescaped single quotes with double quotes (heuristic)
    t = re.sub(r"(?<=[:\[,]\s*)'([^'\n\\]*?)'\s*(?=[:,}\]])", r'"\1"', t)  # values
    t = re.sub(r"'([A-Za-z_][A-Za-z0-9_]*)'\s*:", r'"\1":', t)             # keys
    return t


def _balanced_json_from(s: str, start_idx: int) -> Optional[Tuple[str, int]]:
    """
    Return (json_text, end_idx) for a JSON object starting at s[start_idx] == '{'.
    Handles quotes and escapes.
    """
    depth = 0
    i = start_idx
    in_str = False
    esc = False
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start_idx:i+1], i + 1
        i += 1
    return None


def _extract_json_after_tag(inner: str) -> Optional[str]:
    """Find the first '{' inside an <arguments>...</arguments> payload and return a balanced JSON slice."""
    m = ARGS_TAG.search(inner)
    if not m:
        return None
    payload = m.group(1)
    brace = payload.find("{")
    if brace < 0:
        return None
    got = _balanced_json_from(payload, brace)
    if not got:
        return None
    json_text, _ = got
    return json_text


def _parse_hermes_or_anthro_blocks(text: str) -> List[Dict[str, str]]:
    """
    Return list of {"name": str, "arguments": str(JSON)} extracted from XML-ish blocks.
    """
    calls: List[Dict[str, str]] = []
    for tag_re in (HERMES_BLOCK, ANTHRO_BLOCK):
        for m in tag_re.finditer(text):
            inner = m.group(1)
            nm = NAME_TAG.search(inner)
            args_text = _extract_json_after_tag(inner)
            if not nm or not args_text:
                continue
            name = nm.group(1).strip()
            # Try strict parse; if it fails, try fixer; if still fails, wrap raw
            parsed = None
            for attempt in (args_text, _json_fixer_mild(args_text)):
                try:
                    parsed = json.loads(attempt)
                    args_text = json.dumps(parsed, ensure_ascii=False)
                    break
                except Exception:
                    continue
            if parsed is None:
                # Last resort: keep raw but still JSON-encode as a string field
                args_text = json.dumps({"_raw": _strip_code_fences(args_text)}, ensure_ascii=False)
            calls.append({"name": name, "arguments": args_text})
    return calls


def _extract_openai_toolcalls_fast(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Detect a top-level-ish {"tool_calls":[...]} JSON object in the stream text.
    Very heuristic; intended for streaming where content may include JSON blobs.
    """
    idx = text.find('"tool_calls"')
    if idx == -1:
        return None
    # Try to find the nearest '{' before and parse a minimal slice
    brace = text.rfind("{", 0, idx + 1)
    if brace == -1:
        return None
    # Find a plausible end '}' to close the object
    end = text.find("}", idx)
    if end == -1:
        return None
    # Expand outwards a bit
    slice_text = text[brace: end+1]
    # Mild fixes, then parse
    try:
        obj = json.loads(slice_text)
        tc = obj.get("tool_calls")
        if isinstance(tc, list):
            return tc
    except Exception:
        try:
            obj = json.loads(_json_fixer_mild(slice_text))
            tc = obj.get("tool_calls")
            if isinstance(tc, list):
                return tc
        except Exception:
            return None
    return None


def _strip_tool_blocks(text: str) -> str:
    text = HERMES_BLOCK.sub("", text)
    text = ANTHRO_BLOCK.sub("", text)
    return text


def _parse_sse_data_line(line: str) -> Optional[Dict[str, Any]]:
    """If `line` looks like 'data: {...}', return the parsed JSON dict."""
    if not isinstance(line, str) or not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return {"[DONE]": True}
    try:
        return json.loads(payload)
    except Exception:
        return None


def _mk_role_delta() -> Dict[str, Any]:
    return {"role": "assistant"}


def _mk_tool_name_delta(call_index: int, call_id: str, name: str) -> Dict[str, Any]:
    return {
        "tool_calls": [
            {
                "index": call_index,
                "id": call_id,
                "type": "function",
                "function": {"name": name}
            }
        ]
    }


def _mk_tool_args_delta(call_index: int, call_id: str, arguments_chunk: str) -> Dict[str, Any]:
    return {
        "tool_calls": [
            {
                "index": call_index,
                "id": call_id,
                "type": "function",
                "function": {"arguments": arguments_chunk}
            }
        ]
    }


def _mk_finish_tool_calls_delta() -> Dict[str, Any]:
    # Empty delta with finish_reason="tool_calls" will be set by _make_stream_chunk
    return {}


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
    Streaming-aware hook: normalizes <tool_call>/<tool_use> and native OpenAI tool_calls
    into OpenAI-style streaming deltas.
    """

    def __init__(self):
        super().__init__()

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:

        buffer_text = ""
        emitted_visible = ""
        tool_calls_seen: List[Tuple[str, str]] = []  # (name, arguments) for dedupe
        role_emitted = False
        last_template: Optional[Dict[str, Any]] = None
        last_kind: Optional[str] = None
        emitted_any_tool = False

        async for item in response:
            parsed: Optional[Dict[str, Any]] = None
            kind: Optional[str] = None

            # Normalize incoming item into dict if possible
            if isinstance(item, ModelResponseStream):
                parsed = item.model_dump()
                kind = "model"
            elif isinstance(item, (bytes, str)):
                line = item.decode() if isinstance(item, bytes) else item
                parsed = _parse_sse_data_line(line)
                if parsed is None:
                    # pass through opaque lines verbatim
                    yield item
                    continue
                if "[DONE]" in parsed:
                    # Before forwarding [DONE], flush any remaining visible text.
                    visible = _strip_tool_blocks(buffer_text)

                    # Try extracting native OpenAI tool_calls if present late in the stream
                    native = _extract_openai_toolcalls_fast(buffer_text)
                    if native:
                        if not role_emitted:
                            role_chunk = _make_stream_chunk(last_template, _mk_role_delta())
                            yield _pack_stream(role_chunk, last_kind or kind or "sse")
                            role_emitted = True
                        for idx, entry in enumerate(native):
                            name = entry.get("function", {}).get("name")
                            args = entry.get("function", {}).get("arguments", "")
                            if name is None:
                                continue
                            call_id = f"call_{uuid4().hex[:12]}"
                            name_chunk = _make_stream_chunk(last_template, _mk_tool_name_delta(idx, call_id, name))
                            yield _pack_stream(name_chunk, last_kind or kind or "sse")
                            if isinstance(args, (dict, list)):
                                args = json.dumps(args, ensure_ascii=False)
                            args_chunk = _make_stream_chunk(last_template, _mk_tool_args_delta(idx, call_id, str(args)))
                            yield _pack_stream(args_chunk, last_kind or kind or "sse")
                            emitted_any_tool = True

                    remaining = visible[len(emitted_visible):]
                    if remaining:
                        if not role_emitted:
                            role_chunk = _make_stream_chunk(last_template, _mk_role_delta())
                            yield _pack_stream(role_chunk, last_kind or kind or "sse")
                            role_emitted = True
                        chunk_dict = _make_stream_chunk(last_template, _mk_content_delta(remaining))
                        yield _pack_stream(chunk_dict, last_kind or kind or "sse")
                        emitted_visible += remaining

                    if emitted_any_tool:
                        finish_chunk = _make_stream_chunk(last_template, _mk_finish_tool_calls_delta(), finish_reason="tool_calls")
                        yield _pack_stream(finish_chunk, last_kind or kind or "sse")
                        emitted_any_tool = False
                        tool_calls_seen.clear()

                    buffer_text = ""
                    emitted_visible = ""
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

                # Enforce buffer bound
                if len(buffer_text.encode("utf-8")) > MAX_BUFFER_BYTES:
                    # Flush visible portion to keep buffer manageable
                    visible_tmp = _strip_tool_blocks(buffer_text)
                    new_visible = visible_tmp[len(emitted_visible):]
                    if new_visible:
                        if not role_emitted:
                            role_chunk = _make_stream_chunk(parsed, _mk_role_delta())
                            yield _pack_stream(role_chunk, kind or last_kind or "dict")
                            role_emitted = True
                        chunk_dict = _make_stream_chunk(parsed, _mk_content_delta(new_visible))
                        yield _pack_stream(chunk_dict, kind or last_kind or "dict")
                        emitted_visible += new_visible
                    # Keep only the tail from the last opening tag to end
                    last_open = max(buffer_text.rfind("<tool_call>"), buffer_text.rfind("<tool_use>"))
                    buffer_text = buffer_text[last_open:] if last_open != -1 else ""

                # Extract and emit normalized tool calls (Hermes/Anthropic blocks)
                calls = _parse_hermes_or_anthro_blocks(buffer_text)
                for c in calls:
                    key = (c["name"], c["arguments"])
                    if key in tool_calls_seen:
                        continue
                    tool_calls_seen.append(key)
                    call_id = f"call_{uuid4().hex[:12]}"
                    if not role_emitted:
                        role_chunk = _make_stream_chunk(parsed, _mk_role_delta())
                        yield _pack_stream(role_chunk, kind or last_kind or "dict")
                        role_emitted = True
                    name_chunk = _make_stream_chunk(parsed, _mk_tool_name_delta(len(tool_calls_seen)-1, call_id, c["name"]))
                    yield _pack_stream(name_chunk, kind or last_kind or "dict")
                    args_chunk = _make_stream_chunk(parsed, _mk_tool_args_delta(len(tool_calls_seen)-1, call_id, c["arguments"]))
                    yield _pack_stream(args_chunk, kind or last_kind or "dict")
                    emitted_any_tool = True

                # Strip tool blocks from visible content
                visible = _strip_tool_blocks(buffer_text)
                new_visible = visible[len(emitted_visible):]
                if new_visible:
                    if not role_emitted:
                        role_chunk = _make_stream_chunk(parsed, _mk_role_delta())
                        yield _pack_stream(role_chunk, kind or last_kind or "dict")
                        role_emitted = True
                    chunk_dict = _make_stream_chunk(parsed, _mk_content_delta(new_visible))
                    yield _pack_stream(chunk_dict, kind or last_kind or "dict")
                    emitted_visible += new_visible

                buffer_text = visible
                continue

            # No content text; pass through original item unchanged.
            yield item


class PolyglotToolsNonStreamingHandler(CustomLogger):
    """
    Normalizes tool calls for non-streaming chat completions (stream=false).
    """

    def __init__(self):
        super().__init__()

    def _make_openai_tool_calls(self, text: str):
        # 1) Native OpenAI JSON fast-path
        native = _extract_openai_toolcalls_fast(text or "")
        calls = []
        if native:
            for entry in native:
                name = entry.get("function", {}).get("name")
                args = entry.get("function", {}).get("arguments", "")
                if name is None:
                    continue
                call_id = f"call_{uuid4().hex[:12]}"
                if isinstance(args, (dict, list)):
                    args = json.dumps(args, ensure_ascii=False)
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                })
            cleaned = text  # you can also strip the native blob if it's echoed
            return calls, cleaned

        # 2) Hermes/Anthropic tags → extract and normalize
        tags = _parse_hermes_or_anthro_blocks(text or "")
        if tags:
            calls = []
            for c in tags:
                call_id = f"call_{uuid4().hex[:12]}"
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]}
                })
            cleaned = _strip_tool_blocks(text or "")
            return calls, cleaned

        return [], text

    async def async_post_call_hook(
        self,
        user_api_key_dict,
        original_response: dict,
        request_data: dict,
    ):
        """
        LiteLLM calls this for non-streaming responses.
        Modify `original_response` in-place to add OpenAI tool_calls if needed.
        """
        try:
            choices = original_response.get("choices") or []
            if not choices:
                return
            msg = choices[0].get("message") or {}
            content = msg.get("content", "")
            tool_calls, cleaned = self._make_openai_tool_calls(content)
            if tool_calls:
                msg["tool_calls"] = tool_calls
                msg["content"] = cleaned or ""
                choices[0]["finish_reason"] = "tool_calls"
                original_response["choices"][0]["message"] = msg
        except Exception:
            # Fail open on parser errors
            return


# Export instances for LiteLLM config
proxy_handler_instance = PolyglotToolsStreamingHandler()
nonstream_handler_instance = PolyglotToolsNonStreamingHandler()
