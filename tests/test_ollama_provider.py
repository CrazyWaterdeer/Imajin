"""Tests for OllamaProvider against a faked NDJSON line-stream.

The one network seam (`_iter_ndjson_lines`) is faked, so nothing here needs a
real Ollama daemon or network access. Lines are the raw JSON strings Ollama's
native `POST /api/chat` (stream: true) would write, one object per line.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from imajin.agent.providers import ollama as ollama_mod
from imajin.agent.providers.base import Stop, TextDelta, ToolUse, ToolUseStart
from imajin.agent.providers.ollama import OllamaProvider


class _FakeStream:
    """Stands in for ollama._iter_ndjson_lines. Records every call's (url,
    payload, timeout) so tests can inspect exactly what would have been POSTed,
    and replays a canned list of JSON-line strings instead of reading a socket.
    """

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> list[str]:
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        return list(self.lines)


def _line(**kwargs: Any) -> str:
    return json.dumps(kwargs)


def _done(**extra: Any) -> str:
    return _line(message={"content": ""}, done=True, **extra)


# -- basic text streaming -----------------------------------------------------


def test_text_only_stream_emits_text_then_stop(monkeypatch) -> None:
    fake = _FakeStream(
        [
            _line(message={"content": "Hello "}, done=False),
            _line(message={"content": "world"}, done=False),
            _done(done_reason="stop"),
        ]
    )
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b")
    events = list(provider.stream([], [], "you are a helpful assistant"))

    assert [e for e in events if isinstance(e, TextDelta)][0].text == "Hello "
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello world"
    assert isinstance(events[-1], Stop)
    assert events[-1].reason == "stop"


def test_stop_reason_falls_back_to_end_turn_when_done_reason_absent(monkeypatch) -> None:
    fake = _FakeStream([_line(message={"content": "hi"}, done=True)])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b")
    events = list(provider.stream([], [], "sys"))

    assert events[-1].reason == "end_turn"


# -- tool calls: complete-in-one-chunk, both argument forms -------------------


def test_tool_call_with_string_arguments(monkeypatch) -> None:
    args_str = json.dumps({"paths": ["/data/exp1"], "recursive": True})
    fake = _FakeStream(
        [
            _line(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "function": {"name": "register_files", "arguments": args_str},
                        }
                    ],
                },
                done=False,
            ),
            _done(done_reason="stop", prompt_eval_count=28323, eval_count=42),
        ]
    )
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b")
    tools = [{"name": "register_files", "description": "", "input_schema": {"type": "object"}}]
    events = list(provider.stream([], tools, "sys"))

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    uses = [e for e in events if isinstance(e, ToolUse)]
    assert starts == [ToolUseStart(id="call_abc", name="register_files")]
    assert uses == [
        ToolUse(
            id="call_abc",
            name="register_files",
            input={"paths": ["/data/exp1"], "recursive": True},
        )
    ]
    stop = events[-1]
    assert isinstance(stop, Stop)
    assert stop.reason == "tool_use"
    assert stop.usage == {"input_tokens": 28323, "output_tokens": 42}
    # ToolUseStart must be emitted before the matching ToolUse.
    assert events.index(starts[0]) < events.index(uses[0])


def test_tool_call_with_object_arguments(monkeypatch) -> None:
    fake = _FakeStream(
        [
            _line(
                message={
                    "tool_calls": [
                        {"id": "call_xyz", "function": {"name": "list_layers", "arguments": {"filter": "image"}}}
                    ]
                },
                done=False,
            ),
            _done(done_reason="stop"),
        ]
    )
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b")
    events = list(provider.stream([], [], "sys"))

    uses = [e for e in events if isinstance(e, ToolUse)]
    assert uses == [ToolUse(id="call_xyz", name="list_layers", input={"filter": "image"})]
    assert events[-1].reason == "tool_use"


def test_missing_tool_call_id_is_synthesized_stably(monkeypatch) -> None:
    fake = _FakeStream(
        [
            _line(message={"tool_calls": [{"function": {"name": "list_layers", "arguments": {}}}]}, done=False),
            _done(),
        ]
    )
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b")
    events = list(provider.stream([], [], "sys"))

    start = next(e for e in events if isinstance(e, ToolUseStart))
    use = next(e for e in events if isinstance(e, ToolUse))
    assert start.id == "ollama_0"
    assert use.id == "ollama_0"


def test_fragmented_tool_call_accumulates_across_chunks(monkeypatch) -> None:
    """Ollama normally sends a tool call complete in one chunk, but the
    accumulator must also cope with a name arriving separately from arguments,
    and arguments arriving in pieces (mirrors OpenAI-style fragmentation)."""
    fake = _FakeStream(
        [
            _line(
                message={"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "list_layers"}}]},
                done=False,
            ),
            _line(
                message={"tool_calls": [{"index": 0, "function": {"arguments": '{"filter": '}}]},
                done=False,
            ),
            _line(
                message={"tool_calls": [{"index": 0, "function": {"arguments": '"image"}'}}]},
                done=False,
            ),
            _done(done_reason="stop"),
        ]
    )
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b")
    events = list(provider.stream([], [], "sys"))

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    uses = [e for e in events if isinstance(e, ToolUse)]
    assert len(starts) == 1  # not re-emitted once the name is already known
    assert starts[0].id == "call_1"
    assert uses == [ToolUse(id="call_1", name="list_layers", input={"filter": "image"})]


# -- mid-stream error ----------------------------------------------------------


def test_mid_stream_error_chunk_raises_runtime_error(monkeypatch) -> None:
    fake = _FakeStream(
        [
            _line(message={"content": "partial"}, done=False),
            _line(error="model requires more system memory than is available"),
        ]
    )
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b")
    gen = provider.stream([], [], "sys")

    collected = []
    with pytest.raises(RuntimeError, match="more system memory"):
        for event in gen:
            collected.append(event)
    assert len(collected) == 1
    assert isinstance(collected[0], TextDelta)


# -- options.num_ctx --------------------------------------------------------


def test_num_ctx_included_when_given(monkeypatch) -> None:
    fake = _FakeStream([_done()])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b", num_ctx=49152)
    list(provider.stream([], [], "sys"))

    payload = fake.calls[0]["payload"]
    assert payload["options"]["num_ctx"] == 49152


def test_num_ctx_omitted_when_not_given(monkeypatch) -> None:
    fake = _FakeStream([_done()])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b")
    list(provider.stream([], [], "sys"))

    payload = fake.calls[0]["payload"]
    assert "num_ctx" not in payload.get("options", {})


def test_max_tokens_maps_to_num_predict(monkeypatch) -> None:
    fake = _FakeStream([_done()])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b", max_tokens=2048)
    list(provider.stream([], [], "sys"))

    assert fake.calls[0]["payload"]["options"]["num_predict"] == 2048


def test_keep_alive_and_model_passed_through(monkeypatch) -> None:
    fake = _FakeStream([_done()])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b", keep_alive="10m", base_url="http://localhost:11434/v1")
    list(provider.stream([], [], "sys"))

    call = fake.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["payload"]["model"] == "qwen3.5:9b"
    assert call["payload"]["keep_alive"] == "10m"
    assert call["payload"]["stream"] is True


# -- vision: image attachment --------------------------------------------------


def _messages_with_overlay_image() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "segment this"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "cellpose_sam", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": [
                        {"type": "text", "text": '{"roi_confidence": "low"}'},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "QUFBQQ==",
                            },
                        },
                    ],
                }
            ],
        },
    ]


def test_image_attached_natively_when_vision_supported(monkeypatch) -> None:
    fake = _FakeStream([_done()])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="qwen3.5:9b", supports_vision=True)
    list(provider.stream(_messages_with_overlay_image(), [], "sys"))

    oai_messages = fake.calls[0]["payload"]["messages"]
    tool_msg = next(m for m in oai_messages if m.get("role") == "tool")
    assert tool_msg["images"] == ["QUFBQQ=="]
    # And it must not be smuggled in as an OpenAI-style image_url block too.
    assert "image_url" not in json.dumps(tool_msg)


def test_image_dropped_when_vision_unsupported(monkeypatch) -> None:
    fake = _FakeStream([_done()])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)

    provider = OllamaProvider(model="llama3:8b", supports_vision=False)
    list(provider.stream(_messages_with_overlay_image(), [], "sys"))

    oai_messages = fake.calls[0]["payload"]["messages"]
    assert all("images" not in m for m in oai_messages)


# -- HTTP error bodies --------------------------------------------------------


def _http_error(code: int, reason: str, body: bytes):
    import io
    import urllib.error

    return urllib.error.HTTPError(
        "http://x/api/chat", code, reason, {}, io.BytesIO(body)
    )


def test_http_error_body_is_surfaced_not_just_the_status(monkeypatch) -> None:
    """A bare "HTTP Error 404: Not Found" tells the user nothing; Ollama's own
    {"error": ...} body names the actual problem (a model that isn't pulled)."""

    def boom(request, timeout):
        raise _http_error(404, "Not Found", b'{"error":"model \'ghost:1b\' not found"}')

    monkeypatch.setattr(ollama_mod.urllib.request, "urlopen", boom)
    provider = OllamaProvider(model="ghost:1b", base_url="http://localhost:11434/v1")

    with pytest.raises(RuntimeError) as excinfo:
        list(provider.stream([{"role": "user", "content": "hi"}], [], "sys"))

    assert "model 'ghost:1b' not found" in str(excinfo.value)
    assert "404" in str(excinfo.value)


def test_http_error_context_message_reaches_runner_detection(monkeypatch) -> None:
    """runner._context_limit_error matches on message text, so the body must
    carry through for the compaction retry to be reachable at all."""
    from imajin.agent.runner import _context_limit_error

    def boom(request, timeout):
        raise _http_error(400, "Bad Request", b'{"error":"context length exceeded"}')

    monkeypatch.setattr(ollama_mod.urllib.request, "urlopen", boom)
    provider = OllamaProvider(model="m", base_url="http://localhost:11434/v1")

    with pytest.raises(RuntimeError) as excinfo:
        list(provider.stream([{"role": "user", "content": "hi"}], [], "sys"))

    assert _context_limit_error(excinfo.value)


def test_http_error_falls_back_to_reason_when_body_is_empty(monkeypatch) -> None:
    def boom(request, timeout):
        raise _http_error(500, "Internal Server Error", b"")

    monkeypatch.setattr(ollama_mod.urllib.request, "urlopen", boom)
    provider = OllamaProvider(model="m", base_url="http://localhost:11434/v1")

    with pytest.raises(RuntimeError) as excinfo:
        list(provider.stream([{"role": "user", "content": "hi"}], [], "sys"))

    assert "Internal Server Error" in str(excinfo.value)


# -- reasoning ("think") and output-budget truncation -------------------------


def test_think_omitted_by_default(monkeypatch) -> None:
    # Default must leave the server's own reasoning behaviour alone: thinking
    # measurably helps a small model pick among 104 tools, and with max_tokens
    # unset there is no num_predict ceiling for it to exhaust.
    fake = _FakeStream([_done(done_reason="stop")])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)
    list(OllamaProvider(model="m").stream([], [], "sys"))
    assert "think" not in fake.calls[0]["payload"]


@pytest.mark.parametrize("think", [True, False])
def test_think_forwarded_when_set(monkeypatch, think: bool) -> None:
    fake = _FakeStream([_done(done_reason="stop")])
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)
    list(OllamaProvider(model="m", think=think).stream([], [], "sys"))
    assert fake.calls[0]["payload"]["think"] is think


def test_tool_call_truncated_by_length_reports_length_not_tool_use(monkeypatch) -> None:
    """Args cut off mid-JSON must not be reported as a clean tool_use.

    Running such a call with `{}` would either raise a confusing validation
    error or execute with defaults the model never chose. Reporting "length"
    routes it to the runner's truncation guard instead.
    """
    lines = [
        _line(
            message={
                "content": "",
                "tool_calls": [
                    {"function": {"name": "register_files", "arguments": '{"paths": ["/dat'}}
                ],
            },
            done=False,
        ),
        _done(done_reason="length"),
    ]
    fake = _FakeStream(lines)
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)
    events = list(OllamaProvider(model="m", max_tokens=16).stream([], [], "sys"))

    stop = [e for e in events if isinstance(e, Stop)][0]
    assert stop.reason == "length"
    # The ToolUse is still emitted (the UI shows the attempt) but with empty args.
    tool_uses = [e for e in events if isinstance(e, ToolUse)]
    assert len(tool_uses) == 1 and tool_uses[0].input == {}


def test_complete_tool_call_still_reports_tool_use_even_at_length_limit(monkeypatch) -> None:
    # done_reason="length" alone must not suppress a call whose arguments did
    # parse — the limit was hit *after* the call was fully emitted.
    lines = [
        _line(
            message={
                "content": "",
                "tool_calls": [
                    {"function": {"name": "register_files", "arguments": '{"paths": ["/d"]}'}}
                ],
            },
            done=False,
        ),
        _done(done_reason="length"),
    ]
    fake = _FakeStream(lines)
    monkeypatch.setattr(ollama_mod, "_iter_ndjson_lines", fake)
    events = list(OllamaProvider(model="m").stream([], [], "sys"))
    assert [e for e in events if isinstance(e, Stop)][0].reason == "tool_use"
    assert [e for e in events if isinstance(e, ToolUse)][0].input == {"paths": ["/d"]}
