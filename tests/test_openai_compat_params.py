from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from openai import BadRequestError, OpenAI

from imajin.agent.providers.base import Stop, TextDelta, ToolUse, ToolUseStart
from imajin.agent.providers.openai_compat import OpenAICompatProvider

# Covers items 2, 3, 5, 6 of the openai_compat.py fix list:
#   2. max_tokens vs max_completion_tokens: retry once on a 400 naming the
#      other key, and remember the fix on the instance.
#   3. Vision: a trailing image_url message, gated on supports_vision.
#   5. stream_options={"include_usage": True} with a tolerant retry.
#   6. Split connect/read timeouts.
#
# No real network or Ollama daemon is used anywhere here: every test routes
# the OpenAI SDK's own HTTP client through httpx.MockTransport, so the SDK's
# real request-building / SSE-parsing / error-parsing code runs against a
# fully in-process fake server.


def _fake_provider(
    handler, *, model: str = "gpt-5", supports_vision: bool = False
) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(
        api_key="test-key", model=model, base_url="http://fake.test/v1",
        supports_vision=supports_vision,
    )
    # Swap in a client whose transport is fully faked -- this is the "fake
    # the HTTP layer" the house rules ask for; OpenAICompatProvider itself
    # gets no test-only constructor hook.
    provider._client = OpenAI(
        api_key="test-key",
        base_url="http://fake.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return provider


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _sse_response(
    *,
    text: str = "",
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
) -> httpx.Response:
    lines = []
    if text:
        lines.append(json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]}))
    stop_line = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
    lines.append(json.dumps(stop_line))
    if usage is not None:
        lines.append(json.dumps({"choices": [], "usage": usage}))
    body = "".join(f"data: {line}\n\n" for line in lines) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _error_response(message: str, *, param: str | None = None) -> httpx.Response:
    err: dict[str, Any] = {"message": message, "type": "invalid_request_error"}
    if param:
        err["param"] = param
    return httpx.Response(400, json={"error": err})


# --- item 2: max_tokens vs max_completion_tokens ----------------------------


def test_sends_max_tokens_by_default_not_max_completion_tokens() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return _sse_response(text="Hello!")

    provider = _fake_provider(handler)
    list(provider.stream([_user("hi")], [], "sys"))

    assert len(seen) == 1
    assert seen[0]["max_tokens"] == provider.max_tokens
    assert "max_completion_tokens" not in seen[0]


def test_retries_with_max_completion_tokens_on_400_and_remembers_it() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        seen.append(body)
        if "max_tokens" in body:
            return _error_response(
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead.",
                param="max_tokens",
            )
        return _sse_response(text="ok")

    provider = _fake_provider(handler, model="o-mini-future")
    events = list(provider.stream([_user("hi")], [], "sys"))

    assert len(seen) == 2
    assert "max_tokens" in seen[0] and "max_completion_tokens" not in seen[0]
    assert "max_completion_tokens" in seen[1] and "max_tokens" not in seen[1]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "ok"

    # Remembered on the instance: a second, independent stream() call goes
    # straight to max_completion_tokens without 400ing again.
    seen.clear()
    list(provider.stream([_user("again")], [], "sys"))
    assert len(seen) == 1
    assert "max_completion_tokens" in seen[0]


def test_unrecognized_400_is_not_swallowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _error_response("Invalid value for 'temperature': 5", param="temperature")

    provider = _fake_provider(handler)
    with pytest.raises(BadRequestError):
        list(provider.stream([_user("hi")], [], "sys"))


# --- item 5: stream_options ---------------------------------------------


def test_sends_stream_options_include_usage_by_default() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return _sse_response(text="ok", usage={"prompt_tokens": 5, "completion_tokens": 2})

    provider = _fake_provider(handler)
    events = list(provider.stream([_user("hi")], [], "sys"))

    assert seen[0]["stream_options"] == {"include_usage": True}
    stop = next(e for e in events if isinstance(e, Stop))
    assert stop.usage == {"input_tokens": 5, "output_tokens": 2}


def test_usage_maps_cached_tokens_too() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(
            text="ok",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        )

    provider = _fake_provider(handler)
    events = list(provider.stream([_user("hi")], [], "sys"))
    stop = next(e for e in events if isinstance(e, Stop))
    assert stop.usage == {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_read_input_tokens": 80,
    }


def test_retries_without_stream_options_on_400_and_remembers_it() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        seen.append(body)
        if "stream_options" in body:
            return _error_response(
                "Unrecognized request argument supplied: stream_options",
                param="stream_options",
            )
        return _sse_response(text="ok")

    provider = _fake_provider(handler)
    events = list(provider.stream([_user("hi")], [], "sys"))

    assert len(seen) == 2
    assert "stream_options" in seen[0]
    assert "stream_options" not in seen[1]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "ok"

    seen.clear()
    list(provider.stream([_user("again")], [], "sys"))
    assert len(seen) == 1
    assert "stream_options" not in seen[0]


def test_both_retry_causes_can_fire_in_one_call() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        seen.append(body)
        if "max_tokens" in body:
            return _error_response(
                "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.",
                param="max_tokens",
            )
        if "stream_options" in body:
            return _error_response("Unrecognized request argument: stream_options")
        return _sse_response(text="ok")

    provider = _fake_provider(handler)
    events = list(provider.stream([_user("hi")], [], "sys"))

    assert len(seen) == 3
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "ok"


# --- item 3: vision ----------------------------------------------------


def _tool_result_with_image(tool_use_id: str = "t1") -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [
                    {"type": "text", "text": "segmented ok"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "ZZZ",
                        },
                    },
                ],
            }
        ],
    }


def _messages_with_tool_result_image() -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "seg", "input": {}}],
        },
        _tool_result_with_image(),
    ]


def test_supports_vision_false_by_default_never_sends_image() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return _sse_response(text="ok")

    provider = _fake_provider(handler)  # supports_vision defaults False
    list(provider.stream(_messages_with_tool_result_image(), [], "sys"))

    sent = seen[0]["messages"]
    assert not any(
        isinstance(m.get("content"), list)
        and any(part.get("type") == "image_url" for part in m["content"])
        for m in sent
    )
    # The text half of the tool result still made it through.
    tool_msgs = [m for m in sent if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == "segmented ok"


def test_supports_vision_true_appends_trailing_image_message() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return _sse_response(text="ok")

    provider = _fake_provider(handler, supports_vision=True)
    list(provider.stream(_messages_with_tool_result_image(), [], "sys"))

    sent = seen[0]["messages"]
    assert sent[-1]["role"] == "user"
    assert sent[-1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZZ"}}
    ]
    tool_msgs = [m for m in sent if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == "segmented ok"


def test_supports_vision_true_with_no_images_appends_nothing_extra() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return _sse_response(text="ok")

    provider = _fake_provider(handler, supports_vision=True)
    list(provider.stream([_user("hi")], [], "sys"))

    sent = seen[0]["messages"]
    # Just system + the one user turn -- no trailing image message conjured
    # out of nothing.
    assert len(sent) == 2
    assert sent[-1] == {"role": "user", "content": "hi"}


# --- item 6: split connect/read timeouts --------------------------------


def test_split_connect_and_read_timeouts_default() -> None:
    provider = OpenAICompatProvider(
        api_key=None, model="qwen3.5:9b", base_url="http://localhost:11434/v1"
    )
    timeout = provider._client.timeout
    # Connect fails fast; read stays generous enough for a local model to
    # load and prefill a full tool catalogue (measured: 25.5s for a small
    # prompt at num_ctx=32768 -- a real catalogue is worse).
    assert timeout.connect is not None and timeout.connect <= 15.0
    assert timeout.read is not None and timeout.read >= 300.0
    assert timeout.read > timeout.connect


def test_timeout_constructor_arg_controls_the_read_timeout() -> None:
    provider = OpenAICompatProvider(
        api_key=None, model="m", base_url="http://localhost:11434/v1", timeout=120.0
    )
    assert provider._client.timeout.read == 120.0
    assert provider._client.timeout.connect == 10.0


# --- sanity: proper tool_calls channel still round-trips end to end -------


def test_proper_tool_calls_channel_and_supports_tools_property_unaffected() -> None:
    sse = (
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"list_layers","arguments":""}}]}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"{}"}}]}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": "text/event-stream"}
        return httpx.Response(200, content=sse.encode(), headers=headers)

    provider = _fake_provider(handler)
    tools = [{"name": "list_layers", "description": "", "input_schema": {"type": "object"}}]
    events = list(provider.stream([_user("list them")], tools, "sys"))

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    uses = [e for e in events if isinstance(e, ToolUse)]
    stop = next(e for e in events if isinstance(e, Stop))
    assert starts and starts[0].name == "list_layers"
    assert uses and uses[0].name == "list_layers" and uses[0].input == {}
    assert stop.reason == "tool_use"
