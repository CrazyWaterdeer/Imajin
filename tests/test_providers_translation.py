from __future__ import annotations

from imajin.agent.providers.openai_compat import _anthropic_to_openai_messages


def test_translates_simple_user_text() -> None:
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    out = _anthropic_to_openai_messages(msgs)
    assert out == [{"role": "user", "content": "hello"}]


def test_translates_assistant_with_tool_use() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Calling list."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "list_layers",
                    "input": {},
                },
            ],
        }
    ]
    out = _anthropic_to_openai_messages(msgs)
    assert len(out) == 1
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] == "Calling list."
    assert out[0]["tool_calls"][0]["function"]["name"] == "list_layers"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_translates_user_tool_result() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": '{"n_cells": 5}',
                }
            ],
        }
    ]
    out = _anthropic_to_openai_messages(msgs)
    assert out == [
        {"role": "tool", "tool_call_id": "tu_1", "content": '{"n_cells": 5}'}
    ]
