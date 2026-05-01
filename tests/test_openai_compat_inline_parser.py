from __future__ import annotations

from imajin.agent.providers.openai_compat import (
    _parse_inline_tool_calls,
    _slice_first_json,
)


KNOWN = {"cellpose_sam", "list_layers", "measure_intensity"}


def test_array_form_with_arguments_key() -> None:
    text = '[{"name":"cellpose_sam","arguments":{"image_layer":"Ch1","do_3D":true}}]'
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "cellpose_sam"
    assert result[0]["input"] == {"image_layer": "Ch1", "do_3D": True}


def test_single_object_with_parameters_key() -> None:
    text = 'Some prelude\n{"name": "list_layers", "parameters": {}}'
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "list_layers"
    assert result[0]["input"] == {}


def test_code_fenced_json() -> None:
    text = (
        "I'll segment now.\n"
        '```json\n{"name": "cellpose_sam", "arguments": {"image_layer": "Ch2"}}\n```'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["input"]["image_layer"] == "Ch2"


def test_array_with_multiple_calls() -> None:
    text = (
        '[{"name":"cellpose_sam","arguments":{"x":1}},'
        '{"name":"measure_intensity","arguments":{"y":2}}]'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 2
    assert [r["name"] for r in result] == ["cellpose_sam", "measure_intensity"]


def test_unknown_tool_name_filtered_out() -> None:
    text = '[{"name":"not_a_real_tool","arguments":{}}]'
    assert _parse_inline_tool_calls(text, KNOWN) == []


def test_plain_text_returns_empty() -> None:
    text = "Hello, I will help you analyze your data. Please share a file path."
    assert _parse_inline_tool_calls(text, KNOWN) == []


def test_string_arguments_decoded() -> None:
    # Some models nest arguments as a JSON-encoded string.
    text = '{"name": "cellpose_sam", "arguments": "{\\"image_layer\\": \\"Ch1\\"}"}'
    result = _parse_inline_tool_calls(text, KNOWN)
    assert result == [
        {"id": "inline_0", "name": "cellpose_sam", "input": {"image_layer": "Ch1"}}
    ]


def test_function_wrapper_form() -> None:
    text = (
        '{"function": {"name": "list_layers", "arguments": "{}"}, "name": "list_layers"}'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "list_layers"


def test_slice_first_json_object() -> None:
    text = 'prefix {"a": 1, "b": [2, 3]} suffix'
    assert _slice_first_json(text) == '{"a": 1, "b": [2, 3]}'


def test_slice_first_json_array() -> None:
    text = 'pre [{"x": "}{"}, {"y": 2}] post'
    assert _slice_first_json(text) == '[{"x": "}{"}, {"y": 2}]'


def test_slice_first_json_handles_strings_with_braces() -> None:
    text = '{"msg": "hello {nested} braces"}'
    assert _slice_first_json(text) == text
