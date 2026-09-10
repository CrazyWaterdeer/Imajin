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


# --- extended shapes (reversed key order, tool_calls wrapper, XML tags,
# python_tag, multiple fenced blocks, ALL calls not just the first) ---------


def test_reversed_key_order_arguments_before_name() -> None:
    # The old gate regex required "name" to appear textually before the
    # arguments-like key; a model that emits arguments first used to be
    # invisible to this parser entirely.
    text = '{"arguments": {"image_layer": "Ch1"}, "name": "cellpose_sam"}'
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "cellpose_sam"
    assert result[0]["input"] == {"image_layer": "Ch1"}


def test_reversed_key_order_inside_array() -> None:
    text = '[{"parameters": {}, "name": "list_layers"}]'
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "list_layers"


def test_tool_calls_wrapper_form() -> None:
    text = (
        '{"tool_calls": ['
        '{"name": "list_layers", "arguments": {}}, '
        '{"name": "measure_intensity", "arguments": {"y": 1}}'
        "]}"
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert [r["name"] for r in result] == ["list_layers", "measure_intensity"]


def test_tool_calls_wrapper_openai_native_shape() -> None:
    # The proper streamed shape (id/type/function.arguments-as-JSON-string),
    # just misplaced in content instead of the tool_calls channel.
    text = (
        '{"tool_calls": [{"id": "call_1", "type": "function", '
        '"function": {"name": "cellpose_sam", "arguments": "{\\"image_layer\\": \\"Ch1\\"}"}}]}'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "cellpose_sam"
    assert result[0]["input"] == {"image_layer": "Ch1"}


def test_function_tag_form() -> None:
    text = 'Sure, calling it now.\n<function=cellpose_sam>{"image_layer": "Ch3"}</function>'
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "cellpose_sam"
    assert result[0]["input"] == {"image_layer": "Ch3"}


def test_function_tag_unknown_name_filtered_out() -> None:
    text = "<function=not_a_real_tool>{}</function>"
    assert _parse_inline_tool_calls(text, KNOWN) == []


def test_tool_call_xml_tag_form() -> None:
    text = (
        '<tool_call>\n{"name": "cellpose_sam", "arguments": {"image_layer": "Ch4"}}\n</tool_call>'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "cellpose_sam"
    assert result[0]["input"] == {"image_layer": "Ch4"}


def test_tool_call_xml_tag_multiple_calls_get_unique_ids() -> None:
    text = (
        '<tool_call>{"name": "list_layers", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "measure_intensity", "arguments": {"y": 2}}</tool_call>'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert [r["name"] for r in result] == ["list_layers", "measure_intensity"]
    assert [r["id"] for r in result] == ["inline_0", "inline_1"]


def test_python_tag_form() -> None:
    text = '<|python_tag|>{"name": "cellpose_sam", "parameters": {"image_layer": "Ch5"}}'
    result = _parse_inline_tool_calls(text, KNOWN)
    assert len(result) == 1
    assert result[0]["name"] == "cellpose_sam"
    assert result[0]["input"] == {"image_layer": "Ch5"}


def test_python_tag_parallel_calls_separated_by_semicolon() -> None:
    # Meta's built-in format allows several ;-separated calls after one marker.
    text = (
        '<|python_tag|>{"name": "list_layers", "arguments": {}}; '
        '{"name": "measure_intensity", "arguments": {"y": 3}}<|eom_id|>'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert [r["name"] for r in result] == ["list_layers", "measure_intensity"]


def test_multiple_separate_fenced_blocks_all_returned() -> None:
    text = (
        "First call:\n"
        '```json\n{"name": "list_layers", "arguments": {}}\n```\n'
        "Second call:\n"
        '```json\n{"name": "measure_intensity", "arguments": {"y": 4}}\n```'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    assert [r["name"] for r in result] == ["list_layers", "measure_intensity"]
    assert len({r["id"] for r in result}) == 2


def test_unique_ids_across_mixed_tag_shapes_in_one_message() -> None:
    text = (
        '<tool_call>{"name": "list_layers", "arguments": {}}</tool_call>'
        '<function=measure_intensity>{"y": 5}</function>'
    )
    result = _parse_inline_tool_calls(text, KNOWN)
    ids = [r["id"] for r in result]
    assert len(ids) == len(set(ids)) == 2
    assert {r["name"] for r in result} == {"list_layers", "measure_intensity"}
