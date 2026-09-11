from __future__ import annotations

import pytest

from imajin.tools import (
    call_tool,
    iter_tools,
    manual_tools,
    tool,
    tools_for_anthropic,
    tools_for_openai,
)


@pytest.fixture(autouse=True)
def reset_registry():
    from imajin.tools import registry

    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def test_tool_registers_with_pydantic_schema() -> None:
    @tool(description="Add two numbers", phase="test")
    def add(a: int, b: int = 1) -> int:
        return a + b

    entries = iter_tools()
    assert len(entries) == 1
    e = entries[0]
    assert e.name == "add"
    assert e.description == "Add two numbers"
    schema = e.json_schema
    assert schema["type"] == "object"
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]
    assert "a" in schema["required"]
    assert "b" not in schema.get("required", [])


def test_anthropic_schema_shape() -> None:
    @tool()
    def t(x: int) -> int:
        return x * 2

    [spec] = tools_for_anthropic()
    assert spec["name"] == "t"
    assert "input_schema" in spec
    assert spec["input_schema"]["properties"]["x"]["type"] == "integer"


def test_openai_schema_shape() -> None:
    @tool()
    def t(x: int) -> int:
        return x * 2

    [spec] = tools_for_openai()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "t"
    assert spec["function"]["parameters"]["properties"]["x"]["type"] == "integer"


def test_manual_and_llm_visibility_flags() -> None:
    @tool(manual=False)
    def hidden_manual(x: int) -> int:
        return x

    @tool(llm=False)
    def hidden_llm(x: int) -> int:
        return x

    assert {e.name for e in manual_tools()} == {"hidden_llm"}
    assert {s["name"] for s in tools_for_anthropic()} == {"hidden_manual"}


def test_call_tool_validates_and_invokes() -> None:
    @tool()
    def square(n: int) -> int:
        return n * n

    assert call_tool("square", n=4) == 16


def test_call_tool_provenance_records(tmp_path, monkeypatch) -> None:
    from imajin.agent import provenance
    from imajin.config import Settings

    settings = Settings(data_dir=tmp_path)
    provenance.start_session(driver="test", settings=settings)

    @tool()
    def add(a: int, b: int) -> int:
        return a + b

    call_tool("add", a=2, b=3)

    log = provenance.current_session_path()
    assert log is not None
    assert log.exists()
    contents = log.read_text().strip().splitlines()
    assert len(contents) == 1
    import json

    rec = json.loads(contents[0])
    assert rec["tool"] == "add"
    assert rec["ok"] is True
    assert rec["driver"] == "test"
    assert rec["output_summary"] == 5


# -- get_tool/call_tool on an unknown name ----------------------------------
#
# A bare `_REGISTRY[name]` KeyError gives the model only `ERROR: 'the_name'`
# (see runner.py's `except Exception as e: content = f"ERROR: {e}"`), which
# doesn't say whether the name is misspelled, renamed, or just not advertised
# this session -- so the model retries the identical call. That bites harder
# now that local models see only the 20-tool core of the full 107-tool
# registry (a real tool is routinely "not advertised"), and it also protects
# a cloud model that misremembers a renamed tool.


def test_get_tool_unknown_name_names_it_in_the_message() -> None:
    from imajin.tools.registry import get_tool

    with pytest.raises(KeyError) as exc_info:
        get_tool("list_registered_files")

    assert "list_registered_files" in str(exc_info.value)


def test_get_tool_unknown_name_is_still_a_key_error() -> None:
    # runner.py's vision-hint overlay check and qt_tool_runner.py's
    # cross-thread dispatch both already do a defensive `except KeyError`
    # around a tool lookup and must keep degrading gracefully rather than
    # seeing an unrecognized exception type -- so this must stay a KeyError.
    from imajin.tools.registry import ToolNotFoundError, get_tool

    with pytest.raises(KeyError) as exc_info:
        get_tool("does_not_exist_at_all")

    assert isinstance(exc_info.value, ToolNotFoundError)


def test_get_tool_near_miss_suggests_the_real_tool_name() -> None:
    from imajin.tools.registry import get_tool

    @tool()
    def measure_intensity(x: int) -> int:
        return x

    @tool()
    def segment_target_objects(x: int) -> int:
        return x

    with pytest.raises(KeyError) as exc_info:
        get_tool("measure_intensityy")

    assert "measure_intensity" in str(exc_info.value)


def test_get_tool_unknown_name_points_at_get_help() -> None:
    from imajin.tools.registry import get_tool

    with pytest.raises(KeyError) as exc_info:
        get_tool("some_made_up_tool")

    assert "get_help" in str(exc_info.value)


def test_call_tool_unknown_name_raises_the_same_clear_error() -> None:
    # call_tool (the path every tool_caller actually dispatches through), not
    # just get_tool, must not regress to a raw `_REGISTRY[tool_name]` KeyError.
    with pytest.raises(KeyError) as exc_info:
        call_tool("list_registered_files", foo=1)

    assert "list_registered_files" in str(exc_info.value)
    assert "get_help" in str(exc_info.value)
