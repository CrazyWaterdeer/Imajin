from __future__ import annotations

import pytest

from imajin.tools import call_tool, iter_tools, tool, tools_for_anthropic, tools_for_openai


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


def test_call_tool_validates_and_invokes() -> None:
    @tool()
    def square(n: int) -> int:
        return n * n

    assert call_tool("square", n=4) == 16


def test_call_tool_provenance_records(tmp_path, monkeypatch) -> None:
    from imajin.agent import provenance
    from imajin.config import Settings

    settings = Settings(data_dir=tmp_path)
    sid = provenance.start_session(driver="test", settings=settings)

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
