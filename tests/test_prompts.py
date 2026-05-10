from __future__ import annotations

from imajin.agent.prompts import SYSTEM_PROMPT


def test_system_prompt_guides_bundle_recipe_reuse() -> None:
    assert "import_recipe_from_bundle" in SYSTEM_PROMPT
    assert "recipe_params" in SYSTEM_PROMPT
    assert "file scope" in SYSTEM_PROMPT
    assert "channel roles" in SYSTEM_PROMPT
    assert "run_context" in SYSTEM_PROMPT
