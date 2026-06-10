"""Backward-compatibility shim.

The session-state module moved to :mod:`imajin.session`. This re-export keeps the
old ``imajin.agent.state`` import path working while call sites are migrated
(Phase 2 of the session-state-extraction plan). Removed in C17 once no references
to ``agent.state`` remain.
"""
from imajin.session import *  # noqa: F401,F403
