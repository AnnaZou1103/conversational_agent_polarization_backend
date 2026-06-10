"""End-to-end check that each pipeline stage runs on the correct model.

Drives the real AgentPipeline.process_turn with two distinct providers — a
"MAIN" model and a "FAST" model — and asserts the per-stage routing:

  OBSERVE        -> fast_llm   (FAST)
  Stage eval     -> fast_llm   (FAST)
  THINK          -> llm        (MAIN)   [only when enable_think]
  EXECUTE stream -> llm        (MAIN)

This is the routing introduced for latency reduction: cheap model for the
internal critical-path steps, main model for THINK and the user-facing reply.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub the DB layer before importing any app module (no MongoDB at import time).
_mock_col = MagicMock()
_db_stub = MagicMock()
_db_stub.user_docs = _mock_col
_db_stub.conversation_docs = _mock_col
sys.modules.setdefault("app.db.documents", _db_stub)
sys.modules.setdefault("app.db.user", MagicMock(
    get_user_party=MagicMock(return_value=None),
    study_id_is_valid=MagicMock(return_value=True),
))
sys.modules.setdefault("app.db.conversation", MagicMock(
    get_conversation=MagicMock(return_value=None),
    save_turn_log=MagicMock(),
    save_safety_event=MagicMock(),
    get_chat_history=MagicMock(return_value=[]),
))

from app.agent.pipeline import AgentPipeline  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm.base import LLMProvider, Message  # noqa: E402


def _classify(prompt: str) -> str:
    """Identify which pipeline step a complete() prompt belongs to."""
    if "stage controller" in prompt:
        return "stage"
    if "internal reasoning module" in prompt:
        return "think"
    return "observe"


class RecordingLLM(LLMProvider):
    """Provider tagged with a model label; records every call it receives."""

    def __init__(self, label: str, calls: list[tuple[str, str]]) -> None:
        self.label = label
        self.calls = calls  # shared sink: (label, step)

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        step = _classify(messages[0].content)
        self.calls.append((self.label, step))
        if step == "stage":
            return json.dumps({"next_stage": "stage_1", "reasoning": "stay"})
        if step == "think":
            return "Internal plan: respond with empathy."
        return json.dumps({"feeling_expressed": True})  # observe

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        self.calls.append((self.label, "execute"))
        yield "Thanks "
        yield "for sharing."


def _run_turn(enable_think: bool) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    main = RecordingLLM("MAIN", calls)
    fast = RecordingLLM("FAST", calls)
    pipeline = AgentPipeline(llm=main, fast_llm=fast)

    async def drive() -> None:
        gen = pipeline.process_turn(
            messages=[{"role": "user", "content": "I feel frustrated about politics"}],
            strategy_name="common_identity",
            study_id="test-session",
        )
        async for _ in gen:
            pass  # exhaust tokens + keep-alives

    prev = settings.enable_think
    settings.enable_think = enable_think
    # Force a fresh session (no DB row) regardless of test-import ordering:
    # patch the names build_session_state actually binds so get_conversation
    # can't leak another test file's MagicMock stub (which would yield a
    # non-None payload and break Stage()).
    try:
        with patch("app.agent.state.get_conversation", return_value=None), \
             patch("app.agent.state.get_user_party", return_value=None):
            asyncio.run(drive())
    finally:
        settings.enable_think = prev
    return calls


def test_observe_and_stage_run_on_fast_model() -> None:
    calls = _run_turn(enable_think=False)
    routing = {step: label for label, step in calls}  # step -> model label
    assert routing.get("observe") == "FAST", calls
    assert routing.get("stage") == "FAST", calls
    assert routing.get("execute") == "MAIN", calls


def test_think_runs_on_main_model() -> None:
    calls = _run_turn(enable_think=True)
    routing = {step: label for label, step in calls}
    assert routing.get("observe") == "FAST", calls
    assert routing.get("stage") == "FAST", calls
    assert routing.get("think") == "MAIN", calls
    assert routing.get("execute") == "MAIN", calls


if __name__ == "__main__":
    # Runnable directly: prints the per-step routing for both modes.
    print("enable_think=False ->", _run_turn(enable_think=False))
    print("enable_think=True  ->", _run_turn(enable_think=True))
