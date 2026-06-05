"""Unit tests for stage-transition correctness.

Two regressions are pinned here:

1. ORDERING — OBSERVE must run before the stage gate, so the gate sees signals
   extracted from the CURRENT user message (not the previous turn's).
2. FORWARD-ONLY — the controller must reject backward stage jumps proposed by
   the LLM, which would reset stage_turn_count and risk an endless loop.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_mock_col = MagicMock()
_db_stub = MagicMock()
_db_stub.user_docs = _mock_col
_db_stub.conversation_docs = _mock_col
sys.modules.setdefault("app.db.documents", _db_stub)

from app.agent.phases import StageController  # noqa: E402
from app.agent.pipeline import AgentPipeline  # noqa: E402
from app.agent.state import SessionState, Stage  # noqa: E402
from app.llm.base import LLMProvider, Message  # noqa: E402


# ---------------------------------------------------------------------------
# 2. FORWARD-ONLY guard (StageController.evaluate_transition)
# ---------------------------------------------------------------------------

class _ScriptedComplete(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, messages, system=None, temperature=0.7, max_tokens=2048) -> str:
        return self._response

    async def stream(self, messages, system=None, temperature=0.7, max_tokens=2048) -> AsyncIterator[str]:
        if False:  # pragma: no cover
            yield ""


def _state(stage: Stage, stage_turn_count: int = 5) -> SessionState:
    return SessionState(
        study_id="sid",
        strategy="common_identity",
        stage=stage,
        stage_turn_count=stage_turn_count,
    )


def test_backward_transition_is_rejected():
    state = _state(Stage.STAGE_3, stage_turn_count=4)
    controller = StageController(
        _ScriptedComplete(json.dumps({"next_stage": "stage_1", "reasoning": "regress"}))
    )

    result = asyncio.run(controller.evaluate_transition(state, "hello"))

    assert result == Stage.STAGE_3
    assert state.stage == Stage.STAGE_3
    assert state.stage_turn_count == 4  # not reset by a rejected transition


def test_forward_transition_is_allowed_and_resets_counter():
    state = _state(Stage.STAGE_3, stage_turn_count=4)
    controller = StageController(
        _ScriptedComplete(json.dumps({"next_stage": "stage_4", "reasoning": "advance"}))
    )

    result = asyncio.run(controller.evaluate_transition(state, "hello"))

    assert result == Stage.STAGE_4
    assert state.stage == Stage.STAGE_4
    assert state.stage_turn_count == 0  # reset on a real transition


def test_control_skip_stage1_to_stage4_is_allowed():
    """control legitimately skips STAGE_1 -> STAGE_4; the guard must not block it."""
    state = _state(Stage.STAGE_1, stage_turn_count=3)
    state.strategy = "control"
    controller = StageController(
        _ScriptedComplete(json.dumps({"next_stage": "stage_4", "reasoning": "skip"}))
    )

    result = asyncio.run(controller.evaluate_transition(state, "hello"))

    assert result == Stage.STAGE_4


# ---------------------------------------------------------------------------
# 1. ORDERING — the gate sees fresh signals from THIS turn (process_turn)
# ---------------------------------------------------------------------------

class _OrderingLLM(LLMProvider):
    """OBSERVE returns questions_answered=8; the stage gate, if it runs AFTER
    observe, must therefore see 8 in state.signals. We capture what the stage
    prompt actually contained to prove ordering."""

    def __init__(self) -> None:
        self.stage_prompt_saw_8: bool | None = None

    async def complete(self, messages, system=None, temperature=0.7, max_tokens=2048) -> str:
        prompt = messages[0].content
        if "next_stage" in prompt:
            # This is the stage-evaluation prompt. Record whether the freshly
            # observed signal (questions_answered: 8) is visible to it.
            self.stage_prompt_saw_8 = '"questions_answered": 8' in prompt
            return json.dumps({"next_stage": "stage_3", "reasoning": "done"})
        # Otherwise it's the OBSERVE prompt — report 8 answered questions.
        return json.dumps({"question_answers": {f"q{i}": "1" for i in range(1, 9)}})

    async def stream(self, messages, system=None, temperature=0.7, max_tokens=2048) -> AsyncIterator[str]:
        yield "ok"


def test_stage_gate_sees_freshly_observed_signals():
    llm = _OrderingLLM()
    pipeline = AgentPipeline(llm=llm)

    # Prior persisted state: misperception_correction in stage_2 with 7 answers.
    doc = {
        "study_id": "sid",
        "payload": {
            "stage": "stage_2",
            "strategy": "misperception_correction",
            "stage_turn_count": 3,
            "signals": {"question_answers": {f"q{i}": "1" for i in range(1, 8)}},
        },
    }
    messages = [
        {"role": "assistant", "content": "Question 8: ..."},
        {"role": "user", "content": "I think 30 percent, that is my honest guess"},
    ]

    async def _consume():
        async for _ in pipeline.process_turn(
            messages=messages, strategy_name="misperception_correction", study_id="sid"
        ):
            pass

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.pipeline.log_turn"), \
         patch("app.agent.pipeline.log_safety_event"), \
         patch("app.config.settings.enable_think", False):
        asyncio.run(_consume())

    assert llm.stage_prompt_saw_8 is True, (
        "stage gate ran before OBSERVE updated signals (saw stale questions_answered)"
    )
