"""Unit tests for stage-transition correctness.

StageController.evaluate_transition is now DETERMINISTIC — it applies the
threshold rules in app.agent.phases._TRANSITIONS directly to the signals that
OBSERVE already extracted, with no LLM call. These tests pin:

  * the full transition table for every condition (and the premature-advance
    cases that must NOT fire);
  * the person_label / person_name naming fix for personal_narrative;
  * forward-only / COMPLETE-terminal invariants;
  * that no LLM is consulted;
  * ORDERING — OBSERVE runs before the gate, so the gate sees signals from the
    CURRENT user message (pipeline-level);
  * POST-COMPLETION short-circuit — a message after stage=complete runs nothing.
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
from app.llm.base import LLMProvider  # noqa: E402

_controller = StageController(llm=None)  # llm must be unused


def _evaluate(strategy: str, stage: Stage, turn: int, signals: dict) -> SessionState:
    """Run one transition evaluation and return the mutated state."""
    state = SessionState(
        study_id="t",
        strategy=strategy,
        stage=stage,
        stage_turn_count=turn,
        signals=dict(signals),
    )
    asyncio.run(_controller.evaluate_transition(state, "irrelevant user message"))
    return state


# ---------------------------------------------------------------------------
# Deterministic transition table — per condition
# ---------------------------------------------------------------------------

def test_common_identity_full_path() -> None:
    s = _evaluate("common_identity", Stage.STAGE_1, 1, {"feeling_expressed": True})
    assert s.stage == Stage.STAGE_2
    assert s.stage_turn_count == 0  # reset on transition

    s = _evaluate("common_identity", Stage.STAGE_2, 2,
                  {"media_distortion_acknowledged": True})
    assert s.stage == Stage.STAGE_3

    s = _evaluate("common_identity", Stage.STAGE_3, 1,
                  {"common_identity_described": True})
    assert s.stage == Stage.STAGE_4

    s = _evaluate("common_identity", Stage.STAGE_4, 1, {})
    assert s.stage == Stage.COMPLETE


def test_common_identity_blocks_without_signal() -> None:
    # Signal missing -> stay, even with plenty of turns.
    s = _evaluate("common_identity", Stage.STAGE_1, 5, {})
    assert s.stage == Stage.STAGE_1
    assert s.stage_turn_count == 5  # not reset when staying


def test_common_identity_blocks_without_turns() -> None:
    # Signal present but turn threshold (>= 2) not met.
    s = _evaluate("common_identity", Stage.STAGE_2, 1,
                  {"media_distortion_acknowledged": True})
    assert s.stage == Stage.STAGE_2


def test_personal_narrative_uses_person_label_not_person_name() -> None:
    # person_label present -> advances.
    s = _evaluate("personal_narrative", Stage.STAGE_1, 1, {"person_label": "my uncle"})
    assert s.stage == Stage.STAGE_2

    # The legacy `person_name` key must NOT trigger anything — it never exists in
    # real signals (OBSERVE emits person_label). This is the masked bug the LLM
    # gate used to paper over.
    s = _evaluate("personal_narrative", Stage.STAGE_1, 1, {"person_name": "my uncle"})
    assert s.stage == Stage.STAGE_1


def test_personal_narrative_details_and_origins() -> None:
    s = _evaluate("personal_narrative", Stage.STAGE_2, 2, {"person_details_count": 2})
    assert s.stage == Stage.STAGE_3

    s = _evaluate("personal_narrative", Stage.STAGE_2, 2, {"person_details_count": 1})
    assert s.stage == Stage.STAGE_2  # below threshold

    s = _evaluate("personal_narrative", Stage.STAGE_3, 1, {"origins_explored": True})
    assert s.stage == Stage.STAGE_4


def test_misperception_quiz_gate() -> None:
    s = _evaluate("misperception_correction", Stage.STAGE_1, 1, {})
    assert s.stage == Stage.STAGE_2  # turn-only

    s = _evaluate("misperception_correction", Stage.STAGE_2, 9,
                  {"questions_answered": 7})
    assert s.stage == Stage.STAGE_2  # holds until all 8 answered
    s = _evaluate("misperception_correction", Stage.STAGE_2, 9,
                  {"questions_answered": 8})
    assert s.stage == Stage.STAGE_3

    s = _evaluate("misperception_correction", Stage.STAGE_3, 1,
                  {"reflection_shared": True})
    assert s.stage == Stage.STAGE_4


def test_control_skips_to_stage_4() -> None:
    for strategy in ("control", "control_politics"):
        s = _evaluate(strategy, Stage.STAGE_1, 2, {})
        assert s.stage == Stage.STAGE_1, strategy  # needs >= 3
        s = _evaluate(strategy, Stage.STAGE_1, 3, {})
        assert s.stage == Stage.STAGE_4, strategy  # skips 2 and 3 legitimately
        s = _evaluate(strategy, Stage.STAGE_4, 1, {})
        assert s.stage == Stage.COMPLETE, strategy


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_forward_transition_resets_counter() -> None:
    s = _evaluate("common_identity", Stage.STAGE_3, 4,
                  {"common_identity_described": True})
    assert s.stage == Stage.STAGE_4
    assert s.stage_turn_count == 0


def test_stay_does_not_reset_counter() -> None:
    s = _evaluate("common_identity", Stage.STAGE_3, 4, {})  # no signal
    assert s.stage == Stage.STAGE_3
    assert s.stage_turn_count == 4


def test_complete_is_terminal() -> None:
    s = _evaluate("common_identity", Stage.COMPLETE, 99, {"feeling_expressed": True})
    assert s.stage == Stage.COMPLETE


def test_no_llm_is_called() -> None:
    # A real LLM call would invoke self.llm.complete; wire a sentinel that blows
    # up if touched, then drive a real transition.
    sentinel = MagicMock(side_effect=AssertionError("LLM must not be called"))
    controller = StageController(llm=None)
    controller.llm = MagicMock(complete=sentinel)
    state = SessionState(study_id="t", strategy="common_identity",
                         stage=Stage.STAGE_1, stage_turn_count=1,
                         signals={"feeling_expressed": True})
    asyncio.run(controller.evaluate_transition(state, "msg"))
    assert state.stage == Stage.STAGE_2
    sentinel.assert_not_called()


# ---------------------------------------------------------------------------
# ORDERING — the gate sees fresh signals from THIS turn (process_turn)
# ---------------------------------------------------------------------------

class _ObserveLLM(LLMProvider):
    """OBSERVE reports all 8 quiz questions answered this turn. With OBSERVE
    running before the deterministic gate, questions_answered becomes 8 and the
    gate must advance stage_2 -> stage_3 on THIS turn."""

    async def complete(self, messages, system=None, temperature=0.7,
                       max_tokens=2048) -> str:
        # Only OBSERVE calls complete now (no LLM stage eval). Report q1..q8.
        return json.dumps({"question_answers": {f"q{i}": "1" for i in range(1, 9)}})

    async def stream(self, messages, system=None, temperature=0.7,
                     max_tokens=2048) -> AsyncIterator[str]:
        yield "ok"


def test_stage_gate_sees_freshly_observed_signals() -> None:
    llm = _ObserveLLM()
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

    captured: dict = {}

    def _capture_build(stage, strategy, state):
        captured["stage"] = stage
        return "SYSTEM"

    async def _consume():
        async for _ in pipeline.process_turn(
            messages=messages, strategy_name="misperception_correction",
            study_id="sid",
        ):
            pass

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.pipeline.build_system_prompt", side_effect=_capture_build), \
         patch("app.agent.pipeline.log_turn"), \
         patch("app.agent.pipeline.log_safety_event"), \
         patch("app.config.settings.enable_think", False):
        asyncio.run(_consume())

    # If the gate had run on stale signals (7 answered) it would have stayed in
    # stage_2; seeing stage_3 proves OBSERVE updated signals first.
    assert captured.get("stage") == Stage.STAGE_3, (
        "stage gate ran before OBSERVE updated signals (saw stale questions_answered)"
    )


# ---------------------------------------------------------------------------
# POST-COMPLETION re-entry short-circuit (process_turn)
# ---------------------------------------------------------------------------

class _CountingLLM(LLMProvider):
    """Records whether the pipeline made any LLM call at all."""

    def __init__(self) -> None:
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, messages, system=None, temperature=0.7,
                       max_tokens=2048) -> str:
        self.complete_calls += 1
        return json.dumps({"question_answers": {}})

    async def stream(self, messages, system=None, temperature=0.7,
                     max_tokens=2048) -> AsyncIterator[str]:
        self.stream_calls += 1
        yield "should-not-happen"


def test_post_completion_message_short_circuits() -> None:
    """Once a prior turn persisted stage=complete, a new user message must NOT
    run the pipeline or generate another agent turn — it returns a brief ack."""
    from app.agent.pipeline import COMPLETION_REENTRY_MESSAGE

    llm = _CountingLLM()
    pipeline = AgentPipeline(llm=llm)

    doc = {
        "study_id": "sid",
        "payload": {
            "stage": "complete",
            "strategy": "common_identity",
            "stage_turn_count": 1,
            "signals": {},
        },
    }
    messages = [
        {"role": "assistant", "content": "You're all done!"},
        {"role": "user", "content": "ok"},
    ]

    async def _consume():
        out = []
        async for tok in pipeline.process_turn(
            messages=messages, strategy_name="common_identity", study_id="sid"
        ):
            out.append(tok)
        return out

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.pipeline.log_turn") as mock_log_turn, \
         patch("app.agent.pipeline.log_safety_event"):
        out = asyncio.run(_consume())

    assert "".join(t for t in out if isinstance(t, str)) == COMPLETION_REENTRY_MESSAGE
    assert llm.complete_calls == 0, "no OBSERVE/STAGE/THINK calls after completion"
    assert llm.stream_calls == 0, "no response generation after completion"
    assert mock_log_turn.call_count == 0, "re-entry pings should not be logged as turns"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all stage-transition tests passed")
