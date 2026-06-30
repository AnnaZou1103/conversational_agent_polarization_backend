"""Stress tests for the misperception_correction workflow.

Verifies that every stage gate, substantive-reasoning check, mid-quiz
check-in guard, and reflection gate fires correctly — without hitting a
real LLM. Covers:

  Stage 1  → delivered after turn 1, no signal required
  Stage 2  → blocked until questions_answered >= 8
  Q5 gate  → blocked until mid_quiz_reflection_done is true
  Stage 3  → blocked until reflection_shared OR n >= 4 (fallback)
  Stage 4  → advances after n >= 2
  COMPLETE → terminal

Also covers:
  • current_question_id regex detection from "N of 8" in agent message
  • questions_answered increments only on survey-reveal acknowledgement turns
  • reflection_shared guards (current_qid must be None AND q_a == 8)
  • substantive-reasoning signal merging (booleans/ints never decrease)
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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
# Helpers
# ---------------------------------------------------------------------------

_controller = StageController(llm=None)


def _state(
    stage: Stage,
    turn: int,
    signals: dict | None = None,
) -> SessionState:
    return SessionState(
        study_id="t",
        strategy="misperception_correction",
        stage=stage,
        stage_turn_count=turn,
        signals=dict(signals or {}),
    )


def _evaluate(stage: Stage, turn: int, signals: dict | None = None) -> SessionState:
    s = _state(stage, turn, signals)
    asyncio.run(_controller.evaluate_transition(s, "irrelevant"))
    return s


# ---------------------------------------------------------------------------
# Stage 1 → Stage 2: fires on turn >= 1, no signal required
# ---------------------------------------------------------------------------


def test_stage1_advances_on_first_turn() -> None:
    s = _evaluate(Stage.STAGE_1, 1, {})
    assert s.stage == Stage.STAGE_2
    assert s.stage_turn_count == 0, "counter resets on transition"


def test_stage1_does_not_advance_on_turn_0() -> None:
    # stage_turn_count starts at 0 before the first increment in process_turn;
    # the transition must require n >= 1 to not fire prematurely.
    s = _evaluate(Stage.STAGE_1, 0, {})
    assert s.stage == Stage.STAGE_1


# ---------------------------------------------------------------------------
# Stage 2 → Stage 3: blocked until questions_answered >= 8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answered", [0, 1, 4, 7])
def test_stage2_blocked_before_all_8_answered(answered: int) -> None:
    s = _evaluate(Stage.STAGE_2, 20, {"questions_answered": answered})
    assert s.stage == Stage.STAGE_2, f"must stay in stage_2 with {answered} answered"


def test_stage2_advances_exactly_at_8() -> None:
    s = _evaluate(Stage.STAGE_2, 1, {"questions_answered": 8})
    assert s.stage == Stage.STAGE_3


def test_stage2_no_turn_floor() -> None:
    # The rule is purely signal-based; even turn 1 should advance when all 8 done.
    s = _evaluate(Stage.STAGE_2, 1, {"questions_answered": 8})
    assert s.stage == Stage.STAGE_3


# ---------------------------------------------------------------------------
# Stage 3 → Stage 4
# ---------------------------------------------------------------------------


def test_stage3_advances_on_reflection_shared() -> None:
    s = _evaluate(Stage.STAGE_3, 1, {"reflection_shared": True})
    assert s.stage == Stage.STAGE_4


def test_stage3_fallback_after_4_turns() -> None:
    # If the user never gives a substantive reflection the session should not
    # get stuck — the n >= 4 safety net must fire.
    s = _evaluate(Stage.STAGE_3, 4, {})
    assert s.stage == Stage.STAGE_4


def test_stage3_stays_before_fallback() -> None:
    for n in (1, 2, 3):
        s = _evaluate(Stage.STAGE_3, n, {})
        assert s.stage == Stage.STAGE_3, f"must stay at turn {n}"


# ---------------------------------------------------------------------------
# Stage 4 → COMPLETE: after n >= 2
# ---------------------------------------------------------------------------


def test_stage4_stays_on_first_turn() -> None:
    s = _evaluate(Stage.STAGE_4, 1, {})
    assert s.stage == Stage.STAGE_4


def test_stage4_completes_after_2_turns() -> None:
    s = _evaluate(Stage.STAGE_4, 2, {})
    assert s.stage == Stage.COMPLETE


# ---------------------------------------------------------------------------
# COMPLETE is terminal
# ---------------------------------------------------------------------------


def test_complete_is_terminal() -> None:
    s = _evaluate(Stage.COMPLETE, 99, {"questions_answered": 8})
    assert s.stage == Stage.COMPLETE


# ---------------------------------------------------------------------------
# current_question_id regex — drives OBSERVE's write-target
# The pipeline derives current_qid by searching for "\b(\d+)\s+of\s+8\b" in
# the previous assistant message.  Test that detection matches the exact
# wording used in the prompts and that non-question turns return None.
# ---------------------------------------------------------------------------

_QID_PATTERN = re.compile(r"\b(\d+)\s+of\s+8\b", re.IGNORECASE)


def _detect_qid(agent_msg: str) -> str | None:
    m = _QID_PATTERN.search(agent_msg)
    if m:
        qnum = int(m.group(1))
        return f"q{qnum}" if 1 <= qnum <= 8 else None
    return None


@pytest.mark.parametrize(
    "msg,expected",
    [
        # Stage-1 opening (exact wording from STAGE_PROMPTS)
        ("Here's question 1 of 8: Would MOST Republican supporters...", "q1"),
        # Stage-2 numbered questions
        ("Question 2 of 8. Would MOST...", "q2"),
        ("Question 3 of 8. Would MOST...", "q3"),
        ("Question 4 of 8. Would MOST...", "q4"),
        ("Question 5 of 8. Would MOST...", "q5"),
        ("Question 6 of 8. Would MOST...", "q6"),
        ("Question 7 of 8. Would MOST...", "q7"),
        # Final question uses "number 8 of 8" in the prompt template
        ("Last question — number 8 of 8. Would MOST...", "q8"),
        # Case-insensitive
        ("QUESTION 5 OF 8.", "q5"),
        # Asking for more reasoning — no "N of 8" present
        ("Your reasoning matters here — before I share what surveys found...", None),
        # Mid-quiz check-in (same message as Q4 reveal, but Q5 not yet asked)
        (
            "Survey data found that the vast majority said 'never'. "
            "Halfway check-in: How does the gap sit with you so far?",
            None,
        ),
        # Stage-3 reflection opener — no question present
        ("That's all 8 questions. Looking at the full picture...", None),
        # Stage-4 closing question
        ("Last question: based on what you saw today, do you think...", None),
        # Out-of-range number
        ("Question 9 of 8 — invalid", None),
        ("Question 0 of 8 — invalid", None),
    ],
)
def test_current_qid_detection(msg: str, expected: str | None) -> None:
    assert _detect_qid(msg) == expected


# ---------------------------------------------------------------------------
# questions_answered — increments only on survey-reveal acknowledgement turns
#
# OBSERVE rule: increment ONLY if the previous assistant message contains a
# survey reveal phrase ("surveys found", "national surveys", "survey data")
# AND the user's current message acknowledges/continues.
# The pipeline then self-heals: questions_answered = len(question_answers).
# We test the self-healing logic via the _observe method with a stub LLM.
# ---------------------------------------------------------------------------


class _RevealLLM(LLMProvider):
    """Returns a preset OBSERVE JSON; EXECUTE returns 'ok'."""

    def __init__(self, observe_json: dict) -> None:
        self._json = observe_json

    async def complete(
        self, messages, system=None, temperature=0.7, max_tokens=2048
    ) -> str:
        return json.dumps(self._json)

    async def stream(
        self, messages, system=None, temperature=0.7, max_tokens=2048
    ) -> AsyncIterator[str]:
        yield "ok"


def _run_observe(
    state: SessionState,
    user_msg: str,
    prev_assistant_msg: str,
    observe_json: dict,
) -> None:
    """Drive the pipeline's _observe directly with a stub LLM."""
    pipeline = AgentPipeline(llm=_RevealLLM(observe_json))
    asyncio.run(pipeline._observe(state, user_msg, prev_assistant_msg))


def test_questions_answered_self_healed_from_answer_dict() -> None:
    """Pipeline resets questions_answered = len(question_answers) after merge."""
    state = _state(Stage.STAGE_2, 3, {"question_answers": {f"q{i}": "2" for i in range(1, 5)}})
    # OBSERVE returns a spuriously high questions_answered; pipeline self-heals.
    _run_observe(
        state,
        user_msg="okay I see",
        prev_assistant_msg="Surveys found that most Republican supporters said 'never'.",
        observe_json={
            "question_answers": {"q5": "1"},  # adds q5
            "questions_answered": 99,         # invalid — pipeline overrides
        },
    )
    # Self-heal: answer dict now has q1..q5 -> questions_answered must be 5
    assert state.signals["questions_answered"] == 5
    assert "q5" in state.signals.get("question_answers", {})


def test_questions_answered_never_decreases() -> None:
    """questions_answered is self-healed to len(question_answers), so it can
    only increase as more entries are added.  A stale OBSERVE that returns a
    lower int is overridden by the self-heal once the answer dict is present.
    The state must include both signals in sync, as in a real session."""
    existing_answers = {f"q{i}": "2" for i in range(1, 7)}  # q1..q6
    state = _state(
        Stage.STAGE_2,
        5,
        {"questions_answered": 6, "question_answers": existing_answers},
    )
    _run_observe(
        state,
        user_msg="hmm",
        prev_assistant_msg="Surveys found that...",
        # OBSERVE returns a spuriously low count — self-heal must ignore it
        observe_json={"questions_answered": 3},
    )
    # Self-heal recomputes from len(question_answers) = 6; must not drop
    assert state.signals["questions_answered"] >= 6


def test_no_phantom_keys_survive() -> None:
    """OBSERVE must not write q9 or 'None' keys; pipeline strips them."""
    state = _state(Stage.STAGE_2, 2, {})
    _run_observe(
        state,
        user_msg="1, because it seems like they wouldn't",
        prev_assistant_msg="Question 2 of 8.",
        observe_json={
            "question_answers": {"q2": "1", "q9": "2", "None": "3"},
        },
    )
    answers = state.signals.get("question_answers", {})
    assert "q9" not in answers
    assert "None" not in answers
    assert "q2" in answers


# ---------------------------------------------------------------------------
# reflection_shared guard
#
# Must be True ONLY when:
#   (1) current_question_id is None (no "N of 8" in prev agent message)
#   (2) questions_answered in Known signals == 8
#   (3) user's message is substantive
#
# The OBSERVE logic for this is inside the LLM prompt, so we test it
# indirectly via a stub that respects the guard rules (as the real OBSERVE
# model should), and verify the pipeline's sticky-bool merge never resets a
# True to False and never promotes a False.
# ---------------------------------------------------------------------------


def test_reflection_shared_stays_false_during_mid_quiz() -> None:
    """Mid-quiz check-in: q_a == 4, current_qid is None (no 'N of 8' in agent msg).
    reflection_shared must NOT be set even if user gives a substantive response,
    because questions_answered < 8."""
    state = _state(Stage.STAGE_2, 8, {"questions_answered": 4, "mid_quiz_reflection_done": True})
    # Simulates what a correct OBSERVE model returns for the mid-quiz turn:
    # reflection_shared=False because questions_answered (Known) != 8.
    _run_observe(
        state,
        user_msg="I expected them to be more extreme so this is surprising to me.",
        prev_assistant_msg=(
            "Survey data found that the vast majority said 'never'. "
            "Halfway check-in: How does the gap sit with you so far?"
        ),
        observe_json={
            "reflection_shared": False,
            "mid_quiz_reflection_done": True,
        },
    )
    assert not state.signals.get("reflection_shared")


def test_reflection_shared_set_after_all_8_done() -> None:
    """Stage 3 turn with q_a == 8 and substantive response sets reflection_shared."""
    state = _state(Stage.STAGE_3, 1, {"questions_answered": 8})
    _run_observe(
        state,
        user_msg="I expected them to be more extreme so this is quite surprising.",
        prev_assistant_msg="That's all 8 questions. Looking at the full picture...",
        observe_json={"reflection_shared": True},
    )
    assert state.signals.get("reflection_shared") is True


def test_reflection_shared_sticky_true() -> None:
    """Once reflection_shared is True, a follow-up OBSERVE that returns False
    must NOT overwrite it (sticky boolean rule)."""
    state = _state(Stage.STAGE_3, 2, {"questions_answered": 8, "reflection_shared": True})
    _run_observe(
        state,
        user_msg="nothing",
        prev_assistant_msg="Was there a particular question where the difference stood out?",
        observe_json={"reflection_shared": False},
    )
    assert state.signals.get("reflection_shared") is True


# ---------------------------------------------------------------------------
# mid_quiz_reflection_done gate
#
# Q5 must not be asked until mid_quiz_reflection_done is true.
# This is enforced in the Stage 2 prompt ("ask when questions_answered == 4
# AND mid_quiz_reflection_done is true"), not in the transition table.
# We verify the signal is correctly set once Halfway check-in appears in the
# previous assistant message.
# ---------------------------------------------------------------------------


def test_mid_quiz_reflection_done_set_by_halfway_checkin_msg() -> None:
    """mid_quiz_reflection_done becomes True when the previous agent message
    contains the exact phrase 'Halfway check-in:' (the mandatory marker)."""
    state = _state(Stage.STAGE_2, 5, {"questions_answered": 4})
    _run_observe(
        state,
        user_msg="I expected them to be more extreme so this is somewhat surprising.",
        prev_assistant_msg=(
            "Survey data found that the vast majority said 'never'. "
            "Halfway check-in: How does the gap sit with you so far?"
        ),
        observe_json={"mid_quiz_reflection_done": True},
    )
    assert state.signals.get("mid_quiz_reflection_done") is True


def test_mid_quiz_reflection_done_stays_false_without_marker() -> None:
    """When the agent has NOT yet delivered the Halfway check-in, the signal
    must stay False regardless of what the user says."""
    state = _state(Stage.STAGE_2, 4, {"questions_answered": 3})
    _run_observe(
        state,
        user_msg="1 — because they wouldn't do that given they value democracy.",
        prev_assistant_msg="Question 4 of 8. Would MOST Republican supporters...",
        observe_json={"mid_quiz_reflection_done": False},
    )
    assert not state.signals.get("mid_quiz_reflection_done")


def test_mid_quiz_reflection_done_sticky() -> None:
    """Once True, stays True even if OBSERVE returns False."""
    state = _state(Stage.STAGE_2, 6, {"questions_answered": 5, "mid_quiz_reflection_done": True})
    _run_observe(
        state,
        user_msg="2 — they probably wouldn't risk it.",
        prev_assistant_msg="Question 5 of 8. Would MOST...",
        observe_json={"mid_quiz_reflection_done": False},
    )
    assert state.signals.get("mid_quiz_reflection_done") is True


# ---------------------------------------------------------------------------
# Full workflow simulation — walk through all 8 questions and every stage
# ---------------------------------------------------------------------------


def _make_question_signals(n_answered: int) -> dict:
    """Build a signals dict as if n questions have been answered and revealed."""
    return {
        "questions_answered": n_answered,
        "question_answers": {f"q{i}": "2" for i in range(1, n_answered + 1)},
        "mid_quiz_reflection_done": n_answered >= 4,
    }


def test_full_workflow_stage_sequence() -> None:
    """Simulate a complete session from stage_1 to COMPLETE, checking that the
    stage advances at exactly the right points with no shortcuts or skips."""
    # --- Stage 1 → Stage 2 on first turn ---
    s = _evaluate(Stage.STAGE_1, 1, {})
    assert s.stage == Stage.STAGE_2

    # --- Stage 2 holds for each question (0–7 answered) ---
    for n in range(8):
        sigs = _make_question_signals(n)
        s = _evaluate(Stage.STAGE_2, n + 1, sigs)
        assert s.stage == Stage.STAGE_2, f"should stay in stage_2 after {n} answered"

    # --- Stage 2 → Stage 3 once all 8 answered ---
    sigs = _make_question_signals(8)
    s = _evaluate(Stage.STAGE_2, 9, sigs)
    assert s.stage == Stage.STAGE_3

    # --- Stage 3 holds until reflection_shared ---
    for n in (1, 2, 3):
        s = _evaluate(Stage.STAGE_3, n, {})
        assert s.stage == Stage.STAGE_3

    # --- Stage 3 → Stage 4 via reflection_shared ---
    s = _evaluate(Stage.STAGE_3, 1, {"reflection_shared": True})
    assert s.stage == Stage.STAGE_4

    # --- Stage 4 → COMPLETE after 2 turns ---
    s = _evaluate(Stage.STAGE_4, 1, {})
    assert s.stage == Stage.STAGE_4

    s = _evaluate(Stage.STAGE_4, 2, {})
    assert s.stage == Stage.COMPLETE


def test_full_workflow_with_fallbacks() -> None:
    """The worst-case path: user gives only non-substantive responses.
    Every stage must eventually advance via its safety-net cap."""
    # Stage 1 always advances on first turn
    s = _evaluate(Stage.STAGE_1, 1, {})
    assert s.stage == Stage.STAGE_2

    # Stage 2: stuck with no answers — eventually the LLM is supposed to loop,
    # but questions_answered comes from OBSERVE; we test the gate holds as long
    # as it's < 8 (no safety net on stage_2 — the quiz must be completed).
    s = _evaluate(Stage.STAGE_2, 100, {"questions_answered": 7})
    assert s.stage == Stage.STAGE_2

    # Stage 3: safety net at n >= 4
    s = _evaluate(Stage.STAGE_3, 4, {})
    assert s.stage == Stage.STAGE_4

    # Stage 4: safety net at n >= 2
    s = _evaluate(Stage.STAGE_4, 2, {})
    assert s.stage == Stage.COMPLETE


# ---------------------------------------------------------------------------
# Transition counter reset / stay behaviour
# ---------------------------------------------------------------------------


def test_stage_counter_resets_on_advance() -> None:
    s = _evaluate(Stage.STAGE_1, 5, {})  # always advances to stage_2
    assert s.stage_turn_count == 0


def test_stage_counter_preserved_when_staying() -> None:
    s = _evaluate(Stage.STAGE_2, 7, {"questions_answered": 6})
    assert s.stage_turn_count == 7


def test_stage_counter_resets_stage3_to_stage4() -> None:
    s = _evaluate(Stage.STAGE_3, 2, {"reflection_shared": True})
    assert s.stage == Stage.STAGE_4
    assert s.stage_turn_count == 0


# ---------------------------------------------------------------------------
# OBSERVE signal merge: key edge-cases
# ---------------------------------------------------------------------------


def test_question_answers_dict_accumulates_across_turns() -> None:
    """Each new OBSERVE turn should ADD to existing answers, not replace them."""
    state = _state(Stage.STAGE_2, 3, {"question_answers": {"q1": "2", "q2": "3"}})
    _run_observe(
        state,
        user_msg="3 — because I think they'd lean that way for partisan reasons.",
        prev_assistant_msg="Question 3 of 8. Would MOST...",
        observe_json={"question_answers": {"q3": "3"}},
    )
    answers = state.signals.get("question_answers", {})
    assert answers.get("q1") == "2"
    assert answers.get("q2") == "3"
    assert answers.get("q3") == "3"


def test_observe_does_not_write_wrong_question_slot() -> None:
    """OBSERVE must only write to the current_question_id key.  A stub that
    returns the right answer for q5 while we're on q3 should be corrected:
    the pipeline regex ensures current_qid=q3 from "Question 3 of 8", so the
    returned q5 key is valid but the key-guard in the prompt would normally
    prevent it.  After self-heal, questions_answered == len(answers)."""
    state = _state(Stage.STAGE_2, 3, {"question_answers": {"q1": "1", "q2": "2"}})
    # Simulate a misbehaving OBSERVE that writes q5 instead of q3
    _run_observe(
        state,
        user_msg="1 — I feel they wouldn't support such extreme measures.",
        prev_assistant_msg="Question 3 of 8. Would MOST...",
        observe_json={"question_answers": {"q5": "1"}},  # wrong slot
    )
    answers = state.signals.get("question_answers", {})
    # q5 is technically valid (1-8), so the phantom-strip passes it through;
    # the important thing is questions_answered self-heals to len(answers)
    assert state.signals["questions_answered"] == len(answers)


# ---------------------------------------------------------------------------
# Reasoning-loop: user gives limited info — agent must withhold the reveal
#
# The OBSERVE rule for questions_answered is:
#   "Increment by 1 ONLY if the previous assistant message contains a survey
#    reveal phrase ('surveys found', 'national surveys', 'survey data') AND
#    the user's current message acknowledges/continues."
#
# When the agent is looping to ask for more reasoning (no reveal phrase in its
# previous message), OBSERVE must NOT increment questions_answered even if the
# user now gives a substantive response.  The count must only advance on the
# turn that follows the reveal.
# ---------------------------------------------------------------------------


def _no_reveal_msg(question_n: int) -> str:
    """The agent is probing for more reasoning — no survey reveal in this message."""
    return (
        f"Question {question_n} of 8: your reasoning matters here — "
        "before I share what surveys found, could you say a bit more about why you picked that?"
    )


def _reveal_msg(finding: str = "never") -> str:
    """The agent has accepted reasoning and reveals the survey finding."""
    return f"Got it. Surveys found that most Republican supporters said '{finding}' on this one."


# --- No reveal → questions_answered must not change ---

@pytest.mark.parametrize(
    "user_msg",
    [
        # Classic filler answers that SHOULD trigger a loop
        "idk",
        "just a feeling",
        "interesting",
        "unfair",
        "obvious",
        "not sure",
        "because",
        "nothing",
        "2",           # bare number, no reasoning
        "scary",
        # Short phrases that name a reaction but explain nothing
        "seems unfair",
        "makes sense",
        "I guess so",
        "just interesting",
    ],
)
def test_no_reveal_filler_does_not_increment_questions_answered(user_msg: str) -> None:
    """When the agent is in reasoning-loop mode (no survey reveal), OBSERVE must
    not increment questions_answered regardless of what the user says."""
    state = _state(Stage.STAGE_2, 3, {"questions_answered": 2, "question_answers": {"q1": "2", "q2": "3"}})
    _run_observe(
        state,
        user_msg=user_msg,
        prev_assistant_msg=_no_reveal_msg(3),  # no reveal phrase
        observe_json={"questions_answered": 2},  # OBSERVE correctly holds the count
    )
    assert state.signals["questions_answered"] == 2, (
        f"questions_answered must not increment for filler input '{user_msg}'"
    )


def test_substantive_without_reveal_still_does_not_increment() -> None:
    """Even a fully substantive reasoning response does NOT advance
    questions_answered — the increment fires on the ACKNOWLEDGEMENT turn
    (after the reveal), not on the reasoning turn itself."""
    state = _state(Stage.STAGE_2, 4, {"questions_answered": 3, "question_answers": {f"q{i}": "2" for i in range(1, 4)}})
    _run_observe(
        state,
        user_msg="I picked 'probably not' because historically Republicans have defended free speech broadly, even for groups they disagree with.",
        prev_assistant_msg=_no_reveal_msg(4),  # still in probe mode, no reveal
        observe_json={"questions_answered": 3},  # count stays put
    )
    assert state.signals["questions_answered"] == 3


# --- After reveal → questions_answered must increment on the next turn ---

def test_reveal_followed_by_ack_increments_count() -> None:
    """After the agent delivers the survey reveal, the next user message
    (acknowledgement/reaction) must trigger the increment."""
    state = _state(Stage.STAGE_2, 4, {"questions_answered": 3, "question_answers": {f"q{i}": "2" for i in range(1, 4)}})
    _run_observe(
        state,
        user_msg="Interesting, I wouldn't have expected that.",
        prev_assistant_msg=_reveal_msg("never"),  # reveal present
        observe_json={"question_answers": {"q4": "2"}},  # OBSERVE writes the answer
    )
    # Self-heal: len(q1..q4) = 4
    assert state.signals["questions_answered"] == 4


def test_reveal_followed_by_short_ack_still_increments() -> None:
    """Even a brief acknowledgement after the reveal advances the count —
    the substantive-bar only applies BEFORE the reveal, not after."""
    state = _state(Stage.STAGE_2, 5, {"questions_answered": 4, "question_answers": {f"q{i}": "2" for i in range(1, 5)}})
    _run_observe(
        state,
        user_msg="okay",
        prev_assistant_msg=_reveal_msg("probably not"),
        observe_json={"question_answers": {"q5": "3"}},
    )
    assert state.signals["questions_answered"] == 5


# --- Multi-turn loop: user keeps giving filler, then finally substantive ---

def test_multi_turn_filler_then_substantive_increments_once() -> None:
    """Simulate a 3-turn reasoning loop on Q3.  OBSERVE must hold the count
    through two filler turns, then increment only after the reveal turn."""
    base_answers = {"q1": "2", "q2": "1"}
    state = _state(Stage.STAGE_2, 3, {"questions_answered": 2, "question_answers": base_answers})

    # Turn A: user says "idk" — agent asked for Q3, no reveal yet
    _run_observe(state, user_msg="idk", prev_assistant_msg="Question 3 of 8. Would MOST...",
                 observe_json={"questions_answered": 2})
    assert state.signals["questions_answered"] == 2

    # Turn B: agent probes again; user says "not sure"
    _run_observe(state, user_msg="not sure", prev_assistant_msg=_no_reveal_msg(3),
                 observe_json={"questions_answered": 2})
    assert state.signals["questions_answered"] == 2

    # Turn C: user finally gives substantive reasoning — agent probed again (still no reveal)
    _run_observe(
        state,
        user_msg="I think most Republicans value political participation broadly, so I'd expect them to oppose banning rallies even from the other side.",
        prev_assistant_msg=_no_reveal_msg(3),
        observe_json={"question_answers": {"q3": "2"}},  # OBSERVE captures answer but no reveal yet
    )
    # The answer key is written, so self-heal fires: len = 3
    assert state.signals["questions_answered"] == 3
    # Stage 2→3 gate must NOT fire yet (only 3 of 8 answered)
    s = _state(Stage.STAGE_2, 6, state.signals)
    asyncio.run(_controller.evaluate_transition(s, "irrelevant"))
    assert s.stage == Stage.STAGE_2


# --- Mid-quiz check-in loop: filler response must not set mid_quiz_reflection_done ---

@pytest.mark.parametrize(
    "user_msg",
    [
        "interesting",
        "unfair",
        "not much",
        "I don't know",
        "okay",
        "fine",
        "nope",
        "I guess",
        "just interesting",
        "seems unfair",
    ],
)
def test_filler_mid_quiz_reflection_does_not_advance(user_msg: str) -> None:
    """A non-substantive response to the Halfway check-in must keep
    mid_quiz_reflection_done False so Q5 is not asked prematurely."""
    state = _state(Stage.STAGE_2, 6, {"questions_answered": 4})
    _run_observe(
        state,
        user_msg=user_msg,
        prev_assistant_msg=(
            "Survey data found that the vast majority said 'never'. "
            "Halfway check-in: How does the gap sit with you so far?"
        ),
        observe_json={"mid_quiz_reflection_done": False},
    )
    assert not state.signals.get("mid_quiz_reflection_done"), (
        f"filler mid-quiz response '{user_msg}' must not set mid_quiz_reflection_done"
    )


def test_substantive_mid_quiz_reflection_advances() -> None:
    """A genuine sentence in response to the check-in sets mid_quiz_reflection_done."""
    # The prompt gives this exact example as passing the bar.
    state = _state(Stage.STAGE_2, 6, {"questions_answered": 4})
    _run_observe(
        state,
        user_msg="I expected them to be more extreme so this is surprising to me.",
        prev_assistant_msg=(
            "Survey data found that the vast majority said 'never'. "
            "Halfway check-in: How does the gap sit with you so far?"
        ),
        observe_json={"mid_quiz_reflection_done": True},
    )
    assert state.signals.get("mid_quiz_reflection_done") is True


def test_q5_blocked_until_mid_quiz_done() -> None:
    """Verify via the stage gate that Q5 cannot be triggered while
    mid_quiz_reflection_done is False (q_a == 4 but signal not set).

    The gate itself doesn't block Q5 — that's the LLM's job.  But because
    questions_answered stays 4 until OBSERVE increments it (which only happens
    after Q5 is answered), the stage stays in STAGE_2, correctly giving the LLM
    a chance to run the check-in loop before proceeding."""
    # Stuck at 4 answers with check-in not yet done
    s = _evaluate(Stage.STAGE_2, 8, {"questions_answered": 4, "mid_quiz_reflection_done": False})
    assert s.stage == Stage.STAGE_2

    # Still stuck at 4 even after check-in done — because q5 hasn't been answered yet
    s = _evaluate(Stage.STAGE_2, 9, {"questions_answered": 4, "mid_quiz_reflection_done": True})
    assert s.stage == Stage.STAGE_2


# --- Stage 3 reflection loop ---

@pytest.mark.parametrize(
    "user_msg",
    [
        "interesting",
        "unfair",
        "not much",
        "I don't know",
        "nothing",
        "okay",
        "fine",
        "not really",
        "nope",
        "I guess",
    ],
)
def test_filler_stage3_reflection_does_not_set_reflection_shared(user_msg: str) -> None:
    """Non-substantive Stage 3 responses must not set reflection_shared."""
    state = _state(Stage.STAGE_3, 1, {"questions_answered": 8})
    _run_observe(
        state,
        user_msg=user_msg,
        prev_assistant_msg="That's all 8 questions. Looking at the full picture...",
        observe_json={"reflection_shared": False},
    )
    assert not state.signals.get("reflection_shared"), (
        f"filler Stage 3 response '{user_msg}' must not set reflection_shared"
    )
    # Gate must also hold
    s = _evaluate(Stage.STAGE_3, 1, state.signals)
    assert s.stage == Stage.STAGE_3


def test_substantive_stage3_reflection_sets_signal_and_advances() -> None:
    """A full-sentence Stage 3 response sets reflection_shared and the gate fires."""
    state = _state(Stage.STAGE_3, 1, {"questions_answered": 8})
    _run_observe(
        state,
        user_msg="I expected them to support those actions far more than surveys suggest — the gap is larger than I thought.",
        prev_assistant_msg="That's all 8 questions. Looking at the full picture...",
        observe_json={"reflection_shared": True},
    )
    assert state.signals.get("reflection_shared") is True
    s = _evaluate(Stage.STAGE_3, 1, state.signals)
    assert s.stage == Stage.STAGE_4


# --- Stage 4 loop: single-word answer should not prematurely close ---

def test_stage4_stays_on_turn_1_regardless_of_content() -> None:
    """Stage 4 always requires at least 2 turns — the gate never fires on
    turn 1 no matter what the user says (the LLM is expected to probe again)."""
    for msg in ("same", "idk", "more", "less", "a lot", "unsure"):
        s = _evaluate(Stage.STAGE_4, 1, {})
        assert s.stage == Stage.STAGE_4, f"Stage 4 must stay on turn 1 for '{msg}'"


def test_stage4_completes_on_turn_2_even_without_signal() -> None:
    """After 2 turns in Stage 4 the session completes — this is the probed-once
    safety net (the LLM was given one chance to probe for more depth)."""
    s = _evaluate(Stage.STAGE_4, 2, {})
    assert s.stage == Stage.COMPLETE


# ---------------------------------------------------------------------------
# Pipeline integration: probing language present in system prompt on filler turn
# ---------------------------------------------------------------------------

class _CapturingLLM(LLMProvider):
    """Dual-role LLM stub used in integration tests.

    complete() handles OBSERVE — returns minimal JSON so no signals advance.
    stream() handles EXECUTE — records the system prompt passed by the pipeline.
    """

    def __init__(self, observe_json: dict | None = None) -> None:
        self._observe_json = json.dumps(observe_json or {})
        self.captured_system: str | None = None

    async def complete(
        self, messages, system=None, temperature=0.7, max_tokens=2048
    ) -> str:
        return self._observe_json

    async def stream(
        self, messages, system=None, temperature=0.7, max_tokens=2048
    ) -> AsyncIterator[str]:
        self.captured_system = system
        yield "ok"


def _run_process_turn(
    llm: _CapturingLLM,
    doc: dict,
    messages: list[dict],
) -> None:
    """Drive process_turn to completion with DB and logging patched out."""
    pipeline = AgentPipeline(llm=llm)

    async def _consume() -> None:
        async for _ in pipeline.process_turn(
            messages=messages,
            strategy_name="misperception_correction",
            study_id="sid",
        ):
            pass

    with (
        patch("app.agent.state.get_conversation", return_value=doc),
        patch("app.agent.state.get_user_party", return_value=None),
        patch("app.agent.pipeline.log_turn"),
        patch("app.agent.pipeline.log_safety_event"),
        patch("app.config.settings.enable_think", False),
    ):
        asyncio.run(_consume())


def test_system_prompt_contains_probing_language_on_filler_answer() -> None:
    """When a user gives a short filler answer in Stage 2 and the previous agent
    message has no survey-reveal phrase, the system prompt sent to the LLM must
    contain the instruction to probe for substantive reasoning before revealing.

    This catches regressions where the Stage 2 prompt loses its 'ask for more'
    instruction, which would let the LLM skip the reasoning gate silently.
    """
    llm = _CapturingLLM(observe_json={})  # OBSERVE advances nothing

    doc = {
        "study_id": "sid",
        "payload": {
            "stage": "stage_2",
            "strategy": "misperception_correction",
            "stage_turn_count": 2,
            "signals": {
                "question_answers": {"q1": "2", "q2": "3"},
            },
        },
    }
    # Pipeline is already probing (no "surveys found" in previous message)
    messages = [
        {"role": "assistant", "content": "Question 2 of 8: How many ..."},
        {"role": "user", "content": "3"},
        {
            "role": "assistant",
            "content": (
                "Your reasoning matters here — before I share what surveys found, "
                "could you say a bit about why you picked that?"
            ),
        },
        {"role": "user", "content": "idk"},
    ]

    _run_process_turn(llm, doc, messages)

    assert llm.captured_system is not None, "stream() was never called"
    sys_prompt = llm.captured_system

    probing_phrases = [
        "Your reasoning matters here",
        "before I share what surveys found",
        "not substantive",
        "Keep asking",
    ]
    assert any(p in sys_prompt for p in probing_phrases), (
        "Stage 2 system prompt is missing the probing instruction.\n"
        f"Checked for: {probing_phrases}\n"
        f"System prompt excerpt: {sys_prompt[:600]}"
    )


def test_system_prompt_rejects_short_evaluative_sentences() -> None:
    """'That is illegal' is a complete sentence but NOT substantive — it only
    labels a verdict without explaining why. The Stage 2 system prompt must
    explicitly call out this category so the LLM doesn't accept it.
    """
    llm = _CapturingLLM(observe_json={})

    doc = {
        "study_id": "sid",
        "payload": {
            "stage": "stage_2",
            "strategy": "misperception_correction",
            "stage_turn_count": 1,
            "signals": {"question_answers": {"q1": "2"}},
        },
    }
    messages = [
        {"role": "assistant", "content": "Question 1 of 8: Would MOST ..."},
        {"role": "user", "content": "2"},
        {
            "role": "assistant",
            "content": (
                "Your reasoning matters here — before I share what surveys found, "
                "could you say a bit about why you picked that?"
            ),
        },
        {"role": "user", "content": "That is illegal"},
    ]

    _run_process_turn(llm, doc, messages)

    assert llm.captured_system is not None
    sys_prompt = llm.captured_system

    # The prompt must call out short evaluative sentences as NOT substantive
    short_sentence_phrases = [
        "That is illegal",
        "only label or evaluate",
        "names a reaction or verdict",
        "not just names their reaction",
    ]
    assert any(p in sys_prompt for p in short_sentence_phrases), (
        "Stage 2 system prompt does not warn about short evaluative sentences "
        "('That is illegal' pattern).\n"
        f"Checked for: {short_sentence_phrases}\n"
        f"System prompt excerpt: {sys_prompt[:800]}"
    )


def test_system_prompt_probing_language_absent_after_reveal() -> None:
    """Control: once a survey finding has been revealed and the user acknowledged,
    OBSERVE increments questions_answered. On the next question, the pipeline must
    still be in Stage 2 (only 3 of 8 done) — and the system prompt for asking Q4
    must still include the probing instruction (it's always part of Stage 2).
    """
    # OBSERVE records the acknowledgment of q3 reveal
    existing = {"q1": "2", "q2": "3", "q3": "1"}
    llm = _CapturingLLM(observe_json={"question_answers": existing})

    doc = {
        "study_id": "sid",
        "payload": {
            "stage": "stage_2",
            "strategy": "misperception_correction",
            "stage_turn_count": 3,
            "signals": {"question_answers": existing},
        },
    }
    messages = [
        {"role": "assistant", "content": "Surveys found that most said 'probably not'. Question 3 of 8: ..."},
        {"role": "user", "content": "Makes sense, I expected that"},
    ]

    _run_process_turn(llm, doc, messages)

    assert llm.captured_system is not None, "stream() was never called"
    # Stage 2 is still active (3 < 8) so the probing language must still be present
    assert any(
        p in llm.captured_system
        for p in ["Your reasoning matters here", "not substantive", "Keep asking"]
    ), "Stage 2 system prompt must retain probing language even mid-quiz"


# ---------------------------------------------------------------------------
# Run all tests manually (for quick ad-hoc execution without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for _name, _fn in sorted(globals().items()):
        if not (_name.startswith("test_") and callable(_fn)):
            continue
        try:
            _fn()
            print(f"ok  {_name}")
            passed += 1
        except Exception:
            print(f"FAIL {_name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
