"""Unit tests for AgentPipeline._observe.

Covers two things:

1. DISPATCH — each strategy uses its own per-condition prompt template from
   OBSERVE_PROMPTS, not the old monolithic prompt.
2. ACCUMULATION — how extracted signals merge into state.signals across
   turns, by value type (bool, int, list, dict, str). These tests double as
   executable documentation of the merge semantics in
   AgentPipeline._observe.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub the entire DB layer before any app module is imported so no MongoDB
# connection is attempted during collection or test execution.
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
from app.agent.state import SessionState, Stage  # noqa: E402
from app.llm.base import LLMProvider, Message  # noqa: E402


class ScriptedLLM(LLMProvider):
    """LLM that returns a pre-scripted queue of responses.

    Records every call so tests can inspect the prompt that was sent.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.complete_calls: list[str] = []

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        self.complete_calls.append(messages[0].content)
        return self._responses.pop(0)

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        if False:
            yield ""  # pragma: no cover


def _state(strategy: str, signals: dict | None = None) -> SessionState:
    return SessionState(
        study_id="test",
        strategy=strategy,
        stage=Stage.STAGE_2,
        political_party="democrat",
        turn_count=1,
        stage_turn_count=1,
        signals=signals or {},
    )


def _observe(state: SessionState, responses: list[str], user_msgs: list[str]) -> ScriptedLLM:
    """Run one _observe call per (response, user_msg) pair, sequentially."""
    mock = ScriptedLLM(responses)
    pipeline = AgentPipeline(llm=mock)
    for msg in user_msgs:
        asyncio.run(pipeline._observe(state, msg))
    return mock


# ---------------------------------------------------------------------------
# 1. DISPATCH — correct per-strategy prompt is used
# ---------------------------------------------------------------------------

def test_dispatch_common_identity_uses_common_identity_schema() -> None:
    state = _state("common_identity")
    mock = _observe(state, [json.dumps({"feeling_expressed": True})], ["I feel frustrated"])
    prompt = mock.complete_calls[0]
    assert '"feeling_expressed"' in prompt
    assert '"person_label"' not in prompt
    assert '"topics_shared"' not in prompt


def test_dispatch_personal_narrative_uses_person_schema() -> None:
    state = _state("personal_narrative")
    mock = _observe(state, [json.dumps({"person_label": "my uncle"})], ["my uncle Dave"])
    prompt = mock.complete_calls[0]
    assert '"person_label"' in prompt
    assert '"feeling_expressed"' not in prompt


def test_dispatch_control_uses_topics_schema() -> None:
    state = _state("control")
    mock = _observe(state, [json.dumps({"topics_shared": ["tired"]})], ["I'm exhausted"])
    prompt = mock.complete_calls[0]
    assert '"topics_shared"' in prompt
    assert '"person_label"' not in prompt


def test_dispatch_misperception_uses_quiz_schema() -> None:
    state = _state("misperception_correction")
    mock = _observe(state, [json.dumps({"questions_answered": 0})], ["yes let's go"])
    prompt = mock.complete_calls[0]
    assert '"mid_quiz_reflection_done"' in prompt
    assert '"questions_answered"' in prompt
    assert '"person_label"' not in prompt


# ---------------------------------------------------------------------------
# 2. ACCUMULATION — behavior per value type
# ---------------------------------------------------------------------------
#
# Summary of merge rules (see AgentPipeline._observe):
#
#   type   │ behavior
#   ───────┼────────────────────────────────────────────────────────────────
#   None   │ skipped (no change)
#   bool   │ sticky: only stored if True; once True, stays True forever
#   int    │ monotonic max: state[key] = max(old, new)
#   list   │ accumulates: old list + new items not already in old (order-preserving)
#   dict   │ key-wise merge: {**existing, **new}  → new keys of same id overwrite
#   str    │ latest non-empty overwrites; "" does NOT overwrite
#   other  │ plain overwrite
# ---------------------------------------------------------------------------


def test_bool_signal_is_sticky_once_true() -> None:
    """Once a bool signal is True, later False values don't flip it back."""
    state = _state("common_identity")
    _observe(
        state,
        [
            json.dumps({"feeling_expressed": True}),
            json.dumps({"feeling_expressed": False}),
        ],
        ["I feel angry", "nvm"],
    )
    assert state.signals["feeling_expressed"] is True


def test_bool_signal_false_is_never_stored() -> None:
    """A bool that's only ever False never appears in signals."""
    state = _state("common_identity")
    _observe(
        state,
        [json.dumps({"feeling_expressed": False})],
        ["not sure"],
    )
    assert "feeling_expressed" not in state.signals


def test_int_signal_is_monotonic_max() -> None:
    """Ints accumulate via max() — regressions from the LLM are ignored."""
    state = _state("personal_narrative")
    _observe(
        state,
        [
            json.dumps({"person_details_count": 3}),
            json.dumps({"person_details_count": 1}),  # LLM regression
            json.dumps({"person_details_count": 5}),
        ],
        ["turn1", "turn2", "turn3"],
    )
    assert state.signals["person_details_count"] == 5


def test_list_signal_accumulates_across_turns_with_dedupe() -> None:
    """Lists merge: old items retained, new items appended, duplicates removed."""
    state = _state("personal_narrative")
    _observe(
        state,
        [
            json.dumps({"person_traits": ["stubborn", "warm"]}),
            json.dumps({"person_traits": ["warm", "loyal"]}),     # warm repeated
            json.dumps({"person_traits": ["funny"]}),              # LLM dropped old ones
        ],
        ["t1", "t2", "t3"],
    )
    assert state.signals["person_traits"] == ["stubborn", "warm", "loyal", "funny"]


def test_empty_list_does_not_overwrite_accumulated_list() -> None:
    """An empty list from the LLM must not wipe prior accumulated entries."""
    state = _state("control", signals={"topics_shared": ["stressed", "tired"]})
    _observe(
        state,
        [json.dumps({"topics_shared": []})],
        ["..."],
    )
    assert state.signals["topics_shared"] == ["stressed", "tired"]


def test_dict_signal_merges_keywise() -> None:
    """Dict signals accumulate by key; new keys add, same keys overwrite."""
    state = _state("misperception_correction")
    _observe(
        state,
        [
            json.dumps({"question_answers": {"q1": 3}}),
            json.dumps({"question_answers": {"q2": 2}}),
            json.dumps({"question_answers": {"q1": 4}}),  # user changed answer
        ],
        ["q1=probably", "q2=probably not", "q1=definitely actually"],
    )
    assert state.signals["question_answers"] == {"q1": 4, "q2": 2}


def test_str_signal_latest_non_empty_wins() -> None:
    """Non-empty strings overwrite; empty strings are ignored."""
    state = _state("common_identity")
    _observe(
        state,
        [
            json.dumps({"user_feeling_text": "frustrated"}),
            json.dumps({"user_feeling_text": ""}),          # must not overwrite
            json.dumps({"user_feeling_text": "exhausted"}),
        ],
        ["t1", "t2", "t3"],
    )
    assert state.signals["user_feeling_text"] == "exhausted"


def test_none_is_skipped() -> None:
    """An explicit null for a field must not overwrite existing state."""
    state = _state("personal_narrative", signals={"person_label": "my uncle"})
    _observe(
        state,
        [json.dumps({"person_label": None})],
        ["hmm"],
    )
    assert state.signals["person_label"] == "my uncle"


# ---------------------------------------------------------------------------
# 3. RESILIENCE
# ---------------------------------------------------------------------------

def test_invalid_json_preserves_existing_state() -> None:
    """If the LLM returns unparseable text, signals must not change."""
    state = _state("common_identity", signals={"feeling_expressed": True})
    _observe(
        state,
        ["this is not json at all"],
        ["..."],
    )
    assert state.signals == {"feeling_expressed": True}


def test_misperception_questions_answered_accumulates() -> None:
    """Integration-style: quiz progression updates both int and dict signals."""
    state = _state("misperception_correction")
    _observe(
        state,
        [
            json.dumps({"questions_answered": 1, "question_answers": {"q1": 3}}),
            json.dumps({"questions_answered": 2, "question_answers": {"q2": 2}}),
            json.dumps({"questions_answered": 3, "question_answers": {"q3": 4}}),
        ],
        ["q1 answer", "q2 answer", "q3 answer"],
    )
    assert state.signals["questions_answered"] == 3
    assert state.signals["question_answers"] == {"q1": 3, "q2": 2, "q3": 4}
