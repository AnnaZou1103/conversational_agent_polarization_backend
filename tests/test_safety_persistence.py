"""Regression tests for safety-counter persistence across stateless requests.

The agent rebuilds SessionState from the stored `verdict` on every request.
A clean message resets the consecutive-reminder streak in memory, but unless
that reset is *persisted*, a clean message sitting between two reminders would
never break the streak — and unrelated, non-consecutive reminders would
accumulate to a wrongful termination. These tests pin the persistence behavior.
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub only the Mongo connection layer before importing app modules so no
# database connection is attempted. The real app.db.conversation / app.db.user
# modules import fine on top of this stub; we drive their behavior per-test by
# patching call-site names (get_conversation, log_safety_event, ...). We
# deliberately do NOT replace those modules with MagicMocks, which would leak
# into other test files via sys.modules.
_mock_col = MagicMock()
_db_stub = MagicMock()
_db_stub.user_docs = _mock_col
_db_stub.conversation_docs = _mock_col
sys.modules.setdefault("app.db.documents", _db_stub)

from app.agent.pipeline import AgentPipeline  # noqa: E402
from app.llm.base import LLMProvider, Message  # noqa: E402


class _FakeLLM(LLMProvider):
    """Returns valid stage/observe JSON for every complete() call and streams
    a single token. Identical responses keep the concurrent observe/stage calls
    order-independent."""

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        return '{"next_stage": "stage_1", "reasoning": "stay"}'

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        yield "ok"


# A clearly-clean final user message (recognized words, not a repeat).
_MESSAGES = [
    {"role": "user", "content": "I feel angry about politics these days"},
    {"role": "assistant", "content": "Tell me more about that."},
    {"role": "user", "content": "I think the country is divided and it makes me tired"},
]


def _run_clean_turn(doc: dict):
    """Drive one clean turn through process_turn with a mocked DB/LLM, and
    return the patched log_safety_event mock for inspection."""
    pipeline = AgentPipeline(llm=_FakeLLM())

    async def _consume():
        async for _ in pipeline.process_turn(
            messages=_MESSAGES, strategy_name="common_identity", study_id="sid"
        ):
            pass

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.pipeline.log_turn"), \
         patch("app.agent.pipeline.log_safety_event") as mock_log_safety:
        asyncio.run(_consume())

    return mock_log_safety


def test_clean_message_persists_streak_reset():
    """With a prior reminder streak, a clean message must persist a verdict
    whose consecutive_reminders is 0 — otherwise the streak never breaks."""
    doc = {
        "study_id": "sid",
        "verdict": {
            "action": "reminder",
            "consecutive_reminders": 2,
            "indecent_count": 0,
            "invalid_count": 1,
        },
    }

    mock_log_safety = _run_clean_turn(doc)

    assert mock_log_safety.call_count == 1, "clean message should persist the reset"
    persisted_verdict = mock_log_safety.call_args.args[2]
    assert persisted_verdict.consecutive_reminders == 0
    # Lifetime tallies must be carried forward, not zeroed.
    assert persisted_verdict.invalid_count == 1


def test_clean_message_without_streak_does_not_write():
    """No prior streak → no wasted DB round-trip on a clean turn."""
    doc = {"study_id": "sid"}  # no verdict → consecutive_reminders starts at 0

    mock_log_safety = _run_clean_turn(doc)

    assert mock_log_safety.call_count == 0


# ---------------------------------------------------------------------------
# Rejected turns must still land in the transcript
# ---------------------------------------------------------------------------
#
# A reminder turn returns early, before the pipeline's own log_turn call. The
# frontend replays its history on the NEXT turn, so an unlogged reminder is
# invisible only when it is the session's last exchange — which is exactly what
# happens when a participant is reminded and then leaves for the survey.

_GIBBERISH_MESSAGES = [
    {"role": "user", "content": "I feel angry about politics these days"},
    {"role": "assistant", "content": "Tell me more about that."},
    {"role": "user", "content": "asdkjhfaksdjhf"},
]


def _run_reminder_turn(doc: dict):
    """Drive one gibberish (reminder) turn and return the log_turn mock."""
    pipeline = AgentPipeline(llm=_FakeLLM())

    async def _consume():
        out = []
        async for tok in pipeline.process_turn(
            messages=_GIBBERISH_MESSAGES,
            strategy_name="common_identity",
            study_id="sid",
        ):
            out.append(tok)
        return out

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.pipeline.log_turn") as mock_log_turn, \
         patch("app.agent.pipeline.log_safety_event"):
        out = asyncio.run(_consume())

    return mock_log_turn, out


def test_reminder_turn_is_logged_as_a_turn():
    doc = {
        "study_id": "sid",
        "payload": {
            "stage": "stage_2",
            "strategy": "common_identity",
            "stage_turn_count": 2,
            "signals": {"feeling_expressed": True},
        },
    }

    mock_log_turn, out = _run_reminder_turn(doc)
    reminder_text = "".join(t for t in out if isinstance(t, str))

    assert mock_log_turn.call_count == 1, "the rejected turn must reach the transcript"
    args, kwargs = mock_log_turn.call_args
    state, system_prompt, logged_messages, response = args[1], args[2], args[3], args[4]

    # Tagged so analysis can separate rejected turns from generated ones, and
    # carrying the reminder itself as the assistant response.
    assert kwargs["turn_type"] == "safety_reminder"
    assert response == reminder_text
    assert system_prompt == "", "no system prompt ran on a rejected turn"
    assert [m.content for m in logged_messages] == [
        m["content"] for m in _GIBBERISH_MESSAGES
    ]

    # Logged AFTER the rollback, so a rejected turn still doesn't burn a stage
    # slot, and signals are untouched because OBSERVE never ran.
    assert state.stage_turn_count == 2
    assert state.stage.value == "stage_2"
    assert state.signals["feeling_expressed"] is True
