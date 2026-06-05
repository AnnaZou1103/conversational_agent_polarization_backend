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
