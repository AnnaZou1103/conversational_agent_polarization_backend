"""Verifies that every message persisted to conversations.payload.messages
carries a timestamp, and that timestamps already on incoming history messages
are preserved rather than overwritten.

Drives a real turn through AgentPipeline.process_turn with a fake LLM and a
mocked DB layer (no real Mongo), then inspects the entry handed to
save_turn_log.
"""
from __future__ import annotations

import asyncio
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

from app.agent.pipeline import AgentPipeline  # noqa: E402
from app.llm.base import LLMProvider, Message  # noqa: E402


class _FakeLLM(LLMProvider):
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


_MESSAGES = [
    # Simulates history loaded from the DB: already has a timestamp.
    {
        "role": "user",
        "content": "I feel angry about politics these days",
        "timestamp": "2024-01-01T00:00:00+00:00",
    },
    {
        "role": "assistant",
        "content": "Tell me more about that.",
        "timestamp": "2024-01-01T00:00:01+00:00",
    },
    # Simulates the just-arrived message from chat.py: no timestamp pre-set
    # in this fixture, exercising the now_iso fallback.
    {"role": "user", "content": "I think the country is divided"},
]


def test_every_saved_message_has_a_timestamp():
    pipeline = AgentPipeline(llm=_FakeLLM())

    async def _consume():
        async for _ in pipeline.process_turn(
            messages=_MESSAGES, strategy_name="common_identity", study_id="sid"
        ):
            pass

    with patch("app.agent.state.get_conversation", return_value={"study_id": "sid"}), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.conversation_logger.save_turn_log") as mock_save, \
         patch("app.agent.conversation_logger.save_safety_event"):
        asyncio.run(_consume())

    assert mock_save.call_count == 1
    entry = mock_save.call_args.kwargs["entry"]
    saved_messages = entry["messages"]

    # 3 history messages + 1 new assistant reply.
    assert len(saved_messages) == 4
    for m in saved_messages:
        assert m.get("timestamp"), f"message missing timestamp: {m}"

    # Pre-existing timestamps must survive the round trip unchanged.
    assert saved_messages[0]["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert saved_messages[1]["timestamp"] == "2024-01-01T00:00:01+00:00"

    # The last user message (no timestamp in the fixture) and the new
    # assistant reply both get stamped with the turn's timestamp.
    assert saved_messages[2]["timestamp"] == entry["timestamp"]
    assert saved_messages[3]["role"] == "assistant"
    assert saved_messages[3]["timestamp"] == entry["timestamp"]
