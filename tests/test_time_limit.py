"""Tests for the session time-limit feature in app/agent/pipeline.py.

Covers:
- _minutes_since_session_start (pure function)
- pipeline: time limit reached → stage=complete, verdict=time_limit, user screened
- pipeline: time limit not reached → normal flow, no screening
- pipeline: disabled (max_session_minutes=0) → no screening
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_mock_col = MagicMock()
_db_stub = MagicMock()
_db_stub.user_docs = _mock_col
_db_stub.conversation_docs = _mock_col
sys.modules.setdefault("app.db.documents", _db_stub)

from app.agent.pipeline import AgentPipeline, _minutes_since_session_start  # noqa: E402
from app.llm.base import LLMProvider, Message  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ago(minutes: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _make_messages(age_minutes: float) -> list[dict]:
    return [
        {"role": "user", "content": "Hello", "timestamp": _ago(age_minutes)},
        {"role": "assistant", "content": "Hi there!", "timestamp": _ago(age_minutes - 1)},
        {"role": "user", "content": "Still here"},
    ]


class _FakeLLM(LLMProvider):
    async def complete(self, messages, system=None, temperature=0.7, max_tokens=2048) -> str:
        return '{"next_stage": "complete", "reasoning": "done"}'

    async def stream(self, messages, system=None, temperature=0.7, max_tokens=2048) -> AsyncIterator[str]:
        yield "closing reply"


def _run(messages: list[dict], max_session_minutes: int, advance_mock=None) -> list[str]:
    pipeline = AgentPipeline(llm=_FakeLLM())
    chunks: list[str] = []

    with patch("app.agent.state.get_conversation", return_value={"study_id": "test123"}), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.conversation_logger.save_turn_log"), \
         patch("app.agent.conversation_logger.save_safety_event"), \
         patch("app.agent.pipeline.settings") as mock_settings, \
         patch("app.agent.pipeline.advance_user_state", new=advance_mock or MagicMock()):
        mock_settings.max_session_minutes = max_session_minutes
        mock_settings.conversations_dir = "conversations"
        mock_settings.enable_think = False
        mock_settings.fast_llm_model = ""

        async def _consume():
            async for chunk in pipeline.process_turn(
                messages=messages, strategy_name="common_identity", study_id="test123"
            ):
                chunks.append(chunk)

        asyncio.run(_consume())

    return chunks


# ---------------------------------------------------------------------------
# _minutes_since_session_start — unit tests (pure function)
# ---------------------------------------------------------------------------

def test_returns_none_when_no_timestamps():
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    assert _minutes_since_session_start(messages) is None


def test_returns_none_for_empty_list():
    assert _minutes_since_session_start([]) is None


def test_measures_from_first_message_with_timestamp():
    messages = [
        {"role": "user", "content": "hello", "timestamp": _ago(30)},
        {"role": "assistant", "content": "hi", "timestamp": _ago(10)},
        {"role": "user", "content": "more", "timestamp": _ago(5)},
    ]
    elapsed = _minutes_since_session_start(messages)
    assert elapsed is not None
    assert 29 < elapsed < 31


def test_handles_z_suffix():
    ts = (datetime.now(timezone.utc) - timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
    messages = [{"role": "user", "content": "hi", "timestamp": ts}]
    elapsed = _minutes_since_session_start(messages)
    assert elapsed is not None
    assert 44 < elapsed < 46


def test_handles_naive_timestamp():
    naive = (datetime.now(timezone.utc) - timedelta(minutes=20)).replace(tzinfo=None).isoformat()
    messages = [{"role": "user", "content": "hi", "timestamp": naive}]
    elapsed = _minutes_since_session_start(messages)
    assert elapsed is not None
    assert 19 < elapsed < 21


def test_returns_none_for_invalid_timestamp():
    messages = [{"role": "user", "content": "hi", "timestamp": "not-a-date"}]
    assert _minutes_since_session_start(messages) is None


def test_skips_messages_without_timestamp():
    messages = [
        {"role": "user", "content": "no timestamp here"},
        {"role": "user", "content": "has timestamp", "timestamp": _ago(15)},
    ]
    elapsed = _minutes_since_session_start(messages)
    assert elapsed is not None
    assert 14 < elapsed < 16


def test_ignores_assistant_timestamp_before_first_user_message():
    messages = [
        {"role": "assistant", "content": "Welcome!", "timestamp": _ago(90)},
        {"role": "user", "content": "Hello", "timestamp": _ago(15)},
    ]
    elapsed = _minutes_since_session_start(messages)
    assert elapsed is not None
    assert 14 < elapsed < 16


# ---------------------------------------------------------------------------
# Pipeline: time limit reached
# ---------------------------------------------------------------------------

def test_time_limit_reached_calls_advance_user_state():
    """When time limit is hit, advance_user_state must be called with screened=True."""
    advance_mock = MagicMock()
    _run(_make_messages(age_minutes=90), max_session_minutes=60, advance_mock=advance_mock)

    advance_mock.assert_called_once()
    kwargs = advance_mock.call_args.kwargs
    assert kwargs["study_id"] == "test123"
    assert kwargs["next_state"].screened is True
    assert kwargs["next_state"].state == "complete"


def test_time_limit_reached_saves_time_limit_verdict():
    """When time limit is hit, save_safety_event must record category=time_limit."""
    saved_verdicts: list[dict] = []

    pipeline = AgentPipeline(llm=_FakeLLM())

    # Patch at the DB level: save_safety_event(study_id=..., verdict=...)
    def _capture(study_id, verdict):
        saved_verdicts.append(verdict)

    with patch("app.agent.state.get_conversation", return_value={"study_id": "test123"}), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.conversation_logger.save_turn_log"), \
         patch("app.agent.conversation_logger.save_safety_event", side_effect=_capture), \
         patch("app.agent.pipeline.settings") as mock_settings, \
         patch("app.agent.pipeline.advance_user_state"):
        mock_settings.max_session_minutes = 60
        mock_settings.conversations_dir = "conversations"
        mock_settings.enable_think = False
        mock_settings.fast_llm_model = ""

        asyncio.run(_exhaust(pipeline, _make_messages(age_minutes=90)))

    assert any(v["category"] == "time_limit" for v in saved_verdicts)
    assert any(v["action"] == "terminate" for v in saved_verdicts)


async def _exhaust(pipeline, messages):
    async for _ in pipeline.process_turn(
        messages=messages, strategy_name="common_identity", study_id="test123"
    ):
        pass


def test_time_limit_reached_yields_llm_response():
    """Time limit should produce an LLM closing reply, not a canned error message."""
    chunks = _run(_make_messages(age_minutes=90), max_session_minutes=60)
    assert "".join(chunks)  # got something from the LLM, not empty


# ---------------------------------------------------------------------------
# Pipeline: time limit NOT reached → normal flow
# ---------------------------------------------------------------------------

def test_time_limit_not_reached_does_not_screen():
    """Session within the time limit must not call advance_user_state."""
    advance_mock = MagicMock()
    _run(_make_messages(age_minutes=10), max_session_minutes=60, advance_mock=advance_mock)
    advance_mock.assert_not_called()


def test_disabled_time_limit_does_not_screen():
    """max_session_minutes=0 disables the feature entirely."""
    advance_mock = MagicMock()
    _run(_make_messages(age_minutes=9999), max_session_minutes=0, advance_mock=advance_mock)
    advance_mock.assert_not_called()
