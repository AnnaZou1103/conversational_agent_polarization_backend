"""Tests for conversation history persistence.

Verifies that:
1. save_turn_log overwrites payload (latest state) and appends to signal_history
2. get_chat_history reads messages from payload
3. get_conversation_observation reads strategy/signals from payload
4. build_session_state reconstructs stage/signals from payload
5. signal_history accumulates one snapshot per turn
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub DB layer before any app import so no real MongoDB connection is made.
_mock_col = MagicMock()
_db_stub = MagicMock()
_db_stub.user_docs = _mock_col
_db_stub.conversation_docs = _mock_col
sys.modules.setdefault("app.db.documents", _db_stub)
sys.modules.setdefault("app.db.user", MagicMock(
    get_user_party=MagicMock(return_value=None),
    study_id_is_valid=MagicMock(return_value=True),
))

from app.db.conversation import (  # noqa: E402
    save_turn_log,
    get_chat_history,
    get_conversation_observation,
)
from app.agent.state import build_session_state, Stage  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(turn=1, stage="stage_1", strategy="common_identity", signals=None, messages=None):
    return {
        "turn": turn,
        "stage": stage,
        "strategy": strategy,
        "timestamp": "2024-01-01T00:00:00Z",
        "signals": signals or {},
        "messages": messages or [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
    }


# ---------------------------------------------------------------------------
# save_turn_log — $set payload, $push signal_history
# ---------------------------------------------------------------------------

def test_save_turn_log_sets_payload():
    with patch("app.db.conversation.conversation_docs") as mock_col:
        entry = _entry()
        save_turn_log("sid1", entry)

        args, kwargs = mock_col.update_one.call_args
        update_doc = args[1]
        assert "payload" in update_doc.get("$set", {})
        assert update_doc["$set"]["payload"] == entry


def test_save_turn_log_pushes_signal_snapshot():
    with patch("app.db.conversation.conversation_docs") as mock_col:
        signals = {"feeling_expressed": True, "user_feeling_text": "frustrated"}
        entry = _entry(turn=2, stage="stage_2", signals=signals)
        save_turn_log("sid1", entry)

        args, _ = mock_col.update_one.call_args
        update_doc = args[1]
        snapshot = update_doc["$push"]["signal_history"]
        assert snapshot["turn"] == 2
        assert snapshot["stage"] == "stage_2"
        assert snapshot["signals"] == signals


def test_save_turn_log_signal_snapshot_excludes_messages():
    with patch("app.db.conversation.conversation_docs") as mock_col:
        save_turn_log("sid1", _entry())

        args, _ = mock_col.update_one.call_args
        snapshot = args[1]["$push"]["signal_history"]
        assert "messages" not in snapshot
        assert "system_prompt" not in snapshot


def test_save_turn_log_upserts():
    with patch("app.db.conversation.conversation_docs") as mock_col:
        save_turn_log("sid1", _entry())
        _, kwargs = mock_col.update_one.call_args
        assert kwargs.get("upsert") is True


# ---------------------------------------------------------------------------
# signal_history accumulates across turns (integration-style)
# ---------------------------------------------------------------------------

def test_signal_history_snapshots_are_lean():
    """Each snapshot contains only turn, stage, timestamp, signals — not messages."""
    with patch("app.db.conversation.conversation_docs") as mock_col:
        for i in range(1, 4):
            save_turn_log("sid1", _entry(turn=i, signals={"feeling_expressed": i > 1}))

        calls = mock_col.update_one.call_args_list
        assert len(calls) == 3
        for call in calls:
            snapshot = call.args[1]["$push"]["signal_history"]
            assert set(snapshot.keys()) == {"turn", "stage", "timestamp", "signals"}


# ---------------------------------------------------------------------------
# get_chat_history — reads from payload.messages
# ---------------------------------------------------------------------------

def test_get_chat_history_returns_payload_messages():
    msgs = [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "resp1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "resp2"},
    ]
    doc = {"study_id": "sid1", "payload": _entry(messages=msgs)}

    with patch("app.db.conversation.conversation_docs") as mock_col:
        mock_col.find_one.return_value = doc
        history = get_chat_history("sid1")

    assert len(history) == 4
    assert history[0].content == "msg1"
    assert history[-1].content == "resp2"


def test_get_chat_history_empty_when_no_document():
    with patch("app.db.conversation.conversation_docs") as mock_col:
        mock_col.find_one.return_value = None
        assert get_chat_history("sid1") == []


def test_get_chat_history_empty_when_no_payload():
    with patch("app.db.conversation.conversation_docs") as mock_col:
        mock_col.find_one.return_value = {"study_id": "sid1"}
        assert get_chat_history("sid1") == []


# ---------------------------------------------------------------------------
# get_conversation_observation — reads from payload
# ---------------------------------------------------------------------------

def test_get_conversation_observation_reads_payload_signals():
    signals = {"feeling_expressed": True, "exhausted_majority_introduced": True}
    doc = {"study_id": "sid1", "payload": _entry(strategy="common_identity", signals=signals)}

    with patch("app.db.conversation.conversation_docs") as mock_col:
        mock_col.find_one.return_value = doc
        obs = get_conversation_observation("sid1")

    assert obs is not None
    assert obs.observation.show_survey is True


def test_get_conversation_observation_none_when_no_payload():
    with patch("app.db.conversation.conversation_docs") as mock_col:
        mock_col.find_one.return_value = {"study_id": "sid1"}
        assert get_conversation_observation("sid1") is None


def test_get_conversation_observation_none_when_no_document():
    with patch("app.db.conversation.conversation_docs") as mock_col:
        mock_col.find_one.return_value = None
        assert get_conversation_observation("sid1") is None


# ---------------------------------------------------------------------------
# build_session_state — reconstructs from payload
# ---------------------------------------------------------------------------

def test_build_session_state_reads_stage_from_payload():
    doc = {
        "study_id": "sid1",
        "payload": _entry(stage="stage_3", signals={"feeling_expressed": True}),
    }
    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None):
        state = build_session_state("sid1", "common_identity", [])

    assert state.stage == Stage.STAGE_3


def test_build_session_state_reads_signals_from_payload():
    signals = {"feeling_expressed": True, "user_feeling_text": "frustrated"}
    doc = {"study_id": "sid1", "payload": _entry(stage="stage_2", signals=signals)}

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None):
        state = build_session_state("sid1", "common_identity", [])

    assert state.signals.get("feeling_expressed") is True
    assert state.signals.get("user_feeling_text") == "frustrated"


def test_build_session_state_handles_missing_payload():
    doc = {"study_id": "sid1"}

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None):
        state = build_session_state("sid1", "common_identity", [])

    assert state.stage == Stage.STAGE_1
    assert state.signals == {}


def test_build_session_state_restores_stage_turn_count():
    """stage_turn_count must survive across stateless requests, otherwise gates
    requiring stage_turn_count >= 2/3 never fire and conditions never COMPLETE."""
    payload = _entry(stage="stage_2")
    payload["stage_turn_count"] = 2
    doc = {"study_id": "sid1", "payload": payload}

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None):
        state = build_session_state("sid1", "common_identity", [])

    assert state.stage_turn_count == 2


def test_build_session_state_stage_turn_count_defaults_to_zero():
    doc = {"study_id": "sid1", "payload": _entry(stage="stage_1")}

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None):
        state = build_session_state("sid1", "common_identity", [])

    assert state.stage_turn_count == 0
