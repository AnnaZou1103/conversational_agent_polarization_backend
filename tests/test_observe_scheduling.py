"""OBSERVE scheduling: every condition keeps OBSERVE on the critical path.

`control` / `control_politics` used to skip ahead with a stage gate based
solely on stage_turn_count, letting OBSERVE run concurrently with EXECUTE.
Their gate is now content-driven (it waits for OBSERVE's `winding_down`
signal so the conversation doesn't get cut off mid-thought — see
app.agent.phases._TRANSITIONS), so OBSERVE must finish before the gate runs,
same as every other condition.

The probe LLM makes OBSERVE block until EXECUTE has started streaming:
  * sequential (gated) -> OBSERVE awaited first, EXECUTE never starts in time
                          -> OBSERVE hits its short timeout (proves it ran first)
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

from app.agent.pipeline import (  # noqa: E402
    AgentPipeline,
    _OBSERVE_OFF_CRITICAL_PATH,
)
from app.llm.base import LLMProvider  # noqa: E402


class _OverlapProbe(LLMProvider):
    """OBSERVE (complete) blocks until EXECUTE (stream) starts, up to `timeout`s.

    Records a timeline so a test can see whether EXECUTE began while OBSERVE was
    still pending (overlap) or only after OBSERVE returned (sequential)."""

    def __init__(self, timeout: float = 0.4) -> None:
        self._timeout = timeout
        self._execute_started = asyncio.Event()
        self.timeline: list[str] = []
        self.observe_finished = False
        self.observe_unblocked_by_execute: bool | None = None

    async def complete(self, messages, system=None, temperature=0.7,
                       max_tokens=2048) -> str:
        self.timeline.append("observe:start")
        try:
            await asyncio.wait_for(self._execute_started.wait(), self._timeout)
            self.observe_unblocked_by_execute = True
        except asyncio.TimeoutError:
            self.observe_unblocked_by_execute = False
        self.timeline.append("observe:end")
        self.observe_finished = True
        return json.dumps({
            "topics_shared": ["work stress"],
            "current_mood": "tired",
            "winding_down": False,
        })

    async def stream(self, messages, system=None, temperature=0.7,
                     max_tokens=2048) -> AsyncIterator[str]:
        self.timeline.append("execute:start")
        self._execute_started.set()
        yield "ok"


def _run(strategy: str, probe: _OverlapProbe) -> None:
    pipeline = AgentPipeline(llm=probe)
    doc = {
        "study_id": "sid",
        "payload": {
            "stage": "stage_1",
            "strategy": strategy,
            "stage_turn_count": 0,
            "signals": {},
        },
    }
    messages = [
        {"role": "assistant", "content": "How have you been?"},
        {"role": "user", "content": "Honestly pretty stressed with work lately."},
    ]

    async def _consume():
        async for _ in pipeline.process_turn(
            messages=messages, strategy_name=strategy, study_id="sid"
        ):
            pass

    with patch("app.agent.state.get_conversation", return_value=doc), \
         patch("app.agent.state.get_user_party", return_value=None), \
         patch("app.agent.pipeline.build_system_prompt", return_value="SYSTEM"), \
         patch("app.agent.pipeline.log_turn"), \
         patch("app.agent.pipeline.log_safety_event"), \
         patch("app.config.settings.enable_think", False):
        asyncio.run(_consume())


def test_control_keeps_observe_on_critical_path() -> None:
    # control's gate now reads `winding_down`, which OBSERVE extracts from
    # THIS turn's message, so EXECUTE must wait for OBSERVE — the probe's
    # EXECUTE never runs in time and OBSERVE hits its (short) timeout.
    probe = _OverlapProbe(timeout=0.3)
    _run("control", probe)
    assert probe.observe_unblocked_by_execute is False, probe.timeline
    assert probe.timeline.index("observe:end") < probe.timeline.index("execute:start"), \
        probe.timeline


def test_control_politics_keeps_observe_on_critical_path() -> None:
    probe = _OverlapProbe(timeout=0.3)
    _run("control_politics", probe)
    assert probe.observe_unblocked_by_execute is False, probe.timeline
    assert probe.timeline.index("observe:end") < probe.timeline.index("execute:start"), \
        probe.timeline


def test_signal_gated_condition_keeps_observe_on_critical_path() -> None:
    # common_identity is signal-gated: OBSERVE must finish before EXECUTE, so the
    # probe's EXECUTE never runs in time and OBSERVE hits its (short) timeout.
    probe = _OverlapProbe(timeout=0.3)
    _run("common_identity", probe)
    assert probe.observe_unblocked_by_execute is False, probe.timeline
    assert probe.timeline.index("observe:end") < probe.timeline.index("execute:start"), \
        probe.timeline


def test_off_critical_path_set_is_empty() -> None:
    # Guard against drift: every condition's stage gate is now signal-gated in
    # some way (control/control_politics wait on `winding_down`), so no
    # condition may defer OBSERVE off the critical path. Adding one back here
    # without a matching turn-count-only rule in phases._TRANSITIONS would
    # advance stages a turn late (gate would read stale signals).
    assert _OBSERVE_OFF_CRITICAL_PATH == frozenset()


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all observe-scheduling tests passed")
