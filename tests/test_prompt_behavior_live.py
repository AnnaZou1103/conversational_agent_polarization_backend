"""Live LLM tests — verify each condition's substantive gate actually rejects
non-informative replies and probes for more.

These tests call the real Anthropic API (Haiku, fast/cheap). Run with:
    python -m pytest tests/test_prompt_behavior_live.py -v -s

Each test:
  - Builds the real system prompt for a condition + stage
  - Sends a non-informative user reply
  - Asserts the LLM response probes for more (does not accept and move on)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_mock_col = MagicMock()
_db_stub = MagicMock()
_db_stub.user_docs = _mock_col
_db_stub.conversation_docs = _mock_col
sys.modules.setdefault("app.db.documents", _db_stub)

from app.agent.prompts import build_system_prompt  # noqa: E402
from app.agent.state import SessionState, Stage  # noqa: E402
from app.agent.strategies import get_strategy  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm.anthropic import AnthropicProvider  # noqa: E402
from app.llm.base import Message  # noqa: E402


def _make_llm() -> AnthropicProvider:
    model = settings.fast_llm_model or settings.llm_model
    return AnthropicProvider(api_key=settings.llm_api_key, model=model)


def _msgs(raw: list[dict]) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in raw]


async def _respond(llm: AnthropicProvider, system: str, messages: list[dict]) -> str:
    chunks: list[str] = []
    async for chunk in llm.stream(messages=_msgs(messages), system=system):
        chunks.append(chunk)
    return "".join(chunks)


def _system(strategy_name: str, stage: Stage, signals: dict | None = None,
            turn: int = 1) -> str:
    strategy = get_strategy(strategy_name)
    state = SessionState(
        study_id="test",
        strategy=strategy_name,
        stage=stage,
        stage_turn_count=turn,
        signals=signals or {},
        political_party="Democrat",
    )
    return build_system_prompt(stage, strategy, state)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _probes_for_more(response: str) -> bool:
    """Return True if the response contains clear probing/encouraging language."""
    probing_phrases = [
        "?",               # any follow-up question
        "could you",
        "can you",
        "tell me more",
        "say more",
        "bit more",
        "help me understand",
        "even a rough",
        "even something",
        "what would you",
        "how would you",
        "what do you",
        "try to",
        "worth sharing",
        "really what",
        "important to",
    ]
    lower = response.lower()
    return any(p in lower for p in probing_phrases)


def _advances_stage(response: str, advance_markers: list[str]) -> bool:
    """Return True if the response contains language that would signal stage advance."""
    lower = response.lower()
    return any(m in lower for m in advance_markers)


# ---------------------------------------------------------------------------
# common_identity Stage 1 — must name a feeling, not just say "complicated"
# ---------------------------------------------------------------------------

def test_common_identity_s1_rejects_vague_label() -> None:
    """'That is complicated' should not be accepted as a feeling."""
    llm = _make_llm()
    system = _system("common_identity", Stage.STAGE_1)
    messages = [
        {"role": "assistant", "content": (
            "Welcome, and thanks for taking the time to chat with me today. "
            "We're going to talk about how you experience political division and "
            "where those feelings come from. When you think about people who support "
            "Republicans, what's the feeling that comes up most for you?"
        )},
        {"role": "user", "content": "That is complicated."},
    ]
    response = asyncio.run(_respond(llm, system, messages))
    print(f"\n[common_identity S1] response:\n{response}\n")

    # Should NOT advance to Stage 2 topics
    advances = _advances_stage(response, ["where those feelings come from", "what sources"])
    assert not advances, f"LLM advanced to Stage 2 despite vague reply. Response:\n{response}"

    # Should probe for an actual feeling
    assert _probes_for_more(response), (
        f"LLM did not probe for more after vague reply. Response:\n{response}"
    )


# ---------------------------------------------------------------------------
# personal_narrative Stage 2 — must add a specific detail, not just a trait label
# ---------------------------------------------------------------------------

def test_personal_narrative_s2_rejects_trait_label() -> None:
    """'He is nice' should not count as a detail and should prompt for specifics."""
    llm = _make_llm()
    system = _system(
        "personal_narrative", Stage.STAGE_2,
        signals={"person_label": "my uncle"},
        turn=1,
    )
    messages = [
        {"role": "assistant", "content": (
            "Tell me a bit about your uncle — who is he to you, and what is he like as a person?"
        )},
        {"role": "user", "content": "He is nice."},
    ]
    response = asyncio.run(_respond(llm, system, messages))
    print(f"\n[personal_narrative S2] response:\n{response}\n")

    # Should NOT move to the next Stage 2 question about what they care about
    advances = _advances_stage(response, [
        "what's something they care about",
        "something they care about deeply",
        "has there been a moment",
        "stood out to you",
    ])
    assert not advances, f"LLM moved to next question despite trait-only reply. Response:\n{response}"

    assert _probes_for_more(response), (
        f"LLM did not probe for more after 'He is nice'. Response:\n{response}"
    )


# ---------------------------------------------------------------------------
# misperception_correction Stage 1 quiz — must explain WHY, not just "That is illegal"
# ---------------------------------------------------------------------------

def test_misperception_s1_rejects_evaluative_sentence() -> None:
    """'That is illegal' should not be accepted as substantive reasoning."""
    llm = _make_llm()
    system = _system("misperception_correction", Stage.STAGE_1, turn=2)
    messages = [
        {"role": "assistant", "content": (
            "Question 1: What percentage of Republicans support making it illegal "
            "for a woman to get an abortion for any reason?\n"
            "A) 10%  B) 27%  C) 49%  D) 68%\n\n"
            "Pick one and share a brief reason."
        )},
        {"role": "user", "content": "D. That is illegal."},
    ]
    response = asyncio.run(_respond(llm, system, messages))
    print(f"\n[misperception_correction S1] response:\n{response}\n")

    # Should NOT reveal the survey finding (bare "surveys found" is too broad —
    # "before I share what surveys found" is a correct withhold, not a reveal).
    advances = _advances_stage(response, [
        "surveys found that", "survey found that", "surveys show that",
        "actually, the answer", "the correct answer", "the answer is",
        "in reality, only", "in reality, most", "research shows that",
        "according to surveys", "according to research",
    ])
    assert not advances, (
        f"LLM revealed survey finding despite non-substantive reply. Response:\n{response}"
    )

    assert _probes_for_more(response), (
        f"LLM did not probe for reasoning after 'That is illegal'. Response:\n{response}"
    )


# ---------------------------------------------------------------------------
# control Stage 1 — must share something real, not just "All good"
# ---------------------------------------------------------------------------

def test_control_s1_rejects_status_label() -> None:
    """'All good' should not be treated as sharing anything real."""
    llm = _make_llm()
    system = _system("control", Stage.STAGE_1, turn=1)
    messages = [
        {"role": "assistant", "content": (
            "Welcome, and thanks for taking the time to chat with me today. "
            "This is just a brief, informal check-in conversation about how you've "
            "been doing lately. I'd like to start by checking in — how have you been "
            "doing lately? Is there anything that's been weighing on you or on your mind?"
        )},
        {"role": "user", "content": "All good."},
    ]
    response = asyncio.run(_respond(llm, system, messages))
    print(f"\n[control S1] response:\n{response}\n")

    assert _probes_for_more(response), (
        f"LLM did not probe for more after 'All good'. Response:\n{response}"
    )


# ---------------------------------------------------------------------------
# control_politics Stage 1 — must share a specific political topic, not "Everything is a mess"
# ---------------------------------------------------------------------------

def test_control_politics_s1_rejects_vague_verdict() -> None:
    """'Everything is a mess' should not be accepted as sharing a specific political thought."""
    llm = _make_llm()
    system = _system("control_politics", Stage.STAGE_1, turn=1)
    messages = [
        {"role": "assistant", "content": (
            "Welcome, and thanks for taking the time to chat with me today. "
            "We're going to have an open conversation about whatever's on your mind "
            "politically. I want to start with something open — when you think about "
            "the political situation in the US right now, what's on your mind?"
        )},
        {"role": "user", "content": "Everything is a mess."},
    ]
    response = asyncio.run(_respond(llm, system, messages))
    print(f"\n[control_politics S1] response:\n{response}\n")

    assert _probes_for_more(response), (
        f"LLM did not probe for specifics after 'Everything is a mess'. Response:\n{response}"
    )


if __name__ == "__main__":
    cases = [
        ("common_identity S1 vague label", test_common_identity_s1_rejects_vague_label),
        ("personal_narrative S2 trait label", test_personal_narrative_s2_rejects_trait_label),
        ("misperception S1 evaluative sentence", test_misperception_s1_rejects_evaluative_sentence),
        ("control S1 status label", test_control_s1_rejects_status_label),
        ("control_politics S1 vague verdict", test_control_politics_s1_rejects_vague_verdict),
    ]
    passed = failed = 0
    for name, fn in cases:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
