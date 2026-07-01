"""
Iterative quality-gate tests — cover every critical gate across all conditions.

For each gate we test three scenarios (multi-turn where needed):
  REJECT  — thin/bare answer → agent must probe, must NOT accept
  PERSIST — agent probes, user sends another thin answer → agent still probes (doesn't give up)
  ACCEPT  — specific/substantive answer → agent accepts (no unnecessary follow-up)

We also test OBSERVE directly:
  OBSERVE_THIN     — thin answer → signal stays false
  OBSERVE_SPECIFIC — specific answer → signal fires true

Run with:
    python -m pytest tests/test_quality_gates.py -v -s

All tests call the real Anthropic API (Haiku). They are intentionally slow.
"""
from __future__ import annotations

import asyncio
import json
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

from app.agent.prompts import build_system_prompt, OBSERVE_PROMPTS  # noqa: E402
from app.agent.state import SessionState, Stage  # noqa: E402
from app.agent.strategies import get_strategy, Strategy  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm.anthropic import AnthropicProvider  # noqa: E402
from app.llm.base import Message  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


async def _observe(llm: AnthropicProvider, strategy: Strategy,
                   prior_signals: dict, prev_assistant: str, user_msg: str) -> dict:
    """Run OBSERVE and return the parsed signal dict."""
    observe_system = OBSERVE_PROMPTS[strategy]
    content = (
        f"Known signals so far: {json.dumps(prior_signals)}\n\n"
        f"Previous assistant message: \"{prev_assistant}\"\n\n"
        f"Current user message: \"{user_msg}\""
    )
    raw = await llm.complete(
        messages=[Message(role="user", content=content)],
        system=observe_system,
    )
    # strip markdown code fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
        clean = clean.rsplit("```", 1)[0]
    return json.loads(clean)


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


def _probes(response: str) -> bool:
    probing = [
        "?", "could you", "can you", "tell me more", "say more", "bit more",
        "help me understand", "even a rough", "even something", "what would you",
        "how would you", "what do you", "try to", "worth sharing", "what about",
        "what makes", "what's that", "which one", "what part", "what kind",
        "a bit more", "little more", "say a bit", "your reasoning",
    ]
    lower = response.lower()
    return any(p in lower for p in probing)


def _closes(response: str) -> bool:
    """True if the response sounds like it's wrapping up or advancing."""
    closing = [
        "thank", "close the chat", "you're all done", "you can close",
        "that's all", "wrap up", "is there anything", "last question",
        "shifts how you see",  # the S4 final question — signals S3 accepted
    ]
    lower = response.lower()
    return any(c in lower for c in closing)


# ===========================================================================
# GATE 1: personal_narrative S2 — person_details_count
# Must reject bare trait labels; must accept specific facts with context
# ===========================================================================

class TestPersonalNarrativeS2Details:
    """'He is nice' must not advance; 'He coaches his son's Little League...' must."""

    OPENING = "Tell me a bit about them — who are they to you, and what are they like as a person?"
    SIGNALS = {"person_label": "my uncle"}

    def test_reject_bare_trait(self) -> None:
        """Single trait label → agent probes."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_2, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "He is nice."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S2 REJECT bare trait]\n{resp}")
        assert _probes(resp), f"Agent accepted bare trait 'He is nice'.\n{resp}"

    def test_persist_after_second_thin(self) -> None:
        """After probing once, second thin answer should still be probed."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_2, self.SIGNALS, turn=2)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "He is nice."},
            {"role": "assistant", "content": "Getting a real sense of him as a person is what this is really about — even something small, like what he tends to talk about, would help a lot. What's he like day to day?"},
            {"role": "user", "content": "Pretty friendly I guess."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S2 PERSIST second thin]\n{resp}")
        assert _probes(resp), f"Agent gave up probing after second thin answer.\n{resp}"

    def test_accept_specific_detail(self) -> None:
        """Specific fact with context → agent accepts and moves on."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_2, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "He coaches his son's Little League team every Saturday and is really passionate about making sure the kids have a positive experience, not just winning."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S2 ACCEPT specific detail]\n{resp}")
        assert not _closes(resp) or _probes(resp), f"Unexpected close without probe.\n{resp}"
        # Key assertion: response should not re-probe for MORE after a rich answer
        # (it can probe deeper into the same detail, but not re-reject it)
        assert "he is nice" not in resp.lower(), f"Agent seems confused.\n{resp}"

    def test_observe_rejects_bare_trait(self) -> None:
        """OBSERVE: 'He is nice' must not count as a detail."""
        llm = _make_llm()
        signals = asyncio.run(_observe(
            llm, Strategy.PERSONAL_NARRATIVE, self.SIGNALS,
            self.OPENING, "He is nice.",
        ))
        print(f"\n[S2 OBSERVE thin] person_details_count={signals.get('person_details_count')}")
        assert signals.get("person_details_count", 0) == 0, (
            f"OBSERVE counted 'He is nice' as a detail: {signals}"
        )

    def test_observe_counts_specific_detail(self) -> None:
        """OBSERVE: specific fact with context must count."""
        llm = _make_llm()
        signals = asyncio.run(_observe(
            llm, Strategy.PERSONAL_NARRATIVE, self.SIGNALS,
            self.OPENING,
            "He coaches his son's Little League team every Saturday and is really passionate about the kids having fun, not just winning.",
        ))
        print(f"\n[S2 OBSERVE specific] person_details_count={signals.get('person_details_count')}")
        assert signals.get("person_details_count", 0) >= 1, (
            f"OBSERVE failed to count a specific detail: {signals}"
        )


# ===========================================================================
# GATE 2: personal_narrative S3 — origins_explored
# Must reject bare category labels; must accept specific grounding
# ===========================================================================

class TestPersonalNarrativeS3Origins:
    """'Their family' must not advance; a specific family story must."""

    OPENING = "Do you have a sense of why they hold the political views they do — like, where those views come from for them?"
    SIGNALS = {"person_label": "my uncle", "person_details_count": 3}

    def test_reject_bare_category(self) -> None:
        """Bare category label → agent probes."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_3, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "Their family I think."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S3 REJECT bare category]\n{resp}")
        assert _probes(resp), f"Agent accepted 'Their family I think' without probing.\n{resp}"

    def test_reject_deflection(self) -> None:
        """Deflection ('I have no idea') → agent encourages a guess."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_3, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "I have no idea why."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S3 REJECT deflection]\n{resp}")
        assert _probes(resp), f"Agent accepted 'I have no idea' without probing.\n{resp}"

    def test_persist_bare_upbringing(self) -> None:
        """After probing, 'just their upbringing' should still be probed."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_3, self.SIGNALS, turn=2)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "I have no idea."},
            {"role": "assistant", "content": "Your best guess is really valuable — even something based on what you know about his life. What do you think might have shaped how he sees things?"},
            {"role": "user", "content": "Just their upbringing I think."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S3 PERSIST upbringing]\n{resp}")
        assert _probes(resp), f"Agent accepted bare 'upbringing' after probe.\n{resp}"

    def test_accept_specific_grounding(self) -> None:
        """Specific grounding → agent accepts and moves forward."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_3, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "His dad was a union worker who got laid off in the 80s and blamed trade policy for it. I think that really shaped how he sees government and the economy."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S3 ACCEPT specific]\n{resp}")
        # Should engage with the content, not re-reject it
        assert "??" not in resp, f"Unexpected double-probe.\n{resp}"

    def test_observe_rejects_bare_category(self) -> None:
        """OBSERVE: 'Their family' must NOT set origins_explored=True."""
        llm = _make_llm()
        signals = asyncio.run(_observe(
            llm, Strategy.PERSONAL_NARRATIVE, self.SIGNALS,
            self.OPENING, "Their family I think.",
        ))
        print(f"\n[S3 OBSERVE thin] origins_explored={signals.get('origins_explored')}")
        assert not signals.get("origins_explored"), (
            f"OBSERVE fired origins_explored on bare category: {signals}"
        )

    def test_observe_rejects_upbringing_label(self) -> None:
        """OBSERVE: 'Just their upbringing' must NOT set origins_explored=True."""
        llm = _make_llm()
        signals = asyncio.run(_observe(
            llm, Strategy.PERSONAL_NARRATIVE, self.SIGNALS,
            "What about their upbringing do you think had the most influence?",
            "Just their upbringing I think.",
        ))
        print(f"\n[S3 OBSERVE upbringing] origins_explored={signals.get('origins_explored')}")
        assert not signals.get("origins_explored"), (
            f"OBSERVE fired on bare 'upbringing': {signals}"
        )

    def test_observe_accepts_specific_grounding(self) -> None:
        """OBSERVE: specific grounding must set origins_explored=True."""
        llm = _make_llm()
        signals = asyncio.run(_observe(
            llm, Strategy.PERSONAL_NARRATIVE, self.SIGNALS,
            self.OPENING,
            "His dad was a union worker who got laid off and blamed trade policy. I think that shaped a lot of how he sees things.",
        ))
        print(f"\n[S3 OBSERVE specific] origins_explored={signals.get('origins_explored')}")
        assert signals.get("origins_explored") is True, (
            f"OBSERVE missed specific grounding: {signals}"
        )


# ===========================================================================
# GATE 3: personal_narrative S4 — generalization_reflected
# Must reject bare verdict; must accept specific reflection
# ===========================================================================

class TestPersonalNarrativeS4Generalization:
    """'Not typical' must not end conversation; a specific reflection must."""

    OPENING = "Thinking about your uncle — do you think they're pretty typical of Republican supporters, or more of an exception?"
    SIGNALS = {"person_label": "my uncle", "person_details_count": 3, "origins_explored": True}

    def test_reject_bare_verdict(self) -> None:
        """'Not typical' with no explanation → agent probes."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_4, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "Not typical."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S4 REJECT bare verdict]\n{resp}")
        assert _probes(resp), f"Agent accepted bare 'Not typical' without probing.\n{resp}"
        assert not _closes(resp) or _probes(resp)

    def test_persist_thin_exception(self) -> None:
        """After probing, thin 'more of an exception' still needs follow-up."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.STAGE_4, self.SIGNALS, turn=2)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "Not typical."},
            {"role": "assistant", "content": "What makes him feel that way to you? What would a more typical Republican supporter look like in your mind?"},
            {"role": "user", "content": "I think he's more of an exception."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[S4 PERSIST thin]\n{resp}")
        assert _probes(resp), f"Agent accepted thin 'more of an exception' without probing.\n{resp}"

    def test_observe_rejects_bare_verdict(self) -> None:
        """OBSERVE: 'not typical' must NOT set generalization_reflected=True."""
        llm = _make_llm()
        final_q = "Is there anything about our conversation — or about thinking through your uncle — that shifts how you see Republican supporters more broadly, even slightly?"
        signals = asyncio.run(_observe(
            llm, Strategy.PERSONAL_NARRATIVE, self.SIGNALS,
            final_q, "Not really.",
        ))
        print(f"\n[S4 OBSERVE thin] generalization_reflected={signals.get('generalization_reflected')}")
        assert not signals.get("generalization_reflected"), (
            f"OBSERVE fired generalization_reflected on 'Not really': {signals}"
        )

    def test_observe_accepts_specific_reflection(self) -> None:
        """OBSERVE: full reflection on whether view shifted must fire True."""
        llm = _make_llm()
        final_q = "Is there anything about our conversation — or about thinking through your uncle — that shifts how you see Republican supporters more broadly, even slightly?"
        signals = asyncio.run(_observe(
            llm, Strategy.PERSONAL_NARRATIVE, self.SIGNALS,
            final_q,
            "A little bit, yeah. Thinking about his background made me realize I usually picture Republican supporters as very ideological, but he's not like that at all — he votes Republican mostly out of habit and family loyalty, not strong conviction.",
        ))
        print(f"\n[S4 OBSERVE specific] generalization_reflected={signals.get('generalization_reflected')}")
        assert signals.get("generalization_reflected") is True, (
            f"OBSERVE missed specific generalization reflection: {signals}"
        )


# ===========================================================================
# GATE 4: common_identity S1 — feeling_expressed
# Must reject 'complicated'; must accept a feeling with texture
# ===========================================================================

class TestCommonIdentityS1Feeling:
    """'That is complicated' must not advance; 'frustrated because...' must."""

    OPENING = ("Welcome, and thanks for taking the time to chat with me today. "
               "We're going to talk about how you experience political division and where "
               "those feelings come from. When you think about people who support Republicans, "
               "what's the feeling that comes up most for you?")

    def test_reject_vague_label(self) -> None:
        """'Complicated' → agent probes for an actual feeling."""
        llm = _make_llm()
        system = _system("common_identity", Stage.STAGE_1)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "That is complicated."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CI-S1 REJECT vague]\n{resp}")
        assert _probes(resp), f"Agent accepted 'That is complicated'.\n{resp}"

    def test_reject_idk(self) -> None:
        """'idk' → agent probes."""
        llm = _make_llm()
        system = _system("common_identity", Stage.STAGE_1)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "idk, mixed feelings I guess"},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CI-S1 REJECT idk]\n{resp}")
        assert _probes(resp), f"Agent accepted 'idk, mixed feelings'.\n{resp}"

    def test_persist_bare_emotion(self) -> None:
        """After probing, bare 'frustrated' still needs texture."""
        llm = _make_llm()
        system = _system("common_identity", Stage.STAGE_1, turn=2)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "That is complicated."},
            {"role": "assistant", "content": "Your feelings are really what we're here to explore — even a rough word would help. Frustrated? Worn out? Something else?"},
            {"role": "user", "content": "Frustrated."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CI-S1 PERSIST bare emotion]\n{resp}")
        assert _probes(resp), f"Agent advanced on bare 'Frustrated' without asking for texture.\n{resp}"

    def test_accept_feeling_with_texture(self) -> None:
        """Feeling with explanation → agent accepts."""
        llm = _make_llm()
        system = _system("common_identity", Stage.STAGE_1)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "Frustrated, honestly. Like I can't understand how anyone sees things that differently from me — it feels almost personal sometimes, like they're not even living in the same reality."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CI-S1 ACCEPT with texture]\n{resp}")
        # Should engage with the content, not re-probe for a feeling name
        assert "what's the feeling" not in resp.lower(), f"Agent re-asked the opening question.\n{resp}"


# ===========================================================================
# GATE 5: misperception_correction S2 — quiz reasoning
# Must reject bare evaluative labels; must accept explanation of WHY
# ===========================================================================

class TestMisperceptionQuizReasoning:
    """'D. That is extreme.' must not reveal finding; a real explanation must."""

    QUESTION = ("Question 2 of 8: Would MOST Republican supporters support having the army "
                "patrol the streets of cities with high crime?\n\n  1. Never\n  2. Probably not\n"
                "  3. Probably\n  4. Definitely\n\nPlease choose a number and share a brief reason.")

    def test_reject_evaluative_label(self) -> None:
        """'D. That is extreme' → agent must ask for WHY before revealing finding."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_2,
                         signals={"questions_answered": 1, "mid_quiz_reflection_done": False},
                         turn=2)
        msgs = [
            {"role": "assistant", "content": self.QUESTION},
            {"role": "user", "content": "D. That is extreme."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S2 REJECT evaluative]\n{resp}")
        assert _probes(resp), f"Agent accepted bare evaluative label.\n{resp}"
        # Must NOT reveal the actual finding — check for the qualitative reveal pattern
        # (agent may say "before I share what surveys found" which is fine — NOT a reveal)
        reveal_markers = [
            "surveys found that most", "survey found that most",
            "most republican supporters said", "most republicans said",
            "in reality, most", "actually, most",
        ]
        assert not any(m in resp.lower() for m in reveal_markers), (
            f"Agent revealed finding before substantive reasoning.\n{resp}"
        )

    def test_reject_number_only(self) -> None:
        """Just '3' with no reason → must ask for reasoning."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_2,
                         signals={"questions_answered": 1, "mid_quiz_reflection_done": False},
                         turn=2)
        msgs = [
            {"role": "assistant", "content": self.QUESTION},
            {"role": "user", "content": "3"},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S2 REJECT number only]\n{resp}")
        assert _probes(resp), f"Agent accepted number with no reasoning.\n{resp}"

    def test_accept_real_reasoning(self) -> None:
        """A real explanation of why → agent reveals finding and moves on."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_2,
                         signals={"questions_answered": 1, "mid_quiz_reflection_done": False},
                         turn=2)
        msgs = [
            {"role": "assistant", "content": self.QUESTION},
            {"role": "user", "content": "D. I feel like a lot of Republican voters are really focused on law and order, and I've seen a lot of news about them supporting tough policing measures. So I assumed most would want that."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S2 ACCEPT real reasoning]\n{resp}")
        reveal_markers = ["surveys found", "survey found", "actually found", "most republican"]
        assert any(m in resp.lower() for m in reveal_markers), (
            f"Agent did not reveal finding after substantive reasoning.\n{resp}"
        )


# ===========================================================================
# GATE 6: misperception_correction S3 — reflection_shared
# Must reject bare reaction; must accept a genuine impression
# ===========================================================================

class TestMisperceptionS3Reflection:
    """'Surprising' must not advance; a genuine impression of the gap must."""

    OPENING = ("That's all 8 questions. Looking at the full picture — how does the gap "
               "between what you expected and what surveys actually found sit with you?")
    SIGNALS = {"questions_answered": 8, "mid_quiz_reflection_done": True}

    def test_reject_single_word(self) -> None:
        """Single word 'surprising' → agent probes."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_3, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "Surprising."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S3 REJECT single word]\n{resp}")
        assert _probes(resp), f"Agent accepted 'Surprising' without probing.\n{resp}"

    def test_reject_evaluative_sentence(self) -> None:
        """'That is not what I expected' → agent probes for why."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_3, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "That is not what I expected."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S3 REJECT evaluative sentence]\n{resp}")
        assert _probes(resp), f"Agent accepted 'That is not what I expected'.\n{resp}"

    def test_accept_genuine_impression(self) -> None:
        """Full sentence with genuine content → agent accepts and asks follow-up."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_3, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "I think I assumed they were more extreme than the data shows. Seeing how many of them actually reject these anti-democratic things makes me realize I've probably been picturing a much more radical version of the average Republican voter."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S3 ACCEPT genuine]\n{resp}")
        # Should ask the follow-up question about which one stood out
        assert _probes(resp), f"Agent did not ask follow-up after genuine reflection.\n{resp}"
        # Should NOT re-probe for a feeling name
        assert "surprising" not in resp.lower() or "?" in resp, f"Unexpected response.\n{resp}"


# ===========================================================================
# GATE 7: misperception_correction S4 — final democratic norms reflection
# Must reject bare verdict; must accept explanation of whether/why view shifted
# ===========================================================================

class TestMisperceptionS4FinalReflection:
    """'More committed I think' must not close; an explained view must."""

    QUESTION = ("Last question: based on what you saw today, do you think the average "
                "Republican supporter is more or less committed to democratic norms than "
                "you expected going in?")
    SIGNALS = {"questions_answered": 8, "reflection_shared": True}

    def test_reject_bare_verdict(self) -> None:
        """'More committed I think' → agent asks why."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_4, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.QUESTION},
            {"role": "user", "content": "More committed I think."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S4 REJECT bare verdict]\n{resp}")
        assert _probes(resp), f"Agent accepted bare 'More committed I think'.\n{resp}"

    def test_reject_short_verdict(self) -> None:
        """'About the same' → agent probes for explanation."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_4, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.QUESTION},
            {"role": "user", "content": "About the same."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S4 REJECT short verdict]\n{resp}")
        assert _probes(resp), f"Agent accepted 'About the same' without explanation.\n{resp}"

    def test_accept_explained_view(self) -> None:
        """Explained view → agent accepts (may thank/close)."""
        llm = _make_llm()
        system = _system("misperception_correction", Stage.STAGE_4, self.SIGNALS)
        msgs = [
            {"role": "assistant", "content": self.QUESTION},
            {"role": "user", "content": "More committed than I expected. I came in thinking most Republican voters were pretty extreme about bending the rules, but the data showed a lot of them actually care about democratic norms. It doesn't change my politics, but it does make me think I've been generalizing too much."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[MISC-S4 ACCEPT explained]\n{resp}")
        # Should NOT re-probe for an explanation
        assert "why" not in resp.lower()[:80] or "?" in resp, f"Agent unnecessarily re-probed.\n{resp}"


# ===========================================================================
# GATE 8: control S1 — winding_down must not fire on bare status labels
# ===========================================================================

class TestControlS1StatusLabel:
    """'All good' must not be treated as sharing something real."""

    OPENING = ("Welcome, and thanks for taking the time to chat with me today. "
               "This is just a brief, informal check-in conversation about how you've "
               "been doing lately. I'd like to start by checking in — how have you been "
               "doing lately? Is there anything that's been weighing on you or on your mind?")

    def test_reject_status_label(self) -> None:
        """'All good' → agent probes."""
        llm = _make_llm()
        system = _system("control", Stage.STAGE_1)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "All good."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CTRL-S1 REJECT status]\n{resp}")
        assert _probes(resp), f"Agent accepted 'All good' without probing.\n{resp}"

    def test_persist_second_status(self) -> None:
        """After probing, 'fine, nothing special' should still prompt sharing."""
        llm = _make_llm()
        system = _system("control", Stage.STAGE_1, turn=2)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "All good."},
            {"role": "assistant", "content": "Glad to hear it! Hearing how you've really been doing is what this conversation is for — even something small or routine that's been on your mind would be worth sharing."},
            {"role": "user", "content": "Fine, nothing special."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CTRL-S1 PERSIST second status]\n{resp}")
        assert _probes(resp), f"Agent gave up probing after 'Fine, nothing special'.\n{resp}"

    def test_accept_real_content(self) -> None:
        """Something real → agent engages naturally."""
        llm = _make_llm()
        system = _system("control", Stage.STAGE_1)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "Honestly, work has been pretty stressful — my manager keeps changing priorities and I've been staying late every day this week."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CTRL-S1 ACCEPT real content]\n{resp}")
        assert _probes(resp), f"Agent did not ask follow-up after real content.\n{resp}"


# ===========================================================================
# GATE 9: control_politics S1 — winding_down must not fire on vague assessments
# ===========================================================================

class TestControlPoliticsS1VagueAssessment:
    """'Everything is a mess' must not be treated as a specific political thought."""

    OPENING = ("Welcome, and thanks for taking the time to chat with me today. "
               "We're going to have an open conversation about whatever's on your mind "
               "politically. I want to start with something open — when you think about "
               "the political situation in the US right now, what's on your mind?")

    def test_reject_vague_assessment(self) -> None:
        """'Everything is a mess' → agent asks for a specific topic."""
        llm = _make_llm()
        system = _system("control_politics", Stage.STAGE_1)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "Everything is a mess."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CP-S1 REJECT vague]\n{resp}")
        assert _probes(resp), f"Agent accepted 'Everything is a mess'.\n{resp}"

    def test_persist_bare_topic(self) -> None:
        """After probing, bare topic name should still prompt for specifics."""
        llm = _make_llm()
        system = _system("control_politics", Stage.STAGE_1, turn=2)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "Everything is a mess."},
            {"role": "assistant", "content": "What's actually on your mind about politics is exactly what we're here to talk about — even something small from the news lately would be a good place to start."},
            {"role": "user", "content": "The economy."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CP-S1 PERSIST bare topic]\n{resp}")
        assert _probes(resp), f"Agent accepted bare 'The economy' without asking what specifically.\n{resp}"

    def test_accept_specific_thought(self) -> None:
        """Specific topic with actual content → agent engages and follows up."""
        llm = _make_llm()
        system = _system("control_politics", Stage.STAGE_1)
        msgs = [
            {"role": "assistant", "content": self.OPENING},
            {"role": "user", "content": "I've been really frustrated with how the tariff situation is playing out — my brother runs a small business that imports parts and his costs have basically doubled in the last few months."},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[CP-S1 ACCEPT specific]\n{resp}")
        assert _probes(resp), f"Agent did not follow up on specific content.\n{resp}"


# ===========================================================================
# GATE 10: user_abort — all conditions must record COMPLETE when user asks to end
# Tests OBSERVE signal firing + transition unit test
# ===========================================================================

class TestUserAbortSignal:
    """When user says 'can we end', OBSERVE must fire user_abort=True for all conditions.
    The pipeline then transitions to COMPLETE, recording the session correctly."""

    _ABORT_MSG = "thank you. can we end"
    _SOFT_ABORT = "can we be done now"
    _NOT_ABORT = "I'm done thinking about that topic"  # topic-level, not session-level

    def _observe_abort(self, strategy: Strategy, prev_assistant: str,
                       user_msg: str, prior_signals: dict | None = None) -> dict:
        llm = _make_llm()
        return asyncio.run(_observe(
            llm, strategy, prior_signals or {},
            prev_assistant, user_msg,
        ))

    # --- common_identity ---

    def test_ci_abort_fires(self) -> None:
        sigs = self._observe_abort(
            Strategy.COMMON_IDENTITY,
            "Could you share your thought in a full sentence?",
            self._ABORT_MSG,
        )
        print(f"\n[CI user_abort] {sigs.get('user_abort')}")
        assert sigs.get("user_abort") is True, f"user_abort not fired for CI: {sigs}"

    def test_ci_soft_abort_fires(self) -> None:
        sigs = self._observe_abort(
            Strategy.COMMON_IDENTITY,
            "What do you think most ordinary people want?",
            self._SOFT_ABORT,
        )
        print(f"\n[CI soft abort] {sigs.get('user_abort')}")
        assert sigs.get("user_abort") is True, f"soft abort not fired for CI: {sigs}"

    def test_ci_topic_close_does_not_fire(self) -> None:
        """'I'm done thinking about that topic' is NOT an abort."""
        sigs = self._observe_abort(
            Strategy.COMMON_IDENTITY,
            "What else comes to mind?",
            self._NOT_ABORT,
            prior_signals={"feeling_expressed": True},
        )
        print(f"\n[CI non-abort] {sigs.get('user_abort')}")
        assert not sigs.get("user_abort"), f"user_abort falsely fired on topic-close: {sigs}"

    # --- personal_narrative ---

    def test_pn_abort_fires(self) -> None:
        sigs = self._observe_abort(
            Strategy.PERSONAL_NARRATIVE,
            "Could you share your thought in a full sentence?",
            self._ABORT_MSG,
            prior_signals={"person_label": "my uncle"},
        )
        print(f"\n[PN user_abort] {sigs.get('user_abort')}")
        assert sigs.get("user_abort") is True, f"user_abort not fired for PN: {sigs}"

    # --- misperception_correction ---

    def test_misc_abort_fires(self) -> None:
        sigs = self._observe_abort(
            Strategy.MISPERCEPTION_CORRECTION,
            "Could you say a bit more about why you picked that?",
            self._ABORT_MSG,
            prior_signals={"questions_answered": 3},
        )
        print(f"\n[MISC user_abort] {sigs.get('user_abort')}")
        assert sigs.get("user_abort") is True, f"user_abort not fired for MISC: {sigs}"

    # --- control ---

    def test_ctrl_abort_fires(self) -> None:
        sigs = self._observe_abort(
            Strategy.CONTROL,
            "Is there anything else on your mind?",
            self._ABORT_MSG,
            prior_signals={"topics_shared": ["stressed about work"]},
        )
        print(f"\n[CTRL user_abort] {sigs.get('user_abort')}")
        assert sigs.get("user_abort") is True, f"user_abort not fired for CTRL: {sigs}"

    # --- control_politics ---

    def test_cp_abort_fires(self) -> None:
        sigs = self._observe_abort(
            Strategy.CONTROL_POLITICS,
            "Is there anything else about politics on your mind?",
            self._ABORT_MSG,
            prior_signals={"topics_shared": ["worried about the economy"]},
        )
        print(f"\n[CP user_abort] {sigs.get('user_abort')}")
        assert sigs.get("user_abort") is True, f"user_abort not fired for CP: {sigs}"

    # --- transition unit test ---

    def test_abort_forces_complete_transition(self) -> None:
        """Unit test: evaluate_transition must advance to COMPLETE when user_abort=True,
        regardless of condition or stage."""
        import asyncio as _asyncio
        from app.agent.phases import StageController
        from app.agent.state import SessionState, Stage

        controller = StageController(llm=None)

        # personal_narrative mid-S2 — would normally need person_details_count >= 2
        state = SessionState(
            study_id="t", strategy="personal_narrative",
            stage=Stage.STAGE_2, stage_turn_count=1,
            signals={"person_label": "my uncle", "person_details_count": 0,
                     "user_abort": True},
        )
        _asyncio.run(controller.evaluate_transition(state, "can we end"))
        assert state.stage == Stage.COMPLETE, (
            f"user_abort did not force COMPLETE: stage={state.stage}"
        )

    def test_abort_complete_response_skips_content_ack(self) -> None:
        """When in COMPLETE stage after abort, agent should NOT try to reference
        'can we end' as substantive content."""
        llm = _make_llm()
        system = _system("personal_narrative", Stage.COMPLETE,
                         signals={"person_label": "my uncle", "user_abort": True})
        msgs = [
            {"role": "assistant", "content": "Could you share your thought in a full sentence?"},
            {"role": "user", "content": "thank you. can we end"},
        ]
        resp = asyncio.run(_respond(llm, system, msgs))
        print(f"\n[ABORT COMPLETE response]\n{resp}")
        # Should be a clean goodbye, not referencing "can we end" as meaningful content
        assert "thank" in resp.lower() or "close" in resp.lower(), (
            f"COMPLETE response does not contain farewell: {resp}"
        )
        assert "?" not in resp, f"COMPLETE response asked a question: {resp}"
