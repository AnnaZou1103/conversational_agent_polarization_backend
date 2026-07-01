from __future__ import annotations

import logging
from collections.abc import Callable

from app.agent.state import Stage, SessionState
from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Canonical stage ordering. Transitions may only move forward (or stay): control
# conditions legitimately skip STAGE_1 -> STAGE_4, but a backward jump would
# reset stage_turn_count and could loop a session so it never reaches COMPLETE.
_STAGE_ORDER = (
    Stage.STAGE_1,
    Stage.STAGE_2,
    Stage.STAGE_3,
    Stage.STAGE_4,
    Stage.COMPLETE,
)

# Deterministic stage-transition table — one ordered list of rules per condition.
#
# Each rule is (from_stage, to_stage, predicate(signals, stage_turn_count)). The
# first rule whose `from_stage` matches the current stage AND whose predicate is
# true fires; otherwise the session stays put. These predicates are exactly the
# criteria the LLM stage controller used to be prompted with — but they only
# ever *read* signals that the OBSERVE step already extracted (booleans/ints),
# so there is nothing for an LLM to decide here. Computing them directly removes
# a full blocking LLM call from every turn's critical path and cannot fail to
# parse or "freelance" past the documented thresholds.
#
# NOTE on personal_narrative STAGE_1 -> STAGE_2: the legacy prompt named this
# signal `person_name`, but OBSERVE actually emits `person_label` (there is no
# `person_name` key in signals). The LLM gate tolerated the mismatch by fuzzy-
# matching; a literal translation would never fire, so we read `person_label`.
Predicate = Callable[[dict, int], bool]

_TRANSITIONS: dict[str, list[tuple[Stage, Stage, Predicate]]] = {
    "common_identity": [
        # n>=2 floor / n>=4 cap: spend at least 2 turns on feeling before advancing.
        (Stage.STAGE_1, Stage.STAGE_2,
         lambda s, n: (bool(s.get("feeling_expressed")) and n >= 2) or n >= 4),
        # n>=3 floor / n>=5 cap: spend at least 3 turns on media distortion before advancing.
        (Stage.STAGE_2, Stage.STAGE_3,
         lambda s, n: (bool(s.get("media_distortion_acknowledged")) and n >= 3) or n >= 5),
        # n>=2 floor / n>=4 cap: spend at least 2 turns on common identity before advancing.
        (Stage.STAGE_3, Stage.STAGE_4,
         lambda s, n: (bool(s.get("common_identity_described")) and n >= 2) or n >= 4),
        # n>=2 floor: match control S4 minimum; total min = 2+3+2+2 = 9.
        (Stage.STAGE_4, Stage.COMPLETE, lambda s, n: n >= 2),
    ],
    "personal_narrative": [
        # n>=2 floor / n>=4 cap: spend at least 2 turns identifying the person.
        (Stage.STAGE_1, Stage.STAGE_2,
         lambda s, n: (s.get("person_label") is not None and n >= 2) or n >= 4),
        # n>=3 floor / n>=6 cap: spend at least 3 turns gathering detail.
        (Stage.STAGE_2, Stage.STAGE_3,
         lambda s, n: (s.get("person_details_count", 0) >= 2 and n >= 3) or n >= 6),
        # n>=2 floor / n>=5 cap: spend at least 2 turns on origins.
        (Stage.STAGE_3, Stage.STAGE_4,
         lambda s, n: (bool(s.get("origins_explored")) and n >= 2) or n >= 5),
        # n>=2 floor on signal path / n>=6 cap; total min = 2+3+2+2 = 9.
        (Stage.STAGE_4, Stage.COMPLETE,
         lambda s, n: (bool(s.get("generalization_reflected")) and n >= 2) or n >= 6),
    ],
    "misperception_correction": [
        (Stage.STAGE_1, Stage.STAGE_2, lambda s, n: n >= 1),
        # n>=15 safety net: quiz has 8 questions; 15 turns is generous for uncooperative users.
        (Stage.STAGE_2, Stage.STAGE_3,
         lambda s, n: s.get("questions_answered", 0) >= 8 or n >= 15),
        (Stage.STAGE_3, Stage.STAGE_4,
         lambda s, n: bool(s.get("reflection_shared")) or n >= 4),
        (Stage.STAGE_4, Stage.COMPLETE, lambda s, n: n >= 2),
    ],
    # Content-driven: wait for the user to signal they're winding down (OBSERVE
    # re-evaluates `winding_down` fresh every turn off the user's latest message
    # — it is NOT a sticky fact, so a user who says "actually, one more thing"
    # after appearing done correctly un-flags it). The turn-count floor (n>=2)
    # stops the agent from cutting Stage 1 short on a single message.
    #
    # In practice most users never volunteer an explicit "I'm done" — they just
    # keep answering follow-ups (the base prompt always asks one), so a session
    # relying solely on `winding_down` can run indefinitely. The caps (n>=5 /
    # n>=2) bound that, matching the study's original 3-5 minute target; the
    # STAGE_1 prompt also now proactively checks in around turn 4 instead of
    # waiting for the user to bring up closing.
    "control": [
        # n>=8 floor: match ~10-turn natural depth of structured conditions.
        # n>=12 safety net: ensures completion for fully uncooperative participants.
        (Stage.STAGE_1, Stage.STAGE_4,
         lambda s, n: (bool(s.get("winding_down")) and n >= 8) or n >= 12),
        (Stage.STAGE_4, Stage.COMPLETE,
         lambda s, n: bool(s.get("winding_down")) or n >= 2),
    ],
    "control_politics": [
        (Stage.STAGE_1, Stage.STAGE_4,
         lambda s, n: (bool(s.get("winding_down")) and n >= 8) or n >= 12),
        (Stage.STAGE_4, Stage.COMPLETE,
         lambda s, n: bool(s.get("winding_down")) or n >= 2),
    ],
}


class StageController:
    """Manages workflow stage transitions with a deterministic rule table.

    The transition criteria are pure threshold checks on signals OBSERVE has
    already extracted, so no LLM call is made here. `llm` is accepted for
    backward-compatible construction but is unused.
    """

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    async def evaluate_transition(
        self, state: SessionState, user_message: str
    ) -> Stage:
        """Determine if the workflow stage should advance.

        Mutates `state.stage` (and resets `state.stage_turn_count` on a real
        transition) and returns the resulting stage. Async only to preserve the
        call site's `await`/`create_task` usage; it performs no I/O.
        """
        # COMPLETE is terminal
        if state.stage == Stage.COMPLETE:
            return Stage.COMPLETE

        # Global escape hatch: user explicitly asked to end the conversation.
        # Overrides all per-condition rules so the state is always recorded as
        # COMPLETE (the COMPLETE prompt then generates the proper goodbye).
        if state.signals.get("user_abort"):
            logger.info(
                "User abort detected — forcing COMPLETE (condition=%s, stage=%s)",
                state.strategy,
                state.stage.value,
            )
            state.stage_turn_count = 0
            state.stage = Stage.COMPLETE
            return Stage.COMPLETE

        rules = _TRANSITIONS.get(state.strategy, [])
        n = state.stage_turn_count
        next_stage = state.stage
        for from_stage, to_stage, predicate in rules:
            if state.stage == from_stage and predicate(state.signals, n):
                next_stage = to_stage
                break

        # Defensive: the table is forward-only, but never regress a stage — a
        # backward jump would reset stage_turn_count and risk an infinite loop
        # that never reaches COMPLETE.
        if _STAGE_ORDER.index(next_stage) < _STAGE_ORDER.index(state.stage):
            logger.warning(
                "Rejected backward stage transition %s -> %s",
                state.stage.value,
                next_stage.value,
            )
            return state.stage

        if next_stage != state.stage:
            logger.info(
                "Stage transition: %s -> %s (condition=%s, stage_turn_count=%d)",
                state.stage.value,
                next_stage.value,
                state.strategy,
                n,
            )
            state.stage_turn_count = 0
        state.stage = next_stage
        return next_stage
