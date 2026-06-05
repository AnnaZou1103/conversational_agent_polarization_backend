from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.db.user import get_user_party
from app.db.conversation import get_conversation


class Stage(str, Enum):
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"
    STAGE_4 = "stage_4"
    COMPLETE = "complete"


@dataclass
class SessionState:
    study_id: str
    stage: Stage = Stage.STAGE_1
    strategy: str = "common_identity"  # fixed for the session
    political_party: str | None = None  # "republican" or "democrat", set during intake
    stage_turn_count: int = 0  # turns within the current stage (resets on transition)
    # Condition-specific signals extracted by OBSERVE step
    signals: dict = field(default_factory=dict)
    memory: list[dict] = field(default_factory=list)
    turn_count: int = 0
    metadata: dict = field(default_factory=dict)
    consecutive_reminders: int = (
        0  # streak of consecutive reminders; resets on clean message
    )
    # --- Safety monitor state (see app/agent/safety.py) ---
    invalid_count: int = 0  # lifetime gibberish tally (logging only)
    indecent_count: int = 0  # lifetime indecent tally (logging only)
    terminated_by_safety: bool = False  # short-circuit flag for re-entry
    safety_history: list[dict] = field(default_factory=list)
    previous_user_message: str | None = None  # for exact-repeat detection


def build_session_state(
    study_id: str, strategy: str, messages: list[dict]
) -> SessionState:
    """Reconstruct SessionState from DB payload and current messages."""
    conversation = get_conversation(study_id)
    user_party = get_user_party(study_id)
    political_party = user_party.party if user_party is not None else None

    state = SessionState(
        study_id=study_id, strategy=strategy, political_party=political_party
    )

    # Populate from latest DB entry
    if conversation and conversation.get("payload"):
        payload = conversation["payload"]
        raw_stage = payload.get("stage", "stage_1")
        state.stage = Stage(raw_stage)
        # Restore the per-stage turn counter. It is stateless across requests,
        # so without this it always resets to 0 (then increments to 1), and any
        # stage gate requiring stage_turn_count >= 2/3 could never fire — leaving
        # conditions stuck short of COMPLETE and the "continue to survey" button
        # never appearing.
        state.stage_turn_count = payload.get("stage_turn_count", 0)
        state.political_party = political_party
        if political_party:
            state.signals["political_party"] = political_party
        saved_signals = payload.get("signals", {})
        if isinstance(saved_signals, dict):
            state.signals.update(saved_signals)
        # system_prompt is also in payload if needed
        state.metadata["last_observation"] = payload.get("last_observation", {})

    verdict = (conversation or {}).get("verdict")
    if isinstance(verdict, dict):
        state.consecutive_reminders = verdict.get("consecutive_reminders", 0)
        state.indecent_count = verdict.get("indecent_count", 0)
        state.invalid_count = verdict.get("invalid_count", 0)
        state.safety_history.append(verdict)

        if verdict.get("action") == "terminate":
            state.terminated_by_safety = True
            state.stage = Stage.COMPLETE

    # Count turns from messages
    state.turn_count = sum(1 for m in messages if m.get("role") == "user")

    return state
