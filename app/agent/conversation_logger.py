from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.agent.state import SessionState
from app.agent.safety import SafetyVerdict
from app.db.conversation import save_turn_log, save_safety_event
from app.llm.base import Message

logger = logging.getLogger(__name__)

# Signal keys that are ever surfaced to the participant, directly or via the
# /observation endpoint's *Observation schemas (see app/db/conversation.py
# _get_*_observation helpers). Everything else in state.signals is internal
# stage-gating bookkeeping (see app/agent/phases.py) and research-analysis
# data never shown in the UI. Kept as a separate constant so
# visible_signals can be trimmed without touching the full `signals` field,
# which build_session_state() must read back unchanged to keep stage
# transitions working across stateless requests.
VISIBLE_SIGNAL_KEYS = {
    "exhausted_majority_introduced",  # -> CIObservation.show_survey
    "user_feeling_text",
    "user_media_text",
    "person_label",
    "person_traits",
    "person_cares_about",
    "person_memories",
    "person_political_origin",
    "topics_shared",
    "current_mood",
    "main_takeaway",
    "main_concern",
    "question_answers",
}


def _filter_visible_signals(signals: dict) -> dict:
    return {k: v for k, v in signals.items() if k in VISIBLE_SIGNAL_KEYS}


def log_safety_event(
    conversations_dir: str,
    state: SessionState,
    verdict: SafetyVerdict,
) -> None:
    """Append a safety event to a sidecar JSONL file for the session.

    Separate file (`{session_id}_safety.jsonl`) keeps research transcripts
    clean from moderation artifacts.
    """
    try:
        study_id = state.metadata.get("study_id") or state.study_id
        if study_id:
            save_safety_event(
                study_id=study_id,
                verdict=verdict.to_dict(),
            )

    except Exception as e:
        logger.warning("Failed to log safety event: %s", e)


def log_turn(
    conversations_dir: str,
    state: SessionState,
    system_prompt: str,
    messages: list[Message],
    response: str,
) -> None:
    """Append a single conversation turn to the session's JSONL log file."""
    try:
        # path = Path(conversations_dir)
        # path.mkdir(parents=True, exist_ok=True)

        now_iso = datetime.now(timezone.utc).isoformat()
        entry = {
            "turn": state.turn_count,
            "stage": state.stage.value,
            "stage_turn_count": state.stage_turn_count,
            "strategy": state.strategy,
            "political_party": state.political_party,
            "timestamp": now_iso,
            "system_prompt": system_prompt,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp or now_iso,
                }
                for m in messages
            ]
            + [{"role": "assistant", "content": response, "timestamp": now_iso}],
            "signals": dict(state.signals),
            "visible_signals": _filter_visible_signals(state.signals),
        }

        study_id = state.metadata.get("study_id") or state.study_id
        if study_id:
            save_turn_log(
                study_id=study_id,
                entry=entry,
            )

        # log_file = path / f"{state.session_id}.jsonl"
        # with log_file.open("a", encoding="utf-8") as f:
        #     f.write(json.dumps(entry) + "\n")

    except Exception as e:
        logger.warning("Failed to log conversation turn: %s", e)
