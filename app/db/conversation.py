from datetime import datetime, timezone

from app.agent.survey_data import COMMON_IDENTITY_DATA_CARD, QUIZ_QUESTIONS
from app.schema import (
    Message,
    CIObservation,
    PNObservation,
    MCObservation,
    ChatObservation,
    QuizQuestion,
    ControlObservation,
)
from app.db.documents import conversation_docs


def get_conversation(study_id: str) -> dict:
    return conversation_docs.find_one({"study_id": study_id})


def save_turn_log(study_id: str, entry: dict):
    now = datetime.now(timezone.utc)
    signal_snapshot = {
        "turn": entry.get("turn"),
        "stage": entry.get("stage"),
        "timestamp": entry.get("timestamp"),
        "signals": entry.get("signals", {}),
    }
    conversation_docs.update_one(
        {"study_id": study_id},
        {
            "$set": {"payload": entry, "updated_at": now},
            "$push": {"signal_history": signal_snapshot},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


def save_safety_event(study_id: str, verdict: dict):
    now = datetime.now(timezone.utc)
    conversation_docs.update_one(
        {"study_id": study_id},
        {
            "$set": {
                "verdict": verdict,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


def _coerce_messages(messages: list | None) -> list[Message]:
    history: list[Message] = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history.append(Message(role=role, content=content))

    return history


def get_chat_history(study_id: str) -> list:
    conversation_doc = conversation_docs.find_one({"study_id": study_id})

    if not conversation_doc:
        return []

    payload = conversation_doc.get("payload") or {}

    messages = payload.get("messages") or payload.get("message") or []
    return _coerce_messages(messages)


def _get_common_identity_observation(signals: dict) -> CIObservation:
    show_survey = bool(signals.get("exhausted_majority_introduced"))
    survey_text = COMMON_IDENTITY_DATA_CARD
    user_feeling_text = signals.get("user_feeling_text")
    user_media_text = signals.get("user_media_text")
    return CIObservation(
        show_survey=show_survey,
        survey_text=survey_text,
        user_feeling_text=user_feeling_text,
        user_media_text=user_media_text,
    )


def _get_personal_narrative_observation(signals: dict) -> PNObservation:
    person_label = signals.get("person_label")
    person_traits = signals.get("person_traits", [])
    person_cares_about = signals.get("person_cares_about", [])
    person_memories = signals.get("person_memories", [])
    person_political_origin = signals.get("person_political_origin")
    return PNObservation(
        person_label=person_label,
        person_traits=person_traits,
        person_cares_about=person_cares_about,
        person_memories=person_memories,
        person_political_origin=person_political_origin,
    )


def _get_control_observation(signals: dict) -> ControlObservation:
    topics_shared = signals.get("topics_shared", [])
    current_mood = signals.get("current_mood")
    return ControlObservation(topics_shared=topics_shared, current_mood=current_mood)


def _get_misperception_correction_observation(signals: dict) -> MCObservation:
    question_answers = signals.get("question_answers", {})
    questions: list[QuizQuestion] = []

    for question in QUIZ_QUESTIONS:
        question_id = question.get("id")
        if question_id not in question_answers:
            continue

        try:
            user_answer = int(question_answers[question_id])
        except (TypeError, ValueError):
            continue

        questions.append(
            QuizQuestion(
                label=question.get("label"),
                user_answer=user_answer,
                survey_average=question.get("survey_average"),
            )
        )

    return MCObservation(questions=questions)


strategy_observation = {
    "common_identity": _get_common_identity_observation,
    "personal_narrative": _get_personal_narrative_observation,
    "misperception_correction": _get_misperception_correction_observation,
    "control": _get_control_observation,
    "control_politics": _get_control_observation,
}


def get_conversation_observation(study_id: str) -> ChatObservation | None:
    conversation_doc = conversation_docs.find_one({"study_id": study_id})

    if not conversation_doc:
        return None

    payload = conversation_doc.get("payload")

    if not payload:
        return None

    strategy = payload.get("strategy")
    signals = payload.get("signals")

    if not strategy or not signals:
        return None

    stage = payload.get("stage", "stage_1")
    return ChatObservation(stage=stage, observation=strategy_observation[strategy](signals))
