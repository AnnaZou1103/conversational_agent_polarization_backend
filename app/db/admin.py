import os
import random
import string
from datetime import datetime, timezone

from app.db.documents import conversation_docs, conversations_archive, user_docs
from app.schema import GetUserResponse

from app.agent.strategies import Strategy

base_url = os.getenv("PLATFORM_URL")


def generate_study_id():
    return "".join(random.choices(string.ascii_letters + string.digits, k=6))


def create_study_user(
    strategy: str = None,
    participant_id: str = None,
    assignment_id: str = None,
    project_id: str = None,
) -> str:
    """Create a single study user. If strategy is given, use it; otherwise assign via balanced round-robin."""
    if strategy is None:
        strategies = [s.value for s in Strategy]
        counts = {s: user_docs.count_documents({"type": "study", "strategy": s}) for s in strategies}
        strategy = min(counts, key=counts.get)
    study_id = generate_study_id()
    user_docs.insert_one(
        {
            "study_id": study_id,
            "type": "study",
            "strategy": strategy,
            "state": "not_started",
            "participant_id": participant_id,
            "assignment_id": assignment_id,
            "project_id": project_id,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return study_id


def generate_users(count: int):
    for stragegy in list(Strategy):
        user_docs.insert_many(
            [
                {
                    "study_id": generate_study_id(),
                    "type": "study",
                    "strategy": stragegy.value,
                    "state": "not_started",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                for _ in range(count)
            ]
        )


def generate_users_by_agent_strategy(strategy: str, count: int):
    user_docs.insert_many(
        [
            {
                "study_id": generate_study_id(),
                "type": "study",
                "strategy": strategy,
                "state": "not_started",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            for _ in range(count)
        ]
    )


def get_users_by_state_and_strategy(state: str, strategy: str) -> list:
    cursor = user_docs.find(
        {"state": state, "strategy": strategy},
        {
            "_id": 0,
            "study_id": 1,
        },
    )

    return [
        GetUserResponse(
            study_id=user_doc["study_id"], url=f"{base_url}/{user_doc['study_id']}"
        )
        for user_doc in cursor
    ]


def delete_all_users() -> int:
    deleted_users = user_docs.delete_many({}).deleted_count
    conversation_docs.delete_many({})
    return deleted_users


def delete_user_by_id(study_id: str) -> int:
    user_docs.delete_one({"study_id": study_id}).deleted_count
    conversation_docs.delete_many({"study_id": study_id})


def reset_user(study_id: str):
    user_docs.update_one(
        {"study_id": study_id},
        {
            "$set": {"state": "not_started", "party": None},
            "$currentDate": {"updated_at": True},
        },
    )
    doc = conversation_docs.find_one({"study_id": study_id})
    if doc:
        doc.pop("_id", None)
        doc["reset_at"] = datetime.now(timezone.utc)
        conversations_archive.insert_one(doc)
    conversation_docs.delete_many({"study_id": study_id})
