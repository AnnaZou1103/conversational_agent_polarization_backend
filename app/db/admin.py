import os
import random
import string
from datetime import datetime, timezone

from app.db.documents import conversation_docs, conversations_archive, user_docs
from app.schema import GetUserResponse

from app.agent.strategies import Strategy

base_url = os.getenv("PLATFORM_URL")

# How a study user record came into existence. Only "manual" is excluded from
# the balanced round-robin in create_study_user: those are bare /survey opens
# with no CloudResearch participantId (researcher testing, link-preview
# crawlers, stray reloads). They never yield usable data, but counting them
# skews condition assignment for real participants for weeks afterwards — see
# the 2026-08-07 and 2026-08-14 records. Admin-generated links are meant to be
# handed to real participants, so they do count.
SOURCE_CLOUDRESEARCH = "cloudresearch"
SOURCE_ADMIN = "admin"
SOURCE_MANUAL = "manual"

# Study flow order, used to pick the furthest-along record when a participant
# somehow ended up with more than one (all such duplicates predate the
# participant_id lookup below).
_STATE_ORDER = (
    "not_started",
    "pre_survey",
    "to_intervention",
    "intervention",
    "to_post_survey",
    "post_survey",
    "complete",
)


def _existing_study_id(participant_id: str) -> str | None:
    """The study_id already issued to this CloudResearch participant, if any.

    /survey mints a user on every GET and redirects to /{study_id}, so a
    participant who re-opens the CloudResearch link or navigates back to
    /survey used to get a brand new study_id — and a brand new random
    condition. Twelve participants did exactly that, one of them four times,
    and one worked through three conditions until a screener let them in.
    Returning their existing study_id sends them back to /{study_id}, which
    routes on state and drops them where they left off (including the
    screened-out page, so a screen-out stays a screen-out).
    """
    docs = list(user_docs.find({"type": "study", "participant_id": participant_id}))
    if not docs:
        return None
    return max(
        docs,
        key=lambda d: (
            _STATE_ORDER.index(d["state"]) if d.get("state") in _STATE_ORDER else -1,
            d.get("updated_at") or d["created_at"],
        ),
    )["study_id"]



def generate_study_id():
    return "".join(random.choices(string.ascii_letters + string.digits, k=6))


def _assign_strategy(project_id: str | None) -> str:
    """Pick the condition with the fewest participants so far, within this batch.

    The count is scoped to `project_id`, so every CloudResearch project balances
    itself from zero instead of inheriting whatever imbalance the study has
    accumulated. Records with no project_id (manual opens, admin-generated
    links) form their own bucket the same way.

    Ties are broken at random rather than by enum order. With a per-project
    reset every project starts from an all-zero count, so a fixed tie-break
    would hand the first participant of every batch the same condition and make
    the opening sequence identical batch to batch.
    """
    scope = {
        "type": "study",
        "source": {"$ne": SOURCE_MANUAL},
        "project_id": project_id,
    }
    counts = {
        s.value: user_docs.count_documents({**scope, "strategy": s.value})
        for s in Strategy
    }
    fewest = min(counts.values())
    return random.choice([s for s, n in counts.items() if n == fewest])


def create_study_user(
    strategy: str = None,
    participant_id: str = None,
    assignment_id: str = None,
    project_id: str = None,
) -> str:
    """Create a single study user, or return the one this participant already has.

    If strategy is given, use it; otherwise assign via balanced round-robin.
    """
    if participant_id:
        existing = _existing_study_id(participant_id)
        if existing is not None:
            return existing
    source = SOURCE_CLOUDRESEARCH if participant_id else SOURCE_MANUAL
    if strategy is None:
        strategy = _assign_strategy(project_id)
    study_id = generate_study_id()
    user_docs.insert_one(
        {
            "study_id": study_id,
            "type": "study",
            "strategy": strategy,
            "state": "not_started",
            "source": source,
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
                    "source": SOURCE_ADMIN,
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
                "source": SOURCE_ADMIN,
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
