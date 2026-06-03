import random
from datetime import datetime, timezone

from app.db.admin import generate_study_id
from app.db.documents import user_docs

from app.agent.strategies import Strategy


def generate_experiment_user(strategy: str | None = None) -> str:
    """Generate a user for pre-experiment"""
    study_id = generate_study_id()
    resolved_strategy = strategy if strategy else random.choice(list(Strategy)).value
    user_docs.insert_one(
        {
            "study_id": study_id,
            "type": "experiment",
            "strategy": resolved_strategy,
            "state": "pre_survey",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return study_id
