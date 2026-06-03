from fastapi import APIRouter
from pydantic import BaseModel
from app.db.experiment import generate_experiment_user


router = APIRouter(prefix="/experiment", tags=["Experiment"])


class GenerateExperimentUserRequest(BaseModel):
    strategy: str | None = None


@router.post("/generate")
def generate_experiment_user_route(body: GenerateExperimentUserRequest = GenerateExperimentUserRequest()):
    """Generate a User for Pre-experiment"""
    study_id = generate_experiment_user(body.strategy)
    return {"id": study_id}
