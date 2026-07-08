from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.db.user import (
    study_id_is_valid,
    get_user_state,
    get_user_agent_strategy,
    advance_user_state,
    get_user_party,
    save_user_party,
    get_user_study_type,
)
from app.db.admin import create_study_user
from app.agent.strategies import Strategy
from app.schema import UserState, UserParty

router = APIRouter(prefix="/user", tags=["User"])


@router.post("/create")
def create_user_route(
    strategy: Optional[str] = Query(default=None),
    participantId: Optional[str] = Query(default=None),
    assignmentId: Optional[str] = Query(default=None),
    projectId: Optional[str] = Query(default=None),
):
    valid_strategies = {s.value for s in Strategy}
    if strategy is not None and strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Must be one of: {sorted(valid_strategies)}",
        )
    study_id = create_study_user(
        strategy=strategy,
        participant_id=participantId,
        assignment_id=assignmentId,
        project_id=projectId,
    )
    return {"study_id": study_id}


@router.get("/validate/{study_id}")
def validate_study_id(study_id: str):
    if study_id_is_valid(study_id=study_id):
        return {"message": "Study ID Found"}
    else:
        raise HTTPException(status_code=404, detail="Study ID Not Found")


@router.get("/state/{study_id}")
def get_user_state_route(study_id: str):
    if study_id_is_valid(study_id=study_id):
        curr_state = get_user_state(study_id=study_id)
        return curr_state.model_dump(by_alias=True)
    else:
        raise HTTPException(status_code=404, detail="Study ID Not Found")


@router.get("/agent_strategy/{study_id}")
def get_user_agent_strategy_route(study_id: str):
    if study_id_is_valid(study_id=study_id):
        agent_strategy = get_user_agent_strategy(study_id=study_id)
        return agent_strategy.model_dump(by_alias=True)
    else:
        raise HTTPException(status_code=404, detail="Study ID Not Found")


@router.get("/party/{study_id}")
def get_user_party_route(study_id: str):
    if study_id_is_valid(study_id=study_id):
        party = get_user_party(study_id=study_id)
        if party is None:
            raise HTTPException(status_code=404, detail="Party Not Found")
        return party.model_dump(by_alias=True)
    else:
        raise HTTPException(status_code=404, detail="Study ID Not Found")


@router.post("/advance/{study_id}")
def advance_user_state_route(study_id: str, next_state: UserState):
    if study_id_is_valid(study_id=study_id):
        advance_user_state(study_id=study_id, next_state=next_state)
        return {"message": "Advance User State Successfully"}
    else:
        raise HTTPException(status_code=404, detail="Study ID Not Found")


@router.post("/party/{study_id}")
def save_user_party_route(study_id: str, user_party: UserParty):
    if study_id_is_valid(study_id=study_id):
        save_user_party(study_id=study_id, user_party=user_party)
        return {"message": "Save User Party Successfully"}
    else:
        raise HTTPException(status_code=404, detail="Study ID Not Found")


@router.get("/type/{study_id}")
def get_user_study_type_router(study_id: str):
    if study_id_is_valid(study_id=study_id):
        study_type = get_user_study_type(study_id=study_id)
        return study_type.model_dump(by_alias=True)
    else:
        raise HTTPException(status_code=404, detail="Study ID Not Found")
