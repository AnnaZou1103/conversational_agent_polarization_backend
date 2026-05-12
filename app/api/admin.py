import os
from fastapi import APIRouter, HTTPException
from app.db.admin import (
    generate_users,
    generate_users_by_agent_strategy,
    get_users_by_state_and_strategy,
    delete_all_users,
    delete_user_by_id,
    reset_user,
)
from app.db.user import study_id_is_valid
from app.schema import (
    AdminRequest,
    GenerateUserRequest,
    GenerateUserByStrategyRequest,
    GetUserByStrategyRequest,
)

router = APIRouter(prefix="/admin", tags=["Admin"])
admin_password = os.getenv("ADMIN_PASSWORD")


@router.post("/generate")
def generate_users_route(request: GenerateUserRequest):
    """Generate User for each Strategy"""
    if request.password != admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    generate_users(count=request.count)
    return {"message": "Generate Users Successfully"}


@router.post("/agent_strategy/generate")
def generate_users_by_agent_strategy_route(request: GenerateUserByStrategyRequest):
    """Generate User for Specified Strategy"""
    if request.password != admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    generate_users_by_agent_strategy(strategy=request.strategy, count=request.count)
    return {"message": f"Generate Users for {request.strategy} Successfully"}


@router.post("/agent_strategy/list/users")
def get_users_by_state_and_strategy_route(request: GetUserByStrategyRequest):
    """Get user list by state and strategy"""
    if request.password != admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    return get_users_by_state_and_strategy(
        state=request.state, strategy=request.strategy
    )


@router.delete("/delete/all")
def delete_all_users_route(request: AdminRequest):
    """Delete all users and their associated conversations/messages."""
    if request.password != admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    delete_count = delete_all_users()
    return f"Delete {delete_count} Users Successfully"


@router.delete("/delete/{study_id}")
def delete_user_route(study_id: str, request: AdminRequest):
    """Delete one user and associated conversations/messages by study_id."""
    if request.password != admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    if not study_id_is_valid(study_id=study_id):
        raise HTTPException(status_code=404, detail="Study ID Not Found")
    delete_user_by_id(study_id=study_id)
    return f"Delete Users {study_id} Successfully"


@router.post("/reset/{study_id}")
def reset_user_route(study_id: str, request: AdminRequest):
    """Reset a user's state and clear their conversation history."""
    if request.password != admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    if not study_id_is_valid(study_id=study_id):
        raise HTTPException(status_code=404, detail="Study ID Not Found")
    reset_user(study_id=study_id)
    return f"Reset user {study_id} successfully"
