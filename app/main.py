import logging

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, user, survey, chat, models, experiment
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="Partisan Animosity Study API",
    description="Research study agents: Common Identity and Personal Narrative conditions.",
    version="0.1.0",
)

origins = [
    "http://localhost:3000",
    "https://conversational-agent-polarization.vercel.app",
    "https://conversational-agent-frontend-b8f9b1baaad0.herokuapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router)
app.include_router(user.router)
app.include_router(survey.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(experiment.router)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Heroku/Vercel"}


@app.get("/health", status_code=status.HTTP_204_NO_CONTENT)
def server_status() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)
