from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Dict, Literal, Optional, Union


class CaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AdminRequest(CaseModel):
    password: str


class GenerateUserRequest(AdminRequest):
    count: int


class GenerateUserByStrategyRequest(GenerateUserRequest):
    strategy: Literal[
        "common_identity",
        "personal_narrative",
        "misperception_correction",
        "control",
        "control_politics",
    ]


class GetUserByStrategyRequest(AdminRequest):
    state: Literal[
        "not_started",
        "pre_survey",
        "to_intervention",
        "intervention",
        "to_post_survey",
        "post_survey",
        "complete",
    ]
    strategy: Literal[
        "common_identity",
        "personal_narrative",
        "misperception_correction",
        "control",
        "control_politics",
    ]


class GetUserResponse(CaseModel):
    study_id: str
    url: str


class StudyType(CaseModel):
    type: Literal["study", "experiment"]


class UserState(CaseModel):
    state: Literal[
        "not_started",
        "pre_survey",
        "to_intervention",
        "intervention",
        "to_post_survey",
        "post_survey",
        "complete",
    ]
    screened: bool = False


class AgentStrategy(CaseModel):
    strategy: Literal[
        "common_identity",
        "personal_narrative",
        "misperception_correction",
        "control",
        "control_politics",
    ]


class SurveyResponses(CaseModel):
    responses: Dict[str, str]


class UserParty(CaseModel):
    party: Literal["republican", "democrat"]


class ChatRequest(CaseModel):
    message: str


class ChatResponse(CaseModel):
    type: Literal["token", "done"]
    content: Optional[str]


class Message(CaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


class ChatMessage(CaseModel):
    role: str
    content: str | list

    def text_content(self) -> str:
        """Extract plain text from content, whether string or list format."""
        if isinstance(self.content, str):
            return self.content
        # Handle OpenAI multi-modal format: [{"type": "text", "text": "..."}]
        parts = []
        for item in self.content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)


class ChatCompletionRequest(CaseModel):
    study_id: str
    model: Literal[
        "common-identity",
        "personal-narrative",
        "misperception-correction",
        "control",
        "control-politics",
    ] = "common-identity"
    message: ChatMessage
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class CIObservation(CaseModel):
    user_feeling_text: Optional[str] = None
    user_media_text: Optional[str] = None
    additional_common_ground_text: Optional[str] = None


class PNObservation(CaseModel):
    person_label: Optional[str] = None
    person_traits: list[str] = []
    person_cares_about: list[str] = []
    person_memories: list[str] = []
    person_political_origin: Optional[str] = None


class QuizQuestion(CaseModel):
    label: str
    user_answer: int
    survey_average: str


class MCObservation(CaseModel):
    questions: list[QuizQuestion]


class ControlObservation(CaseModel):
    topics_shared: list[str] = []
    current_mood: Optional[str] = None
    main_takeaway: Optional[str] = None


class ControlPoliticsObservation(CaseModel):
    topics_shared: list[str] = []
    current_mood: Optional[str] = None
    main_concern: Optional[str] = None


class ChatObservation(CaseModel):
    stage: str
    observation: Union[CIObservation, PNObservation, MCObservation, ControlObservation, ControlPoliticsObservation]
