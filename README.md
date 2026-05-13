# Partisan Animosity Study — Backend API

FastAPI backend for a conversational agent research study on partisan animosity reduction. Participants are assigned to one of five conditions and guided through a structured multi-stage conversation with an LLM agent. Pre/post surveys and conversation data are persisted in MongoDB.

## Study Conditions

Each participant is randomly assigned to one condition, mapped to the `model` field in chat requests:

| Model ID | Condition | Description |
|---|---|---|
| `common-identity` | Common Identity | Guides the user to recognize media distortion and shared exhaustion with polarization |
| `personal-narrative` | Personal Narrative | Helps the user reflect on a specific person they know from the opposing party |
| `misperception-correction` | Misperception Correction | Reveals accurate survey data to correct the user's misperceptions about the opposing party |
| `control` | Control (Wellbeing) | Mental health check-in with no political content |
| `control-politics` | Control (Politics) | Open-ended political conversation with no guided intervention |

## Agent Architecture

Each conversation follows a 4-stage workflow (`stage_1` → `stage_2` → `stage_3` → `stage_4` → `complete`). A `StageController` uses an LLM call to evaluate transition criteria (turn counts + behavioral signals) after each user message. Transition rules differ per condition.

## API Reference

---

### Chat

#### `POST /v1/chat/completions`

OpenAI-compatible chat endpoint. The frontend sends each new user message here; the backend loads history from MongoDB, runs the agent pipeline, and streams or returns the assistant reply.

**Request**
```json
{
  "studyId": "abc123",
  "model": "common-identity",
  "message": { "role": "user", "content": "I really dislike Republicans." },
  "stream": false
}
```

**Response (non-streaming)**
```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1715000000,
  "model": "common-identity",
  "studyId": "abc123",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "I hear you. Can you tell me more about what shapes that feeling?" },
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

**Response (streaming)** — Server-Sent Events in OpenAI chunk format:
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"I hear"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

#### `GET /chat/history/{study_id}`

Returns the full message history for a participant.

**Response**
```json
[
  { "role": "user", "content": "I really dislike Republicans." },
  { "role": "assistant", "content": "I hear you. Can you tell me more about what shapes that feeling?" }
]
```

---

#### `GET /observation/{study_id}`

Returns structured data the frontend can use to render condition-specific UI elements (e.g. survey cards, quiz results). Shape varies by condition.

**Response — `common_identity`**
```json
{
  "observation": {
    "showSurvey": true,
    "surveyText": "In a recent national survey, most Americans — across party lines — said they feel exhausted with political division...",
    "userFeelingText": "frustrated and unheard",
    "userMediaText": "cable news"
  }
}
```

**Response — `personal_narrative`**
```json
{
  "observation": {
    "personLabel": "my uncle Dave",
    "personTraits": ["hardworking", "funny"],
    "personCaresAbout": ["family", "job security"],
    "personMemories": ["fishing trips as a kid"],
    "personPoliticalOrigin": "grew up in a rural town where everyone voted Republican"
  }
}
```

**Response — `misperception_correction`**
```json
{
  "observation": {
    "questions": [
      { "label": "Banning FAR-LEFT group rallies", "userAnswer": 3, "surveyAverage": 2.0 },
      { "label": "Using violence to block major laws", "userAnswer": 4, "surveyAverage": 3.0 }
    ]
  }
}
```

**Response — `control` / `control_politics`**
```json
{
  "observation": {
    "topicsShared": ["work stress", "family"],
    "currentMood": "anxious"
  }
}
```

---

### User

User state is managed by the frontend — the backend stores and returns it without enforcing transition logic.

State machine: `not_started` → `pre_survey` → `to_intervention` → `intervention` → `to_post_survey` → `post_survey` → `complete`

---

#### `GET /user/validate/{study_id}`

Checks whether a study ID exists in the database. Called by the frontend on page load.

**Response (found)**
```json
{ "message": "Study ID Found" }
```
**Response (not found)** — `404`
```json
{ "detail": "Study ID Not Found" }
```

---

#### `GET /user/state/{study_id}`

Returns the participant's current position in the study flow.

**Response**
```json
{ "state": "intervention" }
```
Possible values: `not_started`, `pre_survey`, `to_intervention`, `intervention`, `to_post_survey`, `post_survey`, `complete`

---

#### `GET /user/agent_strategy/{study_id}`

Returns the condition assigned to this participant.

**Response**
```json
{ "strategy": "common_identity" }
```
Possible values: `common_identity`, `personal_narrative`, `misperception_correction`, `control`, `control_politics`

---

#### `GET /user/party/{study_id}`

Returns the participant's self-reported political party.

**Response**
```json
{ "party": "democrat" }
```
Possible values: `democrat`, `republican`. Returns `404` if not yet set.

---

#### `GET /user/type/{study_id}`

Returns whether this is a real study participant or a pre-experiment test user.

**Response**
```json
{ "type": "study" }
```
Possible values: `study`, `experiment`

---

#### `POST /user/advance/{study_id}`

Moves the participant to the specified state. The frontend calls this after completing each step (e.g. after submitting the pre-survey).

**Request**
```json
{ "state": "to_intervention" }
```

**Response**
```json
{ "message": "Advance User State Successfully" }
```

---

#### `POST /user/party/{study_id}`

Saves the participant's political party (collected during intake).

**Request**
```json
{ "party": "democrat" }
```

**Response**
```json
{ "message": "Save User Party Successfully" }
```

---

### Survey

Survey responses are a free-form key-value map — the backend stores them as-is for offline analysis.

#### `POST /survey/pre/{study_id}`
#### `POST /survey/post/{study_id}`

**Request**
```json
{
  "responses": {
    "q1": "3",
    "q2": "5",
    "q3": "2"
  }
}
```

**Response**
```json
{ "message": "Save Pre-Survey Responses Successfully" }
```

---

### Admin (password-protected)

All admin endpoints require a `password` field in the request body matching `ADMIN_PASSWORD`. Returns `403` on mismatch.

---

#### `POST /admin/generate`

Generates `count` participants for **each** condition (5 conditions × count = total users created).

**Request**
```json
{ "password": "secret", "count": 10 }
```
**Response**
```json
{ "message": "Generate Users Successfully" }
```

---

#### `POST /admin/agent_strategy/generate`

Generates `count` participants for one specific condition.

**Request**
```json
{ "password": "secret", "strategy": "common_identity", "count": 10 }
```
**Response**
```json
{ "message": "Generate Users for common_identity Successfully" }
```

---

#### `POST /admin/agent_strategy/list/users`

Lists all participants in a given state and condition, with their study URLs.

**Request**
```json
{ "password": "secret", "state": "not_started", "strategy": "common_identity" }
```
**Response**
```json
[
  { "studyId": "abc123", "url": "https://your-platform.com/abc123" },
  { "studyId": "def456", "url": "https://your-platform.com/def456" }
]
```

---

#### `DELETE /admin/delete/all`

Deletes all participants and their conversation data.

**Request**
```json
{ "password": "secret" }
```
**Response**
```json
"Delete 50 Users Successfully"
```

---

#### `DELETE /admin/delete/{study_id}`

Deletes one participant and their conversation data.

**Request**
```json
{ "password": "secret" }
```
**Response**
```json
"Delete Users abc123 Successfully"
```

---

#### `POST /admin/reset/{study_id}`

Resets a participant's state to `not_started`, clears their party, and deletes their conversation history. Useful for re-running a participant.

**Request**
```json
{ "password": "secret" }
```
**Response**
```json
"Reset user abc123 successfully"
```

---

### Experiment

#### `POST /experiment/generate`

Creates a single test participant (type `experiment`) with a randomly assigned condition. Used for pre-study testing.

**Response**
```json
{ "id": "abc123" }
```

---

### Other

#### `GET /`
```json
{ "message": "Hello from FastAPI on Heroku/Vercel" }
```

#### `GET /health` — `204 No Content`

#### `GET /v1/models`
```json
{
  "object": "list",
  "data": [
    { "id": "common-identity", "object": "model", "owned_by": "partisan-animosity-study" },
    { "id": "personal-narrative", "object": "model", "owned_by": "partisan-animosity-study" },
    { "id": "misperception-correction", "object": "model", "owned_by": "partisan-animosity-study" },
    { "id": "control", "object": "model", "owned_by": "partisan-animosity-study" },
    { "id": "control-politics", "object": "model", "owned_by": "partisan-animosity-study" }
  ]
}
```

---

## Setup

### Requirements

- Python 3.11+
- MongoDB instance

### Environment Variables

Create a `.env` file:

```env
# LLM provider: "openai", "anthropic", or "azure"
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-...
LLM_BASE_URL=               # optional, for OpenAI-compatible endpoints
LLM_API_VERSION=2024-12-01-preview  # Azure only

# MongoDB
MONGODB_URI=mongodb+srv://...

# Admin
ADMIN_PASSWORD=your-admin-password

# Server
API_HOST=0.0.0.0
API_PORT=8080
LOG_LEVEL=info
DEFAULT_STRATEGY=common_identity
ENABLE_THINK=false
```

### Install & Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## Deployment

**Heroku** — uses `Procfile`:
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Vercel** — uses `vercel.json` with `@vercel/python` runtime.

CORS is configured for `http://localhost:3000` and `https://conversational-agent-polarization.vercel.app`.
