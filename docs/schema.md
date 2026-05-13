# Schema Reference

## MongoDB

Database: `study_db`  
Collections: `users`, `conversations`

---

### `users` collection

One document per participant. Created by the admin generate endpoints.

```json
{
  "study_id":   "abc123",
  "type":       "study",
  "strategy":   "common_identity",
  "state":      "not_started",
  "party":      null,
  "pre_survey": { "q1": "3", "q2": "5" },
  "post_survey": { "q1": "4", "q2": "5" },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

| Field | Type | Values | Description |
|---|---|---|---|
| `study_id` | string | 6-char alphanumeric | Unique participant ID |
| `type` | string | `study`, `experiment` | `experiment` = pre-study test user |
| `strategy` | string | see conditions | Assigned condition, set at creation, never changes |
| `state` | string | see state machine | Participant's current position in the study flow |
| `party` | string \| null | `republican`, `democrat` | Set during intake; null until collected |
| `pre_survey` | object \| null | `{ question_id: answer }` | Raw survey responses, saved by frontend |
| `post_survey` | object \| null | `{ question_id: answer }` | Raw survey responses, saved by frontend |
| `created_at` | date | ISO 8601 | Document creation time |
| `updated_at` | date | ISO 8601 | Last modification time |

**State machine:**
```
not_started → pre_survey → to_intervention → intervention → to_post_survey → post_survey → complete
```
Transitions are driven by the frontend via `POST /user/advance/{study_id}`.

**Conditions (`strategy`):**
`common_identity`, `personal_narrative`, `misperception_correction`, `control`, `control_politics`

---

### `conversations` collection

One document per participant. Created on the first chat turn. Unique index on `study_id`.

```json
{
  "study_id": "abc123",
  "payload": {
    "turn": 4,
    "stage": "stage_2",
    "stage_turn_count": 2,
    "strategy": "common_identity",
    "political_party": "democrat",
    "timestamp": "2024-01-01T00:05:00Z",
    "system_prompt": "You are a conversational agent...",
    "messages": [
      { "role": "user", "content": "I really dislike Republicans." },
      { "role": "assistant", "content": "I hear you. What shapes that feeling?" },
      { "role": "user", "content": "The news mostly." },
      { "role": "assistant", "content": "How much of your sense of what they're like comes from there?" }
    ],
    "signals": {
      "feeling_expressed": true,
      "user_feeling_text": "frustrated and exhausted",
      "media_mentioned": true,
      "media_distortion_acknowledged": false,
      "exhausted_majority_introduced": false,
      "common_identity_described": false
    }
  },
  "verdict": {
    "action": "reminder",
    "category": "gibberish",
    "reason": "gibberish / invalid input detected",
    "user_message_excerpt": "asdfasdf",
    "consecutive_reminders": 1,
    "indecent_count": 0,
    "invalid_count": 1,
    "timestamp": "2024-01-01T00:03:00Z"
  },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:05:00Z"
}
```

#### `payload` (overwritten each turn)

| Field | Type | Description |
|---|---|---|
| `turn` | int | Total turn count at the time of this write |
| `stage` | string | Current agent stage (`stage_1`–`stage_4`, `complete`) |
| `stage_turn_count` | int | Turns within the current stage |
| `strategy` | string | Assigned condition |
| `political_party` | string \| null | `republican` or `democrat` |
| `timestamp` | string | ISO 8601 timestamp of this turn |
| `system_prompt` | string | Full system prompt sent to the LLM for this turn |
| `messages` | array | Full conversation history including the assistant reply just generated |
| `signals` | object | Accumulated condition-specific signals (see below) |

`payload` is fully overwritten on every turn — it always reflects the latest state.

#### `verdict` (overwritten on safety events)

Only present if a safety event has occurred. Overwritten on each new safety event.

| Field | Type | Description |
|---|---|---|
| `action` | string | `reminder` or `terminate` |
| `category` | string | `gibberish`, `indecent`, `both` |
| `reason` | string | Human-readable reason |
| `user_message_excerpt` | string | First 200 chars of the offending message |
| `consecutive_reminders` | int | Streak count at time of event |
| `indecent_count` | int | Lifetime indecent event count |
| `invalid_count` | int | Lifetime gibberish event count |
| `timestamp` | string | ISO 8601 |

---

### Signals by condition

Signals are extracted by the OBSERVE step and accumulated across turns. Stored inside `payload.signals`.

**`common_identity`**

| Signal | Type | Description |
|---|---|---|
| `feeling_expressed` | bool | User has expressed a genuine emotional feeling about the opposing party |
| `user_feeling_text` | string | Short phrase capturing the user's feeling (max 12 words) |
| `media_mentioned` | bool | User mentioned news/social media as a source |
| `user_media_text` | string | Short phrase capturing what the user said about media |
| `media_distortion_acknowledged` | bool | User gestured toward media not being representative |
| `exhausted_majority_introduced` | bool | Agent shared the survey data card OR user described the exhausted majority independently |
| `common_identity_described` | bool | User described a cross-partisan group of ordinary people exhausted with division |

**`personal_narrative`**

| Signal | Type | Description |
|---|---|---|
| `person_label` | string | Label the user chose for the person (`"my uncle"`, `"Sarah"`) |
| `person_is_real` | bool | Whether the person is real (vs. imagined) |
| `person_details_count` | int | Count of distinct personal details shared |
| `origins_explored` | bool | User has discussed or speculated about why the person holds their views |
| `person_traits` | string[] | Personality traits mentioned (`["stubborn", "caring"]`) |
| `person_cares_about` | string[] | Things the person cares about (`["family", "job security"]`) |
| `person_memories` | string[] | Specific memories or anecdotes |
| `person_political_origin` | string | Why the user thinks this person holds their political views |

**`misperception_correction`**

| Signal | Type | Description |
|---|---|---|
| `intro_completed` | bool | User agreed to start the quiz |
| `questions_answered` | int | Number of questions where both user answered AND agent revealed the finding (0–8) |
| `question_answers` | object | `{ "q1": 2, "q2": 3, ... }` — Likert answer (1–4) per question |
| `reflection_shared` | bool | User shared an overall reaction after all 8 questions |

**`control` / `control_politics`**

| Signal | Type | Description |
|---|---|---|
| `topics_shared` | string[] | Topics the user has mentioned (accumulates across turns) |
| `current_mood` | string | Overall mood or sentiment from the most recent turns |

---

## API Request / Response Models

Defined in [app/schema.py](../app/schema.py). All models use camelCase aliases for JSON serialization.

### `ChatCompletionRequest`

```json
{
  "studyId": "abc123",
  "model": "common-identity",
  "message": { "role": "user", "content": "Hello" },
  "stream": false
}
```

| Field | Type | Values |
|---|---|---|
| `studyId` | string | Required |
| `model` | string | `common-identity`, `personal-narrative`, `misperception-correction`, `control`, `control-politics` |
| `message` | object | `{ role, content }` — content can be a string or OpenAI multimodal array |
| `stream` | bool | Default `false` |

### `UserState`

```json
{ "state": "intervention" }
```

Values: `not_started`, `pre_survey`, `to_intervention`, `intervention`, `to_post_survey`, `post_survey`, `complete`

### `AgentStrategy`

```json
{ "strategy": "common_identity" }
```

Values: `common_identity`, `personal_narrative`, `misperception_correction`, `control`, `control_politics`

### `UserParty`

```json
{ "party": "democrat" }
```

Values: `democrat`, `republican`

### `StudyType`

```json
{ "type": "study" }
```

Values: `study`, `experiment`

### `SurveyResponses`

```json
{ "responses": { "q1": "3", "q2": "5", "q3": "2" } }
```

Free-form key-value map — no schema enforced on the response content.

### `AdminRequest` / `GenerateUserRequest` / `GenerateUserByStrategyRequest`

```json
{ "password": "secret", "count": 10, "strategy": "common_identity" }
```

`password` is required on all admin endpoints. `count` and `strategy` apply only where relevant.

### Observation responses (`GET /observation/{study_id}`)

Shape varies by condition — see [API Reference](../README.md#get-observationstudy_id) in the README.
