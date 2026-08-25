# Schema Reference

## MongoDB

Database: `study_db`  
Collections: `users`, `conversations`

---

### `users` collection

One document per participant. Created either on intake (`POST /user/create`,
which the `/survey` landing page calls on every request) or by the admin
generate endpoints.

```json
{
  "study_id":   "abc123",
  "type":       "study",
  "strategy":   "common_identity",
  "state":      "not_started",
  "source":     "cloudresearch",
  "participant_id": "60B5007270544B84BD9D615FDF5635D0",
  "assignment_id":  null,
  "project_id":     null,
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
| `source` | string | `cloudresearch`, `admin`, `manual` | How the record was created — see below |
| `participant_id` | string \| null | 32-char alphanumeric | CloudResearch participant, from the `participantId` query param |
| `assignment_id` | string \| null | | CloudResearch `assignmentId` query param |
| `project_id` | string \| null | | CloudResearch `projectId` query param |
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

The agent pipeline also writes directly to this document (bypassing the
endpoint) when a session is force-completed by the [time limit](#conversations-collection):
`state` is set to `complete` and `screened` to `true`, so the frontend's
existing screened-out routing shows its conclusion interface instead of the
normal post-survey flow.

**Conditions (`strategy`):**
`common_identity`, `personal_narrative`, `misperception_correction`, `control`, `control_politics`

**Condition assignment and `source`:**

When `POST /user/create` is called without an explicit `strategy`, the condition
is assigned by balanced round-robin: count the existing `type: "study"` documents
per condition and take the smallest, breaking ties at random.

The count is scoped to the request's `project_id`, so **every CloudResearch
project balances itself from zero** rather than inheriting the study's running
totals. Records with no `project_id` (manual opens, admin-generated links) form
their own bucket the same way. Ties are random because a per-project reset
starts every batch from an all-zero count, and a fixed tie-break would hand the
first participant of every batch the same condition.

The trade-off is that a batch whose size is not a multiple of five leaves a
remainder inside that batch, and no later batch corrects it — per-project
balance is bought at the cost of cross-project balance.

`source` records how the document came into existence, and controls whether it
participates in that count:

| `source` | Created by | Counted for balancing |
|---|---|---|
| `cloudresearch` | `/survey` with a `participantId` query param — a real participant | yes |
| `admin` | `POST /admin/generate` or `POST /admin/agent_strategy/generate` — pre-generated links meant to be handed out | yes |
| `manual` | `/survey` with no `participantId` — researcher testing, link-preview crawlers, stray reloads | **no** |

`manual` records are excluded because `/survey` mints a document on every GET,
so a single stray page load permanently shifts the counter. Two such batches
(2026-08-07 and 2026-08-14) skewed assignment for real participants for over two
weeks before the exclusion was added. Documents predating the `source` field are
counted (the query excludes only `manual`, it does not require the field).

Note that balancing counts *assignments*, not usable data: screened-out and
abandoned sessions still occupy a slot, so equal assignment counts do not imply
equal completed-and-valid N per condition.

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
      { "role": "user", "content": "I really dislike Republicans.", "timestamp": "2024-01-01T00:04:50Z" },
      { "role": "assistant", "content": "I hear you. What shapes that feeling?", "timestamp": "2024-01-01T00:04:52Z" },
      { "role": "user", "content": "The news mostly.", "timestamp": "2024-01-01T00:05:00Z" },
      { "role": "assistant", "content": "How much of your sense of what they're like comes from there?", "timestamp": "2024-01-01T00:05:00Z" }
    ],
    "signals": {
      "feeling_expressed": true,
      "user_feeling_text": "frustrated and exhausted",
      "user_media_text": "mostly cable news and Twitter",
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
| `messages` | array | Full conversation history including the assistant reply just generated; each entry has `role`, `content`, and `timestamp` (ISO 8601, when that message was sent/generated) |
| `signals` | object | Accumulated condition-specific signals (see below) |

`payload` is fully overwritten on every turn — it always reflects the latest state.

#### `verdict` (overwritten on safety events)

Only present if a safety event has occurred. Overwritten on each new safety event.

| Field | Type | Description |
|---|---|---|
| `action` | string | `reminder` or `terminate` |
| `category` | string | `gibberish`, `indecent`, `both`, `time_limit` |
| `reason` | string | Human-readable reason |
| `user_message_excerpt` | string | First 200 chars of the offending message |
| `consecutive_reminders` | int | Streak count at time of event |
| `indecent_count` | int | Lifetime indecent event count |
| `invalid_count` | int | Lifetime gibberish event count |
| `timestamp` | string | ISO 8601 |

`category: "time_limit"` marks an early termination caused by the session exceeding `max_session_minutes` (config, default 1440 = 24h), measured from the participant's first message timestamp. Unlike gibberish/indecent terminations, the participant is not shown an abrupt termination message — the conversation is force-completed through the normal closing flow (`payload.stage` becomes `"complete"`) so the LLM still generates a warm closing reply.

Both this and the consecutive-reminder safety termination (`action: "terminate"`, category `gibberish`/`indecent`/`both`) also update the participant's `users` document to `state: "complete"`, `screened: true` (see [users collection](#users-collection)), so the frontend routes them to its screened-out conclusion interface rather than the normal post-survey flow. This `users` write is best-effort and separate from the `verdict`/`payload` writes above — a failure to flag `screened` does not block or alter the conversation's own closing message.

---

### Signals by condition

Signals are extracted by the OBSERVE step and accumulated across turns, stored
inside `payload.signals`. A few are instead computed deterministically in the
pipeline (noted below). `user_abort` (bool — user asked to end the conversation)
is extracted for every condition and is omitted from the per-condition tables.

**`common_identity`**

| Signal | Type | Description |
|---|---|---|
| `feeling_expressed` | bool | User has expressed a genuine emotional feeling about the opposing party |
| `user_feeling_text` | string | Short phrase capturing the user's feeling (max 12 words) |
| `user_media_text` | string | Short phrase capturing what the user said about media/their sources |
| `media_distortion_acknowledged` | bool | User substantively described HOW media distorts their picture of the opposing party |
| `media_distortion_attempted` | bool | Lower bar: user engaged the media topic at all, even a bare verdict |
| `exhausted_majority_introduced` | bool | Agent shared the survey data card OR user described the exhausted majority independently |
| `common_identity_described` | bool | Mandatory Stage-3 question asked AND user substantively described the shared cross-partisan feeling/want |
| `common_identity_attempted` | bool | Lower bar: mandatory Stage-3 question asked AND user gave a brief on-topic answer |
| `identity_label_pushback` | bool | User explicitly rejected the "common ground"/"shared identity" label (sticky) |
| `additional_common_ground_surfaced` | bool | In Stage-4 extension, user named another form of connection beyond shared exhaustion |
| `additional_common_ground_text` | string | Short phrase capturing that additional connection (max 12 words) |
| `common_ground_extension_exposed` | bool | The "fellow Americans"/economic-concerns framing was said aloud in Stage 4 (sticky) |
| `closing_reflection_answered` | bool | User gave a substantive reply to the Stage-4 closing question (gates Stage 4 → COMPLETE) |

**`personal_narrative`**

| Signal | Type | Description |
|---|---|---|
| `person_label` | string | Label the user chose for the person (`"my uncle"`, `"Sarah"`) |
| `person_is_real` | bool | Whether the person is a real, specific individual (vs. imagined/typical) |
| `why_liked_respected` | bool | User gave a real, grounded reason for liking/respecting the person (not a bare trait word) |
| `person_details_count` | int | Count of distinct substantive personal details shared |
| `origins_explored` | bool | User gave a specific, grounded speculation about why the person holds their views |
| `origins_attempted` | bool | Lower bar: user made any attempt to speculate about origins, even a bare category |
| `person_traits` | string[] | Personality traits mentioned (`["stubborn", "caring"]`) |
| `person_cares_about` | string[] | Things the person cares about (`["family", "job security"]`) |
| `person_memories` | string[] | Specific memories or anecdotes |
| `person_political_origin` | string | Why the user thinks this person holds their political views |
| `typical_exception_addressed` | bool | User compared the person to the broader outparty with at least a brief reason |
| `generalization_reflected` | bool | User substantively reflected on what the comparison means for the broader outparty |
| `community_reflected` | bool | User substantively answered the Stage-4 community-and-democracy question (final gate) |

**`misperception_correction`**

| Signal | Type | Description |
|---|---|---|
| `questions_answered` | int | Number of questions where both user answered AND agent revealed the finding (0–8) |
| `question_answers` | object | `{ "q1": 2, "q2": 3, ... }` — Likert answer (1–4) per question |
| `mid_quiz_reflection_done` | bool | The mandatory mid-quiz check-in (after Q4) has occurred |
| `reflection_shared` | bool | User shared a substantive overall reaction after all 8 questions |
| `estimate_avg` | float | *(pipeline-computed once all 8 answered)* mean of the user's Likert estimates |
| `survey_avg` | float | *(pipeline-computed)* mean of the survey answers (fixed, 1.25 on the 1–4 scale) |
| `response_pattern` | string | *(pipeline-computed)* `overestimated` / `close` / `underestimated`, from `estimate_avg` vs `survey_avg`; drives the Stage-3 factual clause |

**`control`**

| Signal | Type | Description |
|---|---|---|
| `topics_shared` | string[] | Things the user has mentioned being on their mind (accumulates across turns) |
| `current_mood` | string | Overall mood/feeling conveyed most recently |
| `main_takeaway` | string | The underlying situation/cause behind what the user is going through |
| `winding_down` | bool | Current message is a short closing acknowledgment after real prior content |

**`control_politics`**

| Signal | Type | Description |
|---|---|---|
| `topics_shared` | string[] | Political topics/concerns the user has raised (accumulates across turns) |
| `current_mood` | string | Overall tone/sentiment conveyed most recently |
| `main_concern` | string | The specific political issue or structural cause the user cares about most |
| `winding_down` | bool | Current message is a short closing acknowledgment after real prior content |

---

## API Request / Response Models

Defined in [app/schema.py](../app/schema.py). All models use camelCase aliases for JSON serialization.

### `ChatCompletionRequest`

```json
{
  "studyId": "abc123",
  "model": "common-identity",
  "message": { "role": "user", "content": "Hello" },
  "stream": false,
  "quizAnswer": 3
}
```

| Field | Type | Values |
|---|---|---|
| `studyId` | string | Required |
| `model` | string | `common-identity`, `personal-narrative`, `misperception-correction`, `control`, `control-politics` |
| `message` | object | `{ role, content }` — content can be a string or OpenAI multimodal array |
| `stream` | bool | Default `false` |
| `temperature` | float \| null | Optional generation temperature |
| `maxTokens` | int \| null | Optional generation token cap |
| `quizAnswer` | int \| null | Misperception-correction only: the Likert value (1–4) the participant clicked for the current quiz question, so the score is recorded from the UI selection instead of re-extracted from text. Omitted for other conditions/typed answers. |

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
