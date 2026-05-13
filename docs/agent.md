# Agent Architecture

## Overview

Each conversation turn passes through a fixed pipeline: **Safety → Observe + Stage Eval → Think (optional) → Execute → Post-Observe**. All pipeline steps are implemented in [app/agent/pipeline.py](../app/agent/pipeline.py). The pipeline is stateless between requests — state is reconstructed from MongoDB at the start of every turn.

```
User message
     │
     ▼
┌─────────────┐
│   SAFETY    │  Rule-based (no LLM). Blocks gibberish/profanity before any LLM call.
└──────┬──────┘
       │ clean
       ▼
┌──────────────────────────────────┐
│  OBSERVE  ║  STAGE EVALUATION    │  Run in parallel (asyncio).
│  (LLM)   ║  (LLM)               │
└──────┬───╨──────────────────────┘
       │
       ▼
┌─────────────┐
│    THINK    │  Optional (ENABLE_THINK=true). LLM plans its response.
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   EXECUTE   │  Streaming LLM call with assembled system prompt.
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ POST-OBSERVE│  Log turn to disk + MongoDB.
└─────────────┘
```

---

## Session State

`SessionState` ([app/agent/state.py](../app/agent/state.py)) is rebuilt from MongoDB at the start of every turn via `build_session_state()`. It is never held in server memory between requests.

Key fields:

| Field | Description |
|---|---|
| `study_id` | Participant identifier |
| `strategy` | Assigned condition (e.g. `common_identity`) |
| `stage` | Current workflow stage (`stage_1`–`stage_4`, `complete`) |
| `stage_turn_count` | Turns taken within the current stage (resets on transition) |
| `turn_count` | Total turns in the conversation |
| `political_party` | `republican` or `democrat`, set during intake |
| `signals` | Condition-specific extracted data (see Observe step) |
| `consecutive_reminders` | Safety streak counter (resets on clean message) |
| `terminated_by_safety` | Short-circuit flag — if true, pipeline exits immediately |

---

## Pipeline Steps

### 1. Safety Monitor

**File:** [app/agent/safety.py](../app/agent/safety.py)

Runs synchronously before any LLM call. Pure regex + heuristics — no network cost. Evaluates the incoming user message and returns a `SafetyVerdict`.

**Checks:**
- **Gibberish** — empty input, long character runs (`aaaaa`), repeating substrings (`asdfasdf`), no vowels, no recognized English word in short messages
- **Indecent language** — regex patterns with leet-substitution tolerance (`f*ck`, `sh!t`, etc.)
- **Exact repeat** — message identical to the previous user message

**Actions:**

| Action | Condition | Effect |
|---|---|---|
| `clean` | No issues | Pipeline continues |
| `reminder` | 1st or 2nd bad message | Returns pre-written reminder text; stage turn counter rolled back |
| `terminate` | 3rd consecutive bad message | Conversation ends; `terminated_by_safety = true` persisted |

Termination is permanent — subsequent requests to a terminated session return a re-entry message immediately without running the pipeline.

**Note:** The gibberish dictionary soft-signal is disabled during `misperception_correction` Stage 2 (the quiz), because short Likert reasoning ("doesn't sound right") frequently contains no words from the common-word list.

---

### 2. Observe

**File:** [app/agent/pipeline.py](../app/agent/pipeline.py) (`_observe`) · Prompts in [app/agent/prompts.py](../app/agent/prompts.py)

Runs in parallel with Stage Evaluation (asyncio). Asks the LLM to extract structured signals from the user's message given the current conversation state. Signals are condition-specific and accumulate across turns — existing values are never overwritten with empty/falsy results.

**Merge rules:**
- `bool`: only updated if new value is `true`
- `int`: takes the max of old and new
- `list`: items are appended (no duplicates)
- `dict`: shallow-merged
- `str`: only updated if non-empty

**Signals per condition:**

| Condition | Key signals extracted |
|---|---|
| `common_identity` | `feeling_expressed`, `user_feeling_text`, `media_mentioned`, `media_distortion_acknowledged`, `exhausted_majority_introduced`, `common_identity_described` |
| `personal_narrative` | `person_label`, `person_is_real`, `person_details_count`, `origins_explored`, `person_traits`, `person_cares_about`, `person_memories`, `person_political_origin` |
| `misperception_correction` | `intro_completed`, `questions_answered`, `question_answers` (dict of q1–q8 → Likert digit), `reflection_shared` |
| `control` / `control_politics` | `topics_shared`, `current_mood` |

Signals are persisted to MongoDB after each turn and reloaded at the start of the next, so they accumulate across the full conversation.

---

### 3. Stage Evaluation

**File:** [app/agent/phases.py](../app/agent/phases.py)

Runs in parallel with Observe. The `StageController` asks the LLM whether the current stage should advance, based on turn counts and the signals extracted so far.

**Stages:** `stage_1` → `stage_2` → `stage_3` → `stage_4` → `complete`

`complete` is terminal — the controller never transitions away from it.

**Transition criteria per condition:**

| Condition | Transition | Criteria |
|---|---|---|
| `common_identity` | stage_1 → stage_2 | `feeling_expressed = true` AND ≥ 2 turns |
| | stage_2 → stage_3 | `media_distortion_acknowledged = true` AND ≥ 3 turns |
| | stage_3 → stage_4 | `common_identity_described = true` AND ≥ 2 turns |
| | stage_4 → complete | ≥ 1 turn |
| `personal_narrative` | stage_1 → stage_2 | `person_name` not null AND ≥ 2 turns |
| | stage_2 → stage_3 | `person_details_count ≥ 3` AND ≥ 4 turns |
| | stage_3 → stage_4 | `origins_explored = true` AND ≥ 2 turns |
| | stage_4 → complete | ≥ 1 turn |
| `misperception_correction` | stage_1 → stage_2 | `intro_completed = true` AND ≥ 1 turn |
| | stage_2 → stage_3 | `questions_answered ≥ 8` |
| | stage_3 → stage_4 | `reflection_shared = true` AND ≥ 1 turn |
| | stage_4 → complete | ≥ 1 turn |
| `control` / `control_politics` | stage_1 → stage_4 | ≥ 8 turns |
| | stage_4 → complete | ≥ 1 turn |

When a transition occurs, `stage_turn_count` resets to 0.

---

### 4. Think (optional)

**File:** [app/agent/pipeline.py](../app/agent/pipeline.py) (`_think`)

Enabled by setting `ENABLE_THINK=true`. An extra LLM call that produces a short internal reasoning plan before the main response. The plan is appended to the system prompt as a hidden section not visible to the user. Useful for improving response quality at the cost of latency and token usage.

---

### 5. Execute

The main LLM call. The system prompt is assembled by `build_system_prompt()` ([app/agent/prompts.py](../app/agent/prompts.py)) from three parts:

1. **Condition base prompt** — the core behavioral rules for the assigned condition (never debate, never correct, keep turns short, etc.)
2. **Stage instructions** — what to do and say in the current stage, including exact opening lines for turn 1 of each stage
3. **Session context** — current stage, turn count, political party, and all accumulated signals

`[opposing party]` and `[user party]` placeholders in the prompt are substituted with the participant's actual party affiliation before the call.

The LLM streams its response token by token. During the parallel Observe/Stage steps, the pipeline yields `KEEP_ALIVE` sentinels every 2 seconds so the SSE connection does not time out.

---

### 6. Post-Observe

After the response is fully generated, the pipeline:
1. Appends a memory entry (turn number + stage) to `state.memory`
2. Calls `log_turn()` — writes the full turn (system prompt, messages, assistant response, signals, stage) to a JSONL file in `conversations/` and saves the payload to MongoDB

---

## Condition Prompts Summary

Each condition has a **base prompt** (always-on behavioral rules) and **stage prompts** (what to do at each stage). All prompts are defined in [app/agent/prompts.py](../app/agent/prompts.py).

| Condition | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| `common_identity` | Surface feelings about opposing party | Explore media's role in shaping those feelings | Surface the exhausted majority | Reflection — what can the user do differently? |
| `personal_narrative` | Identify a real person from opposing party | Build out the person in detail (core stage) | Explore origins of their political views | Generalization — is this person typical? |
| `misperception_correction` | Deliver quiz intro, confirm user is ready | 8-question quiz with survey reveals | Reflection on quiz results | Closing question on democratic norms |
| `control` | Mental health check-in | Continue open conversation | Wind down | Close |
| `control_politics` | Open political check-in | Continue open conversation | Wind down | Close |

---

## LLM Providers

**File:** [app/llm/](../app/llm/)

The pipeline calls LLM via a `LLMProvider` interface ([app/llm/base.py](../app/llm/base.py)) with two methods: `complete()` (single response) and `stream()` (token stream). Implementations:

- `OpenAIProvider` — OpenAI and Azure OpenAI (`LLM_PROVIDER=openai` or `azure`)
- `AnthropicProvider` — Anthropic Claude (`LLM_PROVIDER=anthropic`)

Provider is selected at startup via `get_provider()` ([app/llm/registry.py](../app/llm/registry.py)) and shared as a singleton across all requests.

Observe, Stage Evaluation, and Think use `complete()` at low temperature (0.1–0.3). Execute uses `stream()` at default temperature.
