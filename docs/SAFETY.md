# Safety Guarantee

This document describes the safety layer that sits in front of the research
agent in this project. It is implemented in [`app/agent/safety.py`](../app/agent/safety.py)
and wired into the pipeline at [`app/agent/pipeline.py`](../app/agent/pipeline.py).

## Goal

Guarantee that **malformed, abusive, or adversarial user input never reaches
the research model's context** and that conversations cannot be dragged into
unbounded exchanges of bad input. The layer is deterministic, auditable, and
contains **no LLM calls** — verdicts are reproducible and free from model
drift.

## Scope

The safety layer classifies every incoming user message into one of:

| Category    | Detector                        | Examples                                    |
|-------------|---------------------------------|---------------------------------------------|
| `clean`     | —                               | Normal conversational input                 |
| `gibberish` | `is_gibberish(msg)`             | `""`, `"a"`, `"asdfasdf"`, `"qwrtp"`, `"!!!!!!"` |
| `indecent`  | `contains_indecent(msg)`        | Profanity and slurs (incl. leet variants)   |
| `both`      | Matches both detectors          | Profane gibberish                           |

An exact repeat of the previous user message is also treated as gibberish
(`is_exact_repeat`) to prevent copy-paste stalling.

## Guarantees

1. **No LLM in the filter path.** Detection is pure regex + token heuristics.
   The outcome for a given input + counter state is fully deterministic.
2. **Bad input never contaminates the research model.** The safety check runs
   *before* OBSERVE/THINK/EXECUTE, so flagged messages never become part of the
   main model's prompt or the extracted `signals`.
3. **Bounded bad-input exchange.** After `CONSECUTIVE_REMINDER_LIMIT = 3`
   consecutive strikes, the conversation is terminated with a fixed message.
   One clean message resets the streak to zero.
4. **Pre-written, non-generated responses.** Reminder and termination text are
   static constants — the model cannot be prompt-injected into saying
   something off-script via a safety event.
5. **Pure verdict function.** `evaluate_message` returns a `SafetyVerdict`
   without mutating state; the caller applies it. This makes the logic easy
   to test and audit.
6. **Full audit trail.** Every verdict is logged via
   [`conversation_logger`](../app/agent/conversation_logger.py) with
   category, reason, excerpt, counters, and timestamp.

## Detection rules

### Indecent language — `contains_indecent`

A curated list of profanity and slur patterns compiled as regexes. Each
pattern:

- Tolerates leet substitution (`i→1|!`, `o→0`, `s→$5`, `a→@4`, `e→3`, `u→*`).
- Tolerates spacing/punctuation separators (`f u c k`, `f.u.c.k`, `f_u_c_k`).
- Is word-boundary anchored to avoid false positives. Substring-prone words
  (`cunt`, `dick`) use `\b` on both sides and a constrained suffix list so
  that "Scunthorpe", "Dickinson", and similar do **not** match.
- Common compound-forming words (`fuck`, `shit`) intentionally drop the
  leading `\b` so that `motherfucker`, `bullshit`, `dipshit` are caught.

### Gibberish — `is_gibberish(msg, quiz_mode=False)`

Ordered checks (first match wins):

1. **Empty / whitespace-only** → gibberish.
2. **Long character run** — 5+ of the same character in a row
   (`aaaaaa`, `!!!!!!!`, `hmmmmmmm`).
3. **Repeating substring** — a 2–4 char unit repeating to fill ≥70% of the
   message (`asdfasdf`, `lololol`, `abcabcabc`).
4. **Punctuation-only** (no letters) → **not** gibberish. Messages like `?`,
   `...`, `!?` are passed through as ambiguous reactions for OBSERVE to
   interpret.
5. **Single-letter** alphabetic messages → gibberish.
6. **Allowlist hit** (`_SHORT_VALID_REPLIES`) → clean. Covers short valid
   replies (`ok`, `yes`, `idk`), study vocabulary (`republican`,
   `democrat`), and greetings (`hi`, `hello`, `hey`, `howdy`,
   `good morning`, etc.).
7. **No vowels** in a 4+ letter run → gibberish (`qwrtp`, `zxcvbn`).
8. **Indecent** → routed to the indecent category, not gibberish (no
   double-flagging).
9. **`quiz_mode=True`** → clean (skip the soft-signal below). See
   "Quiz mode" below for rationale.
10. **Short + no recognized word** — message ≤40 chars and no token matches
    the built-in common-word set → gibberish.
11. Otherwise → clean.

#### Quiz mode

The pipeline passes `quiz_mode=True` when the active condition is
`misperception_correction` and the stage is `stage_2` (the 8-question
quiz). Likert reasoning during the quiz often contains zero hits in the
~150-word `_COMMON_WORDS` set (`"1, doesn't sound right"`,
`"feels wrong to me"`), so the soft-signal at step 10 has an unacceptably
high false-positive rate during the quiz — a single false-flag throws
out a participant's answer and offsets all subsequent q-keys. Hard
signals (steps 1–8) still fire in `quiz_mode`, so empty messages, char
runs, repeats, single letters, and indecent input are still caught.

This is condition+stage-scoped: in any other stage or condition the
soft-signal continues to apply.

### Exact repeat — `is_exact_repeat`

If the normalized current message equals the previous normalized user
message, it is treated as gibberish (stalling behavior).

## Response behavior

### Tiered reminders

`evaluate_message` picks reminder tier by the post-strike streak position:

| Strike | Gibberish reminder                                                                 | Indecent reminder                                                                        |
|--------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1      | Gentle — "I'm having a little trouble following that. Could you share your thought in a full sentence?" | Polite redirect — "This is a research conversation, so please keep the language polite and respectful. Could you rephrase that?" |
| 2      | Firmer last-warning tone, offers exit                                              | Firmer polite-language ask, offers exit                                                  |
| 3      | — (terminate)                                                                      | — (terminate)                                                                            |

When a message matches **both** categories, the indecent reminder is used
(more specific feedback than the generic gibberish prompt).

### Termination

On the strike that would push `consecutive_reminders` to
`CONSECUTIVE_REMINDER_LIMIT` (3), the verdict action is `terminate` and the
fixed `TERMINATION_MESSAGE` is returned. Subsequent attempts to send
messages receive `TERMINATION_REENTRY_MESSAGE`.

### Streak reset

A single `clean` verdict resets `consecutive_reminders` to 0. The lifetime
counters `indecent_count` and `invalid_count` are never reset — they are
retained for logging and post-hoc analysis.

## Threat model — what this does and does not cover

**Covered**

- Profanity / slurs, including leet and spacing obfuscation.
- Gibberish, keyboard mashing, punctuation spam, single repeated tokens.
- Session stalling via exact repeats.
- Unbounded bad-input exchanges (bounded by the 3-strike rule).
- Prompt injection via safety response text (responses are static).

**Not covered (by design)**

- Semantic content moderation (e.g. off-topic but well-formed messages).
  The research agent itself handles topic drift.
- Multilingual profanity outside the curated English set.
- Adversarial paraphrasing that is neither profane nor gibberish. Such
  input reaches the main model, which has its own alignment training.
- PII detection / redaction — not in scope for this filter.

## Testing

- [`tests/test_safety.py`](../tests/test_safety.py) — unit tests for
  `is_gibberish`, `contains_indecent`, `is_exact_repeat`, and
  `evaluate_message` across clean, reminder, and termination paths.
- [`tests/test_pipeline_safety.py`](../tests/test_pipeline_safety.py) —
  integration tests confirming the pipeline short-circuits on flagged
  input and that the research model is never invoked with contaminated
  context.

Run: `python -m pytest tests/test_safety.py tests/test_pipeline_safety.py`.

## Tuning knobs

| Symbol                         | Location                      | Effect                                                   |
|--------------------------------|-------------------------------|----------------------------------------------------------|
| `CONSECUTIVE_REMINDER_LIMIT`   | `safety.py`                   | Strikes before termination (default: 3)                  |
| `GIBBERISH_REMINDERS`          | `safety.py`                   | Tiered gibberish reminder text                           |
| `INDECENT_REMINDERS`           | `safety.py`                   | Tiered indecent-language reminder text                   |
| `TERMINATION_MESSAGE`          | `safety.py`                   | Final message shown on termination                       |
| `INDECENT_PATTERNS`            | `safety.py`                   | Curated profanity/slur regex list                        |
| `_SHORT_VALID_REPLIES`         | `safety.py`                   | Exact-match allowlist (short replies, greetings)         |
| `_COMMON_WORDS`                | `safety.py`                   | Recognized-word set for the "short + unrecognized" rule  |
| `quiz_mode` flag in pipeline   | `pipeline.py::_safety_check`  | Conditions/stages where the soft-signal is suppressed    |
