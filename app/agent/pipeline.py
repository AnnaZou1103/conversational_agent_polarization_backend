from __future__ import annotations

import re
import asyncio
import json
import logging
from collections.abc import AsyncIterator

from app.agent.conversation_logger import log_safety_event, log_turn
from app.agent.phases import StageController
from app.agent.strategies import Strategy
from app.agent.prompts import OBSERVE_PROMPTS, THINK_PROMPT, build_system_prompt
from app.agent.safety import (
    SafetyVerdict,
    TERMINATION_REENTRY_MESSAGE,
    evaluate_message,
)
from app.config import settings
from app.agent.state import Stage, SessionState, build_session_state
from app.agent.strategies import get_strategy
from app.llm.base import LLMProvider, Message

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> str:
    """Pull the first balanced JSON object out of a model response.

    Tolerates markdown fences and any leading/trailing prose. Raises
    ValueError if no balanced object is found.
    """
    if not text:
        raise ValueError("empty response")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("no '{' in response")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("unbalanced JSON object")


class AgentPipeline:
    """Core agent pipeline implementing plan-execute-observe loop."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.stage_controller = StageController(llm)

    async def _observe(
        self,
        state: SessionState,
        user_message: str,
        previous_assistant_message: str = "",
    ) -> None:
        """Analyze user message and update condition-specific signals."""
        prompt_template = OBSERVE_PROMPTS[Strategy(state.strategy)]

        # Deterministically compute the question the agent is currently asking
        # about, so OBSERVE doesn't have to infer it from prose. Only meaningful
        # for the misperception_correction strategy; ignored elsewhere. We
        # derive it from the number of q-keys already captured (self-healing
        # against any drift in questions_answered), not from questions_answered.
        answered_keys = state.signals.get("question_answers", {})
        n_answered = len(answered_keys) if isinstance(answered_keys, dict) else 0
        current_qid = f"q{n_answered + 1}" if n_answered < 8 else None

        prompt = prompt_template.format(
            condition=state.strategy,
            stage=state.stage.value,
            stage_turn_count=state.stage_turn_count,
            signals=json.dumps(state.signals),
            previous_assistant_message=previous_assistant_message.replace('"', '\\"'),
            user_message=user_message,
            current_question_id=current_qid,
        )

        try:
            response = await asyncio.wait_for(
                self.llm.complete(
                    messages=[Message(role="user", content=prompt)],
                    temperature=0.1,
                    max_tokens=512,
                ),
                timeout=30.0,
            )
            extracted = json.loads(_extract_json_object(response))

            # Merge extracted signals into state. Never overwrite existing data
            # with an empty/falsy value — the model occasionally drops a field
            # it previously populated, and for lists we want to accumulate.
            for key, value in extracted.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        state.signals[key] = value
                elif isinstance(value, int):
                    state.signals[key] = max(state.signals.get(key, 0), value)
                elif isinstance(value, list):
                    existing = state.signals.get(key, [])
                    if isinstance(existing, list):
                        merged = list(existing)
                        for item in value:
                            if item not in merged:
                                merged.append(item)
                        state.signals[key] = merged
                    else:
                        state.signals[key] = value
                elif isinstance(value, dict):
                    existing = state.signals.get(key, {})
                    state.signals[key] = (
                        {**existing, **value} if isinstance(existing, dict) else value
                    )
                elif isinstance(value, str):
                    if value:
                        state.signals[key] = value
                else:
                    state.signals[key] = value

            # For misperception_correction, derive questions_answered from the
            # set of recorded answers so the stage-transition gate is robust
            # to LLM drift on the counter itself.
            if Strategy(state.strategy) == Strategy.MISPERCEPTION_CORRECTION:
                answers = state.signals.get("question_answers", {})
                if isinstance(answers, dict):
                    state.signals["questions_answered"] = len(answers)

            state.metadata["last_observation"] = extracted

        except asyncio.TimeoutError:
            logger.warning("Observation timed out after 30s; signals unchanged")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "Observation extraction failed (%s): %s", type(e).__name__, e
            )

    async def _think(self, state: SessionState, user_message: str) -> str:
        """Internal reasoning step — LLM plans its approach before responding."""
        prompt = THINK_PROMPT.format(
            condition=state.strategy,
            stage=state.stage.value,
            stage_turn_count=state.stage_turn_count,
            signals=json.dumps(state.signals),
            user_message=user_message,
        )
        try:
            reasoning = await self.llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=300,
            )
            logger.debug("Think output: %s", reasoning)
            return reasoning.strip()
        except Exception as e:
            logger.warning("Think step failed: %s", e)
            return ""

    async def _post_observe(self, state: SessionState, assistant_response: str) -> None:
        """Update state after the agent generates a response."""
        state.memory.append(
            {
                "turn": state.turn_count,
                "stage": state.stage.value,
            }
        )

    def _safety_check(self, state: SessionState, user_message: str) -> SafetyVerdict:
        """Run the rule-based safety monitor and return a verdict.

        Pure regex + heuristics — synchronous and microsecond-cheap. Does not
        mutate state; the caller is responsible for applying the verdict.
        """
        # Disable the dictionary soft-signal during the misperception quiz —
        # Likert reasoning ("doesn't sound right") frequently contains zero
        # _COMMON_WORDS hits and would otherwise be false-flagged.
        quiz_mode = (
            Strategy(state.strategy) == Strategy.MISPERCEPTION_CORRECTION
            and state.stage == Stage.STAGE_2
        )

        return evaluate_message(
            user_message=user_message,
            previous_user_message=state.previous_user_message,
            consecutive_reminders=state.consecutive_reminders,
            indecent_count=state.indecent_count,
            invalid_count=state.invalid_count,
            quiz_mode=quiz_mode,
        )

    # Sentinel yielded during blocking pipeline work to signal keep-alive
    KEEP_ALIVE = object()

    async def _drive(self, task: asyncio.Task) -> AsyncIterator[object]:
        """Await `task`, yielding KEEP_ALIVE every 2s so the caller can emit SSE
        keep-alive comments during the blocking work. The caller retrieves the
        result via `task.result()` after the generator is exhausted."""
        done = asyncio.Event()

        async def _watch():
            try:
                await task
            finally:
                done.set()

        watcher = asyncio.create_task(_watch())
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                yield self.KEEP_ALIVE
        await watcher

    async def process_turn(
        self,
        messages: list[dict],
        strategy_name: str,
        study_id: str,
    ) -> AsyncIterator[str | object]:
        """Process a single conversation turn through the pipeline.

        Yields tokens for streaming response.
        Yields AgentPipeline.KEEP_ALIVE during blocking work so callers
        can send SSE keep-alive comments to prevent connection timeouts.
        """
        # Reconstruct session state from DB
        state = build_session_state(study_id, strategy_name, messages)
        state.metadata["study_id"] = study_id

        # Short-circuit any subsequent requests to an already-terminated session.
        if state.terminated_by_safety:
            yield TERMINATION_REENTRY_MESSAGE
            return

        state.stage_turn_count += 1

        user_message = ""
        previous_assistant_message = ""
        for msg in reversed(messages):
            role = msg.get("role")
            if role == "user" and not user_message:
                user_message = msg["content"]
            elif role == "assistant" and not previous_assistant_message:
                previous_assistant_message = msg["content"]
            if user_message and previous_assistant_message:
                break

        # --- SAFETY MONITOR ---
        # Skip the gate when there is no user message yet (agent-initiated open).
        if user_message:
            verdict = self._safety_check(state, user_message)

            if verdict.action == "reminder":
                # Roll back the stage turn so a rejected turn doesn't burn a slot.
                state.stage_turn_count = max(0, state.stage_turn_count - 1)
                state.consecutive_reminders = verdict.consecutive_reminders
                state.invalid_count = verdict.invalid_count
                state.indecent_count = verdict.indecent_count
                state.safety_history.append(verdict.to_dict())
                state.previous_user_message = user_message
                log_safety_event(settings.conversations_dir, state, verdict)
                logger.info(
                    "Safety reminder for session %s: %s (consecutive=%d)",
                    state.study_id,
                    verdict.reason,
                    state.consecutive_reminders,
                )
                yield verdict.reminder_text
                return

            if verdict.action == "terminate":
                state.terminated_by_safety = True
                state.stage = Stage.COMPLETE
                state.consecutive_reminders = verdict.consecutive_reminders
                state.invalid_count = verdict.invalid_count
                state.indecent_count = verdict.indecent_count
                state.safety_history.append(verdict.to_dict())
                state.previous_user_message = user_message
                log_safety_event(settings.conversations_dir, state, verdict)
                logger.info(
                    "Safety termination for session %s: %s",
                    state.study_id,
                    verdict.reason,
                )
                yield verdict.termination_text
                return

            # Clean message — reset the consecutive-reminder streak and record this
            # message for exact-repeat detection on the next turn.
            #
            # The reset must be PERSISTED, not just held in memory. Safety
            # counters are rebuilt each (stateless) request from the stored
            # `verdict`, which is only written on reminder/terminate events. If
            # we don't persist the clean verdict here, a clean message between
            # two reminders would never actually break the streak — so
            # non-consecutive reminders would accumulate to a wrongful
            # termination, contradicting the documented "a single clean message
            # resets the streak" rule. Only write when there is a streak to
            # clear, to avoid a DB round-trip on every clean turn.
            if state.consecutive_reminders:
                log_safety_event(settings.conversations_dir, state, verdict)
            state.consecutive_reminders = 0
            state.previous_user_message = user_message

        # --- OBSERVE, THEN STAGE EVALUATION ---
        # OBSERVE must complete BEFORE the stage gate runs: the transition
        # criteria (questions_answered, feeling_expressed, person_name, …) read
        # signals that OBSERVE extracts from THIS user message. Running them
        # concurrently made the gate read previous-turn signals, delaying every
        # signal-gated transition by a full turn.
        if state.turn_count >= 1:
            observe_task = asyncio.create_task(
                self._observe(state, user_message, previous_assistant_message)
            )
            async for keep_alive in self._drive(observe_task):
                yield keep_alive

            stage_task = asyncio.create_task(
                self.stage_controller.evaluate_transition(state, user_message)
            )
            async for keep_alive in self._drive(stage_task):
                yield keep_alive
            stage = stage_task.result()
        else:
            stage = state.stage

        logger.info(
            "Session %s: condition=%s, stage=%s, stage_turn=%d, total_turn=%d",
            state.study_id,
            state.strategy,
            stage.value,
            state.stage_turn_count,
            state.turn_count,
        )

        # Get the fixed strategy/condition for this session
        strategy = get_strategy(state.strategy)

        # --- THINK (optional) ---
        think_context = ""

        if settings.enable_think and state.turn_count >= 1:
            think_task = asyncio.create_task(self._think(state, user_message))
            async for keep_alive in self._drive(think_task):
                yield keep_alive

            if think_task.done():
                think_context = think_task.result()

        # Build the system prompt
        system_prompt = build_system_prompt(stage, strategy, state)
        if think_context:
            system_prompt += (
                f"\n\n## Internal Reasoning (not visible to user):\n{think_context}"
            )

        # --- EXECUTE ---
        llm_messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages
            if msg["role"] in ("user", "assistant")
        ]

        # Anthropic requires at least one message; on agent-initiated open the
        # history is empty so we inject a silent trigger.
        if not llm_messages:
            llm_messages = [Message(role="user", content="Hello")]

        full_response = []
        async for token in self.llm.stream(
            messages=llm_messages,
            system=system_prompt,
        ):
            full_response.append(token)
            yield token

        # --- POST-OBSERVE ---
        assistant_response = "".join(full_response)
        await self._post_observe(state, assistant_response)
        log_turn(
            settings.conversations_dir,
            state,
            system_prompt,
            llm_messages,
            assistant_response,
        )
