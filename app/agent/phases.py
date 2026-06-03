from __future__ import annotations

import json
import logging

from app.agent.state import Stage, SessionState
from app.llm.base import LLMProvider, Message

logger = logging.getLogger(__name__)

# Stage transition evaluation prompt
STAGE_EVAL_PROMPT = """You are a stage controller for a partisan animosity research study agent.
The agent guides users through a structured conversation in one of several conditions.

Condition: {condition}
Current stage: {current_stage}
Stage turn count (turns within this stage): {stage_turn_count}
Known signals: {signals}

Transition criteria:

For condition "common_identity":
- STAGE_1 → STAGE_2: feeling_expressed is true AND stage_turn_count >= 1
- STAGE_2 → STAGE_3: media_distortion_acknowledged is true AND stage_turn_count >= 2
- STAGE_3 → STAGE_4: common_identity_described is true AND stage_turn_count >= 1
- STAGE_4 → COMPLETE: stage_turn_count >= 1

For condition "personal_narrative":
- STAGE_1 → STAGE_2: person_name is not null (person identified) AND stage_turn_count >= 1
- STAGE_2 → STAGE_3: person_details_count >= 2 AND stage_turn_count >= 2
- STAGE_3 → STAGE_4: origins_explored is true AND stage_turn_count >= 1
- STAGE_4 → COMPLETE: stage_turn_count >= 1

For condition "misperception_correction":
- STAGE_1 → STAGE_2: stage_turn_count >= 1
- STAGE_2 → STAGE_3: questions_answered >= 8
- STAGE_3 → STAGE_4: reflection_shared is true AND stage_turn_count >= 1
- STAGE_4 → COMPLETE: stage_turn_count >= 1

For condition "control":
- STAGE_1 → STAGE_4: stage_turn_count >= 3
- STAGE_4 → COMPLETE: stage_turn_count >= 1

For condition "control_politics":
- STAGE_1 → STAGE_4: stage_turn_count >= 3
- STAGE_4 → COMPLETE: stage_turn_count >= 1

COMPLETE is terminal — never transition away from it.

Latest user message: "{user_message}"

Based on the signals and turn counts above, determine if the stage should advance.
When in doubt, stay in the current stage.

Respond with a JSON object:
{{"next_stage": "<stage_name>", "reasoning": "<brief explanation>"}}

Valid stage names: stage_1, stage_2, stage_3, stage_4, complete"""


class StageController:
    """Manages workflow stage transitions using LLM evaluation."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def evaluate_transition(
        self, state: SessionState, user_message: str
    ) -> Stage:
        """Determine if the workflow stage should advance."""
        # COMPLETE is terminal
        if state.stage == Stage.COMPLETE:
            return Stage.COMPLETE

        prompt = STAGE_EVAL_PROMPT.format(
            condition=state.strategy,
            current_stage=state.stage.value,
            stage_turn_count=state.stage_turn_count,
            signals=json.dumps(state.signals),
            user_message=user_message,
        )

        try:
            response = await self.llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=0.1,
                max_tokens=256,
            )
            result = json.loads(response.strip().strip("```json").strip("```"))
            next_stage = Stage(result["next_stage"].lower())
            if next_stage != state.stage:
                logger.info(
                    "Stage transition: %s -> %s (%s)",
                    state.stage.value,
                    next_stage.value,
                    result.get("reasoning", ""),
                )
                state.stage_turn_count = 0
            state.stage = next_stage
            return next_stage
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(
                "Stage evaluation failed, staying in %s: %s", state.stage.value, e
            )
            return state.stage
