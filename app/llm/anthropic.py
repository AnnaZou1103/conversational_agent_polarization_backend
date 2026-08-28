from collections.abc import AsyncIterator

import anthropic

from app.llm.base import LLMProvider, Message

# Model families that still accept `temperature`. Anthropic retires sampling
# parameters model by model — sonnet-5 and the rest of the 4.6+ generation
# reject them with a 400 — so this is an allowlist, not a denylist: an unknown
# or newly configured model gets default sampling rather than a hard failure.
_SAMPLING_MODEL_PREFIXES = (
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-3",
)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=5)
        self._send_temperature = model.startswith(_SAMPLING_MODEL_PREFIXES)

    def _sampling_kwargs(self, temperature: float) -> dict:
        """Request kwargs carrying `temperature`, or empty if unsupported.

        anthropic SDK 1.x dropped `temperature` from the `messages.create` /
        `messages.stream` signatures (passing it raises TypeError), so on the
        models that do still honor it we have to smuggle it through the
        `extra_body` escape hatch, which merges straight into the request JSON.
        """
        if not self._send_temperature:
            return {}
        return {"extra_body": {"temperature": temperature}}

    def _disable_temperature(self, e: anthropic.BadRequestError) -> bool:
        """Latch temperature off if the API says this model has retired it.

        Safety net for the allowlist going stale: Anthropic retires sampling
        params on existing models over time, and without this the OBSERVE step
        would start failing every turn (and silently degrading) instead of once.
        """
        if not (self._send_temperature and "temperature" in str(e) and "deprecated" in str(e)):
            return False
        self._send_temperature = False
        return True

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @staticmethod
    def _system_param(system: str) -> list[dict]:
        """Wrap the system prompt in a cacheable content block.

        The system prompt is large (study instructions) and stable across the
        turns of a session, so marking it `ephemeral` lets Anthropic serve it
        from its prompt cache on subsequent turns. That cuts time-to-first-token
        and input cost on every turn after the first within the cache TTL
        (~5 min). Cache misses are free, so this is safe to always apply.
        """
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": self._build_messages(messages),
            "max_tokens": max_tokens,
            **self._sampling_kwargs(temperature),
        }
        if system:
            kwargs["system"] = self._system_param(system)

        try:
            response = await self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            if not self._disable_temperature(e):
                raise
            kwargs.pop("extra_body")
            response = await self.client.messages.create(**kwargs)

        # The model can spontaneously prepend a ThinkingBlock even without
        # extended thinking configured, so content[0] isn't reliably text.
        return next(b.text for b in response.content if b.type == "text")

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        kwargs = {
            "model": self.model,
            "messages": self._build_messages(messages),
            "max_tokens": max_tokens,
            **self._sampling_kwargs(temperature),
        }
        if system:
            kwargs["system"] = self._system_param(system)

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.BadRequestError as e:
            if not self._disable_temperature(e):
                raise
            kwargs.pop("extra_body")
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
