from collections.abc import AsyncIterator

import anthropic

from app.llm.base import LLMProvider, Message


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=5)

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

    @staticmethod
    def _is_temperature_deprecated_error(e: anthropic.BadRequestError) -> bool:
        return "temperature" in str(e) and "deprecated" in str(e)

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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = self._system_param(system)

        try:
            response = await self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            # Some newer models (e.g. claude-sonnet-5) reject any non-default
            # temperature outright. Retry without it rather than hardcoding a
            # model allowlist.
            if not self._is_temperature_deprecated_error(e):
                raise
            kwargs.pop("temperature")
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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = self._system_param(system)

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.BadRequestError as e:
            if not self._is_temperature_deprecated_error(e):
                raise
            kwargs.pop("temperature")
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
