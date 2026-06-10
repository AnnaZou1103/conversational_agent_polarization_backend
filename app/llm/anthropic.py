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

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text

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

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
