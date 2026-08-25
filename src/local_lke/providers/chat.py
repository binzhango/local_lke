"""OpenAI-compatible local chat provider and deterministic test double."""

from collections.abc import Iterator
from typing import Protocol

import httpx
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from local_lke.errors import ProviderUnavailableError
from local_lke.settings import Settings


class ChatProvider(Protocol):
    def check_models(self) -> str: ...

    def check_completion(self) -> str: ...

    def generate(self, prompt: str) -> str: ...

    def stream(self, prompt: str) -> Iterator[str]: ...


class LangChainChatProvider:
    """LangChain adapter for LM Studio, llama-server, or another local endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        api_key = settings.chat_api_key or SecretStr("local")
        self._client = ChatOpenAI(
            base_url=str(settings.chat_base_url),
            model=settings.chat_model,
            api_key=api_key,
            timeout=settings.chat_timeout_seconds,
            max_retries=settings.chat_max_retries,
            temperature=0,
        )

    def check_models(self) -> str:
        url = f"{str(self._settings.chat_base_url).rstrip('/')}/models"
        headers: dict[str, str] = {}
        if self._settings.chat_api_key:
            headers["Authorization"] = (
                f"Bearer {self._settings.chat_api_key.get_secret_value()}"
            )
        try:
            response = httpx.get(url, headers=headers, timeout=5.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderUnavailableError(
                "Cannot reach the local model list. Start LM Studio's local server "
                f"and verify LKE_CHAT_BASE_URL ({url}).",
                component="chat.models",
            ) from exc

        model_ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
        if self._settings.chat_model not in model_ids:
            available = ", ".join(str(item) for item in model_ids) or "none"
            raise ProviderUnavailableError(
                f"Configured model '{self._settings.chat_model}' is not loaded. "
                f"Available model IDs: {available}. Update LKE_CHAT_MODEL or load it.",
                component="chat.models",
            )
        return f"model '{self._settings.chat_model}' is available"

    def check_completion(self) -> str:
        answer = self.generate("Reply with exactly: ok")
        if not answer.strip():
            raise ProviderUnavailableError(
                "The local model returned an empty completion.",
                component="chat.completion",
            )
        return "minimal completion succeeded"

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.invoke(prompt)
        except Exception as exc:
            raise ProviderUnavailableError(
                "Chat completion failed. Confirm the local server is running and "
                "LKE_CHAT_MODEL matches a loaded model.",
                component="chat.completion",
            ) from exc
        return _content_to_text(response.content)

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            for response in self._client.stream(prompt):
                text = _content_to_text(response.content)
                if text:
                    yield text
        except Exception as exc:
            raise ProviderUnavailableError(
                "Streaming completion failed. Check the local model server and model ID.",
                component="chat.completion",
            ) from exc


class FakeChatProvider:
    """Network-free deterministic provider used by the test suite."""

    def __init__(
        self,
        answer: str = (
            "The Atlas support team acknowledges priority-one incidents within 15 minutes."
        ),
    ) -> None:
        self.answer = answer

    def check_models(self) -> str:
        return "fake model is available"

    def check_completion(self) -> str:
        return "fake completion succeeded"

    def generate(self, prompt: str) -> str:
        del prompt
        return self.answer

    def stream(self, prompt: str) -> Iterator[str]:
        del prompt
        words = self.answer.split()
        for index, word in enumerate(words):
            yield word if index == 0 else f" {word}"


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)
