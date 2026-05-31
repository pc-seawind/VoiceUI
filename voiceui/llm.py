from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from voiceui.models import LlmConfig


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class ChatClient:
    def complete(self, messages: list[ChatMessage]) -> str:
        raise NotImplementedError


class MockChatClient(ChatClient):
    def complete(self, messages: list[ChatMessage]) -> str:
        last = next((message.content for message in reversed(messages) if message.role == "user"), "")
        return f"Mock response: {last}"


class OllamaChatClient(ChatClient):
    def __init__(self, config: LlmConfig):
        self.config = config

    def complete(self, messages: list[ChatMessage]) -> str:
        payload = {
            "model": self.config.model,
            "messages": _messages_payload(messages),
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        data = _post_json(
            f"{self.config.endpoint.rstrip('/')}/api/chat",
            payload,
            timeout=self.config.timeout_seconds,
        )
        message = data.get("message", {})
        return str(message.get("content", "")).strip()


class OpenAICompatibleChatClient(ChatClient):
    def __init__(self, config: LlmConfig):
        self.config = config

    def complete(self, messages: list[ChatMessage]) -> str:
        headers = {}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.config.model,
            "messages": _messages_payload(messages),
            "temperature": self.config.temperature,
            "stream": False,
        }
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_text(data)


class MimoChatClient(ChatClient):
    def __init__(self, config: LlmConfig):
        self.config = config

    def complete(self, messages: list[ChatMessage]) -> str:
        headers = {}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if api_key:
                headers["api-key"] = api_key

        payload = {
            "model": self.config.model,
            "messages": _messages_payload(messages),
            "temperature": self.config.temperature,
            "stream": False,
        }
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_text(data)


def create_chat_client(config: LlmConfig) -> ChatClient:
    if config.provider == "mock":
        return MockChatClient()
    if config.provider == "ollama":
        return OllamaChatClient(config)
    if config.provider == "openai_compatible":
        return OpenAICompatibleChatClient(config)
    if config.provider in ("mify", "mimo"):
        return MimoChatClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _messages_payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _chat_completions_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_chat_message_text(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = str(message.get("content") or "").strip()
    if content:
        return content
    return str(message.get("reasoning_content") or "").strip()


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {url}: {exc}") from exc
