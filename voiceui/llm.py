from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass

from voiceui.http_utils import post_json, require_api_key
from voiceui.models import LlmConfig


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class ChatClient:
    def complete(self, messages: list[ChatMessage]) -> str:
        raise NotImplementedError

    def stream_complete(self, messages: list[ChatMessage]) -> Iterator[str]:
        yield self.complete(messages)


class MockChatClient(ChatClient):
    def complete(self, messages: list[ChatMessage]) -> str:
        last = next((message.content for message in reversed(messages) if message.role == "user"), "")
        return f"Mock response: {last}"

    def stream_complete(self, messages: list[ChatMessage]) -> Iterator[str]:
        response = self.complete(messages)
        for part in _chunk_text(response):
            yield part


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

    def stream_complete(self, messages: list[ChatMessage]) -> Iterator[str]:
        payload = {
            "model": self.config.model,
            "messages": _messages_payload(messages),
            "stream": True,
            "options": {"temperature": self.config.temperature},
        }
        for event in _post_json_stream(
            f"{self.config.endpoint.rstrip('/')}/api/chat",
            payload,
            timeout=self.config.timeout_seconds,
        ):
            message = event.get("message", {})
            content = _content_to_text(message.get("content"))
            if content:
                yield content
            if event.get("done"):
                break


class OpenAICompatibleChatClient(ChatClient):
    def __init__(self, config: LlmConfig):
        self.config = config

    def complete(self, messages: list[ChatMessage]) -> str:
        payload = _openai_chat_payload(self.config, messages, stream=False)
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_text(data)

    def stream_complete(self, messages: list[ChatMessage]) -> Iterator[str]:
        payload = _openai_chat_payload(self.config, messages, stream=True)
        yield from _stream_chat_completion_text(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.config.api_key_env:
            api_key = require_api_key(self.config.api_key_env)
            headers["Authorization"] = f"Bearer {api_key}"
        return headers


class MimoChatClient(ChatClient):
    def __init__(self, config: LlmConfig):
        self.config = config

    def complete(self, messages: list[ChatMessage]) -> str:
        payload = _openai_chat_payload(self.config, messages, stream=False)
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_text(data)

    def stream_complete(self, messages: list[ChatMessage]) -> Iterator[str]:
        payload = _openai_chat_payload(self.config, messages, stream=True)
        yield from _stream_chat_completion_text(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.config.api_key_env:
            api_key = require_api_key(self.config.api_key_env)
            headers["api-key"] = api_key
        return headers


def create_chat_client(config: LlmConfig) -> ChatClient:
    if config.provider == "mock":
        return MockChatClient()
    if config.provider == "ollama":
        return OllamaChatClient(config)
    if config.provider in ("openai_compatible", "bailian"):
        return OpenAICompatibleChatClient(config)
    if config.provider in ("mify", "mimo"):
        return MimoChatClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _messages_payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _openai_chat_payload(config: LlmConfig, messages: list[ChatMessage], stream: bool) -> dict:
    return {
        "model": config.model,
        "messages": _messages_payload(messages),
        "temperature": config.temperature,
        **config.extra_body,
        "stream": stream,
    }


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


def _stream_chat_completion_text(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> Iterator[str]:
    for event in _post_json_stream(
        url,
        payload,
        headers=headers,
        timeout=timeout,
    ):
        content = _extract_chat_delta_text(event)
        if content:
            yield content


def _extract_chat_delta_text(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") or {}
    content = _content_to_text(delta.get("content"))
    if content:
        return content
    message = choice.get("message") or {}
    return _content_to_text(message.get("content"))


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _chunk_text(text: str, chunk_chars: int = 8) -> Iterator[str]:
    for offset in range(0, len(text), chunk_chars):
        yield text[offset : offset + chunk_chars]


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict:
    return post_json(
        url,
        payload,
        headers=headers,
        timeout=timeout,
        error_prefix="LLM request failed",
    )


def _post_json_stream(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> Iterator[dict]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break
                yield json.loads(line)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"LLM streaming request failed: {url}: HTTP {exc.code}: "
            f"{error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM streaming request failed: {url}: {exc}") from exc
