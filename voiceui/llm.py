from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field

from voiceui.http_utils import post_json, require_api_key
from voiceui.models import LlmConfig


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw: dict = field(default_factory=dict)


@dataclass(slots=True)
class ToolChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatClient:
    def warm_up(self) -> bool:
        return False

    def complete(self, messages: list[ChatMessage]) -> str:
        raise NotImplementedError

    def stream_complete(self, messages: list[ChatMessage]) -> Iterator[str]:
        yield self.complete(messages)

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        return ToolChatResponse(content=self.complete(messages))


class MockChatClient(ChatClient):
    def complete(self, messages: list[ChatMessage]) -> str:
        last = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return f"Mock response: {last}"

    def stream_complete(self, messages: list[ChatMessage]) -> Iterator[str]:
        response = self.complete(messages)
        yield from _chunk_text(response)


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

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        payload = {
            "model": self.config.model,
            "messages": _messages_payload(messages),
            "stream": False,
            "options": {"temperature": self.config.temperature},
            "tools": tools,
        }
        data = _post_json(
            f"{self.config.endpoint.rstrip('/')}/api/chat",
            payload,
            timeout=self.config.timeout_seconds,
        )
        message = data.get("message", {})
        return _extract_message_response(message)

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

    def warm_up(self) -> bool:
        if self.config.api_key_env:
            require_api_key(self.config.api_key_env)
            return True
        return False

    def complete(self, messages: list[ChatMessage]) -> str:
        payload = _openai_chat_payload(self.config, messages, stream=False)
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_text(data)

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        payload = _openai_chat_payload(self.config, messages, stream=False)
        payload["tools"] = tools
        payload.setdefault("tool_choice", "auto")
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_response(data)

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

    def warm_up(self) -> bool:
        if self.config.api_key_env:
            require_api_key(self.config.api_key_env)
            return True
        return False

    def complete(self, messages: list[ChatMessage]) -> str:
        payload = _openai_chat_payload(self.config, messages, stream=False)
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_text(data)

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        payload = _openai_chat_payload(self.config, messages, stream=False)
        payload["tools"] = tools
        payload.setdefault("tool_choice", "auto")
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_response(data)

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


def _messages_payload(messages: list[ChatMessage]) -> list[dict]:
    payload: list[dict] = []
    for message in messages:
        item = {"role": message.role, "content": message.content}
        if message.tool_calls is not None:
            item["tool_calls"] = message.tool_calls
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        payload.append(item)
    return payload


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
    return _extract_chat_message_response(data).content


def _extract_chat_message_response(data: dict) -> ToolChatResponse:
    choices = data.get("choices", [])
    if not choices:
        return ToolChatResponse()
    message = choices[0].get("message", {})
    return _extract_message_response(message)


def _extract_message_response(message: dict) -> ToolChatResponse:
    content = _content_to_text(message.get("content")).strip()
    if content:
        return ToolChatResponse(content=content, tool_calls=_extract_tool_calls(message))
    reasoning_content = _content_to_text(message.get("reasoning_content")).strip()
    return ToolChatResponse(
        content=reasoning_content,
        tool_calls=_extract_tool_calls(message),
    )


def _extract_tool_calls(message: dict) -> list[ToolCall]:
    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        return []

    tool_calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = _parse_tool_arguments(function.get("arguments"))
        raw = dict(raw_call)
        call_id = str(raw.get("id") or f"call_{index}")
        raw["id"] = call_id
        raw.setdefault("type", "function")
        raw["function"] = dict(function)
        if isinstance(function.get("arguments"), str):
            raw["function"]["arguments"] = function["arguments"]
        else:
            raw["function"]["arguments"] = arguments
        tool_calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                raw=raw,
            )
        )
    return tool_calls


def _parse_tool_arguments(raw_arguments: object) -> dict:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        return {}

    parsed: object = raw_arguments
    for _ in range(5):
        if not isinstance(parsed, str):
            break
        if not parsed.strip():
            return {}
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


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
