from __future__ import annotations

import unittest
from unittest.mock import patch

from voiceui.llm import (
    ChatMessage,
    MimoChatClient,
    OllamaChatClient,
    OpenAICompatibleChatClient,
    create_chat_client,
)
from voiceui.models import LlmConfig


class LlmTests(unittest.TestCase):
    def test_openai_compatible_client_serializes_slot_messages(self) -> None:
        config = LlmConfig(
            provider="openai_compatible",
            endpoint="http://openai-compatible.local",
            model="demo-model",
        )
        client = OpenAICompatibleChatClient(config)
        messages = [ChatMessage(role="user", content="hello")]

        with patch("voiceui.llm._post_json") as post_json:
            post_json.return_value = {"choices": [{"message": {"content": "hi"}}]}
            response = client.complete(messages)

        self.assertEqual(response, "hi")
        _url, payload = post_json.call_args.args[:2]
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])

    def test_mimo_client_uses_api_key_header_and_v1_endpoint(self) -> None:
        config = LlmConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="xiaomi/mimo-v2.5",
        )
        client = MimoChatClient(config)
        messages = [ChatMessage(role="user", content="hello")]

        with patch.dict("os.environ", {"MIFY_API_KEY": "test-token"}):
            with patch("voiceui.llm._post_json") as post_json:
                post_json.return_value = {"choices": [{"message": {"content": "hi"}}]}
                response = client.complete(messages)

        self.assertEqual(response, "hi")
        url, payload = post_json.call_args.args[:2]
        self.assertEqual(url, "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertEqual(payload["model"], "xiaomi/mimo-v2.5")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(post_json.call_args.kwargs["headers"], {"api-key": "test-token"})

    def test_mimo_client_streams_chat_completion_chunks(self) -> None:
        config = LlmConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="xiaomi/mimo-v2.5",
        )
        client = MimoChatClient(config)
        messages = [ChatMessage(role="user", content="hello")]

        with patch.dict("os.environ", {"MIFY_API_KEY": "test-token"}):
            with patch("voiceui.llm._post_json_stream") as post_json_stream:
                post_json_stream.return_value = iter(
                    [
                        {"choices": [{"delta": {"content": "你"}}]},
                        {"choices": [{"delta": {"content": "好"}}]},
                    ]
                )
                chunks = list(client.stream_complete(messages))

        self.assertEqual(chunks, ["你", "好"])
        url, payload = post_json_stream.call_args.args[:2]
        self.assertEqual(url, "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["model"], "xiaomi/mimo-v2.5")
        self.assertEqual(post_json_stream.call_args.kwargs["headers"], {"api-key": "test-token"})

    def test_mimo_client_requires_configured_api_key_env(self) -> None:
        config = LlmConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="xiaomi/mimo-v2.5",
        )
        client = MimoChatClient(config)

        with patch.dict("os.environ", {}, clear=True):
            with patch("voiceui.http_utils.load_dotenv", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "MIFY_API_KEY"):
                    client.complete([ChatMessage(role="user", content="hello")])

    def test_ollama_client_serializes_slot_messages(self) -> None:
        config = LlmConfig(provider="ollama", endpoint="http://ollama.local", model="demo-model")
        client = OllamaChatClient(config)
        messages = [ChatMessage(role="user", content="hello")]

        with patch("voiceui.llm._post_json") as post_json:
            post_json.return_value = {"message": {"content": "hi"}}
            response = client.complete(messages)

        self.assertEqual(response, "hi")
        _url, payload = post_json.call_args.args[:2]
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])

    def test_openai_compatible_client_streams_delta_content(self) -> None:
        config = LlmConfig(
            provider="openai_compatible",
            endpoint="http://openai-compatible.local",
            model="demo-model",
        )
        client = OpenAICompatibleChatClient(config)

        with patch("voiceui.llm._post_json_stream") as post_json_stream:
            post_json_stream.return_value = iter(
                [
                    {"choices": [{"delta": {"content": "hello"}}]},
                    {"choices": [{"delta": {"content": " world"}}]},
                ]
            )
            chunks = list(client.stream_complete([ChatMessage(role="user", content="hello")]))

        self.assertEqual(chunks, ["hello", " world"])
        _url, payload = post_json_stream.call_args.args[:2]
        self.assertTrue(payload["stream"])

    def test_openai_compatible_client_includes_extra_body(self) -> None:
        config = LlmConfig(
            provider="openai_compatible",
            endpoint="http://openai-compatible.local",
            model="demo-model",
            extra_body={"enable_thinking": False},
        )
        client = OpenAICompatibleChatClient(config)

        with patch("voiceui.llm._post_json") as post_json:
            post_json.return_value = {"choices": [{"message": {"content": "hi"}}]}
            client.complete([ChatMessage(role="user", content="hello")])

        _url, payload = post_json.call_args.args[:2]
        self.assertFalse(payload["enable_thinking"])
        self.assertFalse(payload["stream"])

    def test_openai_compatible_client_sends_tools_and_extracts_tool_calls(self) -> None:
        config = LlmConfig(
            provider="openai_compatible",
            endpoint="http://openai-compatible.local",
            model="demo-model",
        )
        client = OpenAICompatibleChatClient(config)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("voiceui.llm._post_json") as post_json:
            post_json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_time",
                                        "arguments": "{\"timezone\":\"Asia/Shanghai\"}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
            response = client.complete_with_tools([ChatMessage(role="user", content="time")], tools)

        _url, payload = post_json.call_args.args[:2]
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(response.tool_calls[0].id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "get_current_time")
        self.assertEqual(response.tool_calls[0].arguments, {"timezone": "Asia/Shanghai"})

    def test_bailian_provider_uses_openai_compatible_bearer_auth(self) -> None:
        config = LlmConfig(
            provider="bailian",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="BAILIAN_API_KEY",
            model="qwen3.6-flash",
            extra_body={"enable_thinking": False},
        )
        client = create_chat_client(config)

        with patch.dict("os.environ", {"BAILIAN_API_KEY": "test-token"}):
            with patch("voiceui.llm._post_json") as post_json:
                post_json.return_value = {"choices": [{"message": {"content": "hi"}}]}
                response = client.complete([ChatMessage(role="user", content="hello")])

        self.assertEqual(response, "hi")
        url, payload = post_json.call_args.args[:2]
        self.assertEqual(url, "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(payload["model"], "qwen3.6-flash")
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(
            post_json.call_args.kwargs["headers"],
            {"Authorization": "Bearer test-token"},
        )

    def test_ollama_client_streams_message_content(self) -> None:
        config = LlmConfig(provider="ollama", endpoint="http://ollama.local", model="demo-model")
        client = OllamaChatClient(config)

        with patch("voiceui.llm._post_json_stream") as post_json_stream:
            post_json_stream.return_value = iter(
                [
                    {"message": {"content": "hello"}, "done": False},
                    {"message": {"content": " world"}, "done": True},
                ]
            )
            chunks = list(client.stream_complete([ChatMessage(role="user", content="hello")]))

        self.assertEqual(chunks, ["hello", " world"])
        _url, payload = post_json_stream.call_args.args[:2]
        self.assertTrue(payload["stream"])

    def test_ollama_client_sends_tools_and_extracts_dict_arguments(self) -> None:
        config = LlmConfig(provider="ollama", endpoint="http://ollama.local", model="demo-model")
        client = OllamaChatClient(config)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("voiceui.llm._post_json") as post_json:
            post_json.return_value = {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": {"location": "Shanghai"},
                            },
                        }
                    ],
                }
            }
            response = client.complete_with_tools(
                [ChatMessage(role="user", content="weather")],
                tools,
            )

        _url, payload = post_json.call_args.args[:2]
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(response.tool_calls[0].id, "call_0")
        self.assertEqual(response.tool_calls[0].arguments, {"location": "Shanghai"})


if __name__ == "__main__":
    unittest.main()
