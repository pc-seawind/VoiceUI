from __future__ import annotations

import unittest
from unittest.mock import patch

from voiceui.llm import ChatMessage, OllamaChatClient, OpenAICompatibleChatClient
from voiceui.models import LlmConfig


class LlmTests(unittest.TestCase):
    def test_openai_compatible_client_serializes_slot_messages(self) -> None:
        config = LlmConfig(provider="mify", endpoint="http://mify.local", model="demo-model")
        client = OpenAICompatibleChatClient(config)
        messages = [ChatMessage(role="user", content="hello")]

        with patch("voiceui.llm._post_json") as post_json:
            post_json.return_value = {"choices": [{"message": {"content": "hi"}}]}
            response = client.complete(messages)

        self.assertEqual(response, "hi")
        _url, payload = post_json.call_args.args[:2]
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])

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


if __name__ == "__main__":
    unittest.main()
