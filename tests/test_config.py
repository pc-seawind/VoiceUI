from __future__ import annotations

import unittest

from voiceui.config import load_config
from voiceui.models import AudioConfig, InputConfig, LlmConfig, SttConfig


class ConfigTests(unittest.TestCase):
    def test_default_config_uses_nested_dataclasses(self) -> None:
        config = load_config()

        self.assertIsInstance(config.input, InputConfig)
        self.assertIsInstance(config.audio, AudioConfig)
        self.assertEqual(config.input.mode, "text")

    def test_example_config_loads_nested_values(self) -> None:
        config = load_config("config.example.yaml")

        self.assertIsInstance(config.stt, SttConfig)
        self.assertIsInstance(config.llm, LlmConfig)
        self.assertEqual(config.input.mode, "audio")
        self.assertEqual(config.audio.channels, 2)
        self.assertEqual(config.audio.wake_stream_channel, 1)

    def test_mify_config_loads_backend_values(self) -> None:
        config = load_config("config.mify.example.yaml")

        self.assertEqual(config.stt.provider, "mify")
        self.assertEqual(config.llm.provider, "mify")
        self.assertEqual(config.stt.api_key_env, "MIFY_API_KEY")
        self.assertEqual(config.llm.api_key_env, "MIFY_API_KEY")


if __name__ == "__main__":
    unittest.main()
