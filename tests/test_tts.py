from __future__ import annotations

import unittest
from unittest.mock import patch

from voiceui.models import TtsConfig
from voiceui.tts import ConsoleTextToSpeech, MimoTextToSpeech, SystemTextToSpeech, create_tts


class TtsTests(unittest.TestCase):
    def test_create_console_tts(self) -> None:
        tts = create_tts(TtsConfig(provider="console"))

        self.assertIsInstance(tts, ConsoleTextToSpeech)

    def test_create_system_tts(self) -> None:
        tts = create_tts(TtsConfig(provider="system"))

        self.assertIsInstance(tts, SystemTextToSpeech)

    def test_create_mimo_tts(self) -> None:
        tts = create_tts(TtsConfig(provider="mify"))

        self.assertIsInstance(tts, MimoTextToSpeech)

    def test_mimo_tts_sends_assistant_text_and_decodes_audio(self) -> None:
        config = TtsConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="mimo-v2.5-tts",
            voice="mimo_default",
            audio_format="wav",
            style_prompt="自然播报",
        )
        tts = MimoTextToSpeech(config)

        with patch.dict("os.environ", {"MIFY_API_KEY": "test-token"}):
            with patch("voiceui.tts._post_json") as post_json:
                post_json.return_value = {
                    "choices": [{"message": {"audio": {"data": "UklGRg=="}}}]
                }
                audio = tts.synthesize("你好")

        self.assertEqual(audio, b"RIFF")
        url, payload = post_json.call_args.args[:2]
        self.assertEqual(url, "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertEqual(payload["model"], "mimo-v2.5-tts")
        self.assertEqual(payload["messages"][-1], {"role": "assistant", "content": "你好"})
        self.assertEqual(payload["audio"], {"format": "wav", "voice": "mimo_default"})
        self.assertEqual(post_json.call_args.kwargs["headers"], {"api-key": "test-token"})


if __name__ == "__main__":
    unittest.main()
