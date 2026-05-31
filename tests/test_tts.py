from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

from voiceui.models import TtsConfig
from voiceui.tts import (
    ConsoleTextToSpeech,
    MimoTextToSpeech,
    SystemTextToSpeech,
    _play_audio_bytes,
    create_tts,
)


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
            model="xiaomi/mimo-v2.5-tts",
            voice="mimo_default",
            audio_format="pcm",
            style_prompt="自然播报",
        )
        tts = MimoTextToSpeech(config)

        with patch.dict("os.environ", {"MIFY_API_KEY": "test-token"}):
            with patch("voiceui.tts._post_json") as post_json:
                post_json.return_value = {
                    "choices": [
                        {
                            "message": {
                                "audio": {
                                    "data": base64.b64encode(b"\x00\x00").decode("ascii"),
                                    "format": "pcm",
                                }
                            }
                        }
                    ]
                }
                audio = tts.synthesize("你好")

        self.assertEqual(audio.data, b"\x00\x00")
        self.assertEqual(audio.format, "pcm")
        url, payload = post_json.call_args.args[:2]
        self.assertEqual(url, "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertEqual(payload["model"], "xiaomi/mimo-v2.5-tts")
        self.assertEqual(payload["messages"][-1], {"role": "assistant", "content": "你好"})
        self.assertEqual(payload["audio"], {"format": "pcm", "voice": "mimo_default"})
        self.assertEqual(post_json.call_args.kwargs["headers"], {"api-key": "test-token"})

    def test_play_pcm_audio_bytes(self) -> None:
        with patch("sounddevice.play") as play, patch("sounddevice.wait") as wait:
            _play_audio_bytes(b"\x00\x00\x00\x40", audio_format="pcm", sample_rate=24000)

        self.assertEqual(play.call_args.args[1], 24000)
        wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
