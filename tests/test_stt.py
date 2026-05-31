from __future__ import annotations

import unittest
from unittest.mock import patch

from voiceui.models import SttConfig, Utterance
from voiceui.stt import MimoAudioUnderstandingSpeechToText


class SttTests(unittest.TestCase):
    def test_mimo_stt_sends_input_audio_base64_payload(self) -> None:
        config = SttConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="mimo-v2.5",
            language="zh",
        )
        stt = MimoAudioUnderstandingSpeechToText(config)
        utterance = Utterance(pcm=(100).to_bytes(2, "little", signed=True) * 160, sample_rate=16000, duration_ms=10)

        with patch.dict("os.environ", {"MIFY_API_KEY": "test-token"}):
            with patch("voiceui.stt._post_json") as post_json:
                post_json.return_value = {"choices": [{"message": {"content": "你好"}}]}
                transcript = stt.transcribe(utterance)

        self.assertEqual(transcript, "你好")
        url, payload = post_json.call_args.args[:2]
        self.assertEqual(url, "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertEqual(payload["model"], "mimo-v2.5")
        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "input_audio")
        self.assertTrue(user_content[0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))
        self.assertEqual(post_json.call_args.kwargs["headers"], {"api-key": "test-token"})

    def test_mimo_stt_falls_back_to_reasoning_content(self) -> None:
        config = SttConfig(provider="mify", endpoint="https://api.xiaomimimo.com/v1", model="mimo-v2.5")
        stt = MimoAudioUnderstandingSpeechToText(config)
        utterance = Utterance(pcm=b"\x00\x00" * 160, sample_rate=16000, duration_ms=10)

        with patch("voiceui.stt._post_json") as post_json:
            post_json.return_value = {
                "choices": [{"message": {"content": "", "reasoning_content": "转写文本"}}]
            }
            transcript = stt.transcribe(utterance)

        self.assertEqual(transcript, "转写文本")


if __name__ == "__main__":
    unittest.main()
