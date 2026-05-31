from __future__ import annotations

import base64
import contextlib
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from voiceui.models import TtsConfig
from voiceui.tts import (
    ConsoleTextToSpeech,
    MimoTextToSpeech,
    SystemTextToSpeech,
    _extract_stream_audio,
    _play_audio_bytes,
    create_tts,
    synthesize_to_wav,
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

    def test_mimo_streaming_tts_requests_pcm16_stream_and_plays_chunks(self) -> None:
        config = TtsConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="xiaomi/mimo-v2.5-tts",
            audio_format="pcm",
            stream=True,
        )
        tts = MimoTextToSpeech(config)
        chunk = base64.b64encode(b"\x00\x00").decode("ascii")

        with patch.dict("os.environ", {"MIFY_API_KEY": "test-token"}):
            with patch("voiceui.tts._post_json_stream") as post_json_stream:
                with patch("voiceui.tts._play_pcm_stream") as play_pcm_stream:
                    post_json_stream.return_value = iter(
                        [{"choices": [{"delta": {"audio": {"data": chunk, "format": "pcm16"}}}]}]
                    )
                    play_pcm_stream.side_effect = lambda chunks, **_kwargs: sum(1 for _ in chunks)

                    with contextlib.redirect_stdout(io.StringIO()):
                        tts.speak("你好")

        url, payload = post_json_stream.call_args.args[:2]
        self.assertEqual(url, "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["audio"]["format"], "pcm16")
        self.assertEqual(post_json_stream.call_args.kwargs["headers"], {"api-key": "test-token"})
        self.assertEqual(play_pcm_stream.call_args.kwargs["sample_rate"], 24000)

    def test_extract_stream_audio_supports_delta_and_message_shapes(self) -> None:
        delta_audio = {"data": "delta"}
        message_audio = {"data": "message"}

        self.assertEqual(
            _extract_stream_audio({"choices": [{"delta": {"audio": delta_audio}}]}),
            delta_audio,
        )
        self.assertEqual(
            _extract_stream_audio({"choices": [{"message": {"audio": message_audio}}]}),
            message_audio,
        )

    def test_synthesize_to_wav_writes_pcm_audio_as_wav(self) -> None:
        class FakeSynthesizer:
            def synthesize(self, text: str):
                self.text = text
                return type("Audio", (), {"data": b"\x00\x00\x00\x40", "format": "pcm"})()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "ack.wav"
            with patch("voiceui.tts.create_tts", return_value=FakeSynthesizer()):
                path = synthesize_to_wav(
                    TtsConfig(provider="mify", sample_rate=24000),
                    "我在",
                    output_path,
                )

            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getframerate(), 24000)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.readframes(wav.getnframes()), b"\x00\x00\x00\x40")

    def test_play_pcm_audio_bytes(self) -> None:
        with patch("sounddevice.play") as play, patch("sounddevice.wait") as wait:
            _play_audio_bytes(b"\x00\x00\x00\x40", audio_format="pcm", sample_rate=24000)

        self.assertEqual(play.call_args.args[1], 24000)
        wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
