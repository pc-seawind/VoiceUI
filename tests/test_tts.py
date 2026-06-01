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
    AliyunNlsTextToSpeech,
    ConsoleTextToSpeech,
    MimoTextToSpeech,
    OpenAISpeechTextToSpeech,
    SystemTextToSpeech,
    _aliyun_tts_audio_format,
    _extract_stream_audio,
    _mimo_audio_format,
    _openai_speech_response_format,
    _play_audio_bytes,
    _split_stream_input_text,
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

    def test_create_openai_speech_tts(self) -> None:
        tts = create_tts(TtsConfig(provider="openai_speech"))

        self.assertIsInstance(tts, OpenAISpeechTextToSpeech)

    def test_create_aliyun_nls_tts(self) -> None:
        tts = create_tts(TtsConfig(provider="aliyun_nls"))

        self.assertIsInstance(tts, AliyunNlsTextToSpeech)

    def test_mimo_tts_sends_assistant_text_and_decodes_audio(self) -> None:
        config = TtsConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="xiaomi/mimo-v2-tts",
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
        self.assertEqual(payload["model"], "xiaomi/mimo-v2-tts")
        self.assertEqual(payload["messages"][-1], {"role": "assistant", "content": "你好"})
        self.assertEqual(payload["audio"], {"format": "pcm", "voice": "mimo_default"})
        self.assertEqual(post_json.call_args.kwargs["headers"], {"api-key": "test-token"})

    def test_mimo_streaming_tts_requests_pcm16_stream_and_plays_chunks(self) -> None:
        config = TtsConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="xiaomi/mimo-v2-tts",
            audio_format="pcm",
            stream=True,
        )
        tts = MimoTextToSpeech(config)
        chunk = base64.b64encode(b"\x00\x00").decode("ascii")

        with patch.dict("os.environ", {"MIFY_API_KEY": "test-token"}):
            with patch("voiceui.tts._post_json_stream") as post_json_stream:
                with patch("voiceui.tts._play_pcm_stream") as play_pcm_stream:
                    post_json_stream.return_value = iter(
                        [{"choices": [{"delta": {"audio": {"data": chunk}}}]}]
                    )
                    play_pcm_stream.side_effect = lambda chunks, **_kwargs: sum(1 for _ in chunks)

                    with contextlib.redirect_stdout(io.StringIO()):
                        tts.speak("你好")

        url, payload = post_json_stream.call_args.args[:2]
        self.assertEqual(url, "https://api.xiaomimimo.com/v1/chat/completions")
        self.assertEqual(payload["model"], "xiaomi/mimo-v2-tts")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["audio"]["format"], "pcm16")
        self.assertEqual(post_json_stream.call_args.kwargs["headers"], {"api-key": "test-token"})
        self.assertEqual(play_pcm_stream.call_args.kwargs["sample_rate"], 24000)

    def test_mimo_streaming_tts_forces_pcm16_even_if_configured_wav(self) -> None:
        self.assertEqual(_mimo_audio_format("wav", stream=True), "pcm16")
        self.assertEqual(_mimo_audio_format("wav", stream=False), "wav")

    def test_openai_speech_streaming_tts_plays_pcm_chunks(self) -> None:
        config = TtsConfig(
            provider="openai_speech",
            endpoint="http://localhost:8000",
            model="tts-1",
            voice="default",
            audio_format="pcm16",
            stream=True,
        )
        tts = OpenAISpeechTextToSpeech(config)

        with patch("voiceui.tts._post_binary_stream") as post_binary_stream:
            with patch("voiceui.tts._play_pcm_stream") as play_pcm_stream:
                played_chunks = []
                post_binary_stream.return_value = iter([b"\x00", b"\x00\x01", b"\x00"])

                def play(chunks, **_kwargs):
                    played_chunks.extend(chunks)
                    return len(played_chunks)

                play_pcm_stream.side_effect = play

                with contextlib.redirect_stdout(io.StringIO()):
                    tts.speak("你好")

        url, payload = post_binary_stream.call_args.args[:2]
        self.assertEqual(url, "http://localhost:8000/v1/audio/speech")
        self.assertEqual(payload["model"], "tts-1")
        self.assertEqual(payload["input"], "你好")
        self.assertEqual(payload["voice"], "default")
        self.assertEqual(payload["response_format"], "pcm")
        self.assertEqual(play_pcm_stream.call_args.kwargs["sample_rate"], 24000)
        self.assertEqual(played_chunks, [b"\x00\x00", b"\x01\x00"])

    def test_openai_speech_response_format_maps_pcm16_to_pcm_for_streaming(self) -> None:
        self.assertEqual(_openai_speech_response_format("pcm16", stream=True), "pcm")
        self.assertEqual(_openai_speech_response_format("pcm16", stream=False), "pcm")
        self.assertEqual(_openai_speech_response_format("wav", stream=False), "wav")

    def test_aliyun_tts_audio_format_maps_pcm16_to_pcm(self) -> None:
        self.assertEqual(_aliyun_tts_audio_format("pcm16"), "pcm")
        self.assertEqual(_aliyun_tts_audio_format("wav"), "wav")

    def test_split_stream_input_text(self) -> None:
        self.assertEqual(
            _split_stream_input_text("你好，我在。有什么可以帮你？"),
            ["你好，我在。", "有什么可以帮你？"],
        )

    def test_aliyun_nls_synthesize_uses_env_credentials(self) -> None:
        config = TtsConfig(
            provider="aliyun_nls",
            endpoint="wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1",
            access_key_id_env="ALIYUN_AccessKeyId",
            access_key_secret_env="ALIYUN_AccessKeySecret",
            app_key_env="ALIYUN_NLS_APPKEY",
            voice="longxiaochun",
            audio_format="pcm",
            sample_rate=24000,
        )
        tts = AliyunNlsTextToSpeech(config)

        with patch.dict(
            "os.environ",
            {
                "ALIYUN_AccessKeyId": "ak",
                "ALIYUN_AccessKeySecret": "secret",
                "ALIYUN_NLS_APPKEY": "appkey",
            },
        ):
            with patch("voiceui.tts.get_aliyun_nls_token", return_value="token") as get_token:
                with patch(
                    "voiceui.tts._aliyun_stream_input_tts_chunks",
                    return_value=iter([b"\x00\x00"]),
                ) as stream_tts:
                    audio = tts.synthesize("你好")

        self.assertEqual(audio.data, b"\x00\x00")
        self.assertEqual(audio.format, "pcm")
        get_token.assert_called_once_with("ak", "secret")
        self.assertEqual(stream_tts.call_args.kwargs["config"], config)
        self.assertEqual(stream_tts.call_args.kwargs["token"], "token")
        self.assertEqual(stream_tts.call_args.kwargs["text"], "你好")

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
