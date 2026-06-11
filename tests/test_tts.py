from __future__ import annotations

import base64
import contextlib
import io
import sys
import tempfile
import threading
import time
import types
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
    TextToSpeech,
    _aliyun_stream_input_tts_chunks_from_text_chunks,
    _aliyun_tts_audio_format,
    _extract_stream_audio,
    _iter_stream_input_text,
    _mimo_audio_format,
    _openai_speech_response_format,
    _play_audio_bytes,
    _play_pcm_stream,
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

    def test_iter_stream_input_text_flushes_soft_breaks_and_final_text(self) -> None:
        self.assertEqual(
            list(
                _iter_stream_input_text(
                    iter(["前面十二个字左右，", "后面继续。"]),
                    max_chars=32,
                    min_chars=6,
                )
            ),
            ["前面十二个字左右，", "后面继续。"],
        )

    def test_iter_stream_input_text_flushes_after_max_wait(self) -> None:
        def slow_chunks():
            yield "abcd"
            time.sleep(0.02)
            yield "e"

        self.assertEqual(
            list(
                _iter_stream_input_text(
                    slow_chunks(),
                    max_chars=100,
                    min_chars=4,
                    max_wait_ms=5,
                )
            ),
            ["abcde"],
        )

    def test_default_text_stream_tts_speaks_segments_and_returns_full_text(self) -> None:
        class SegmentTts(TextToSpeech):
            def __init__(self):
                self.segments: list[str] = []

            def speak(self, text: str, stop_event=None) -> None:
                self.segments.append(text)

        tts = SegmentTts()

        text = tts.speak_text_stream(iter(["第一句。", "第二句。"]))

        self.assertEqual(text, "第一句。第二句。")
        self.assertEqual(tts.segments, ["第一句。", "第二句。"])

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

    def test_aliyun_text_stream_tts_consumes_llm_chunks_and_returns_full_text(self) -> None:
        config = TtsConfig(
            provider="aliyun_nls",
            endpoint="wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1",
            voice="longxiaochun",
            audio_format="pcm",
            sample_rate=24000,
            stream=True,
        )
        tts = AliyunNlsTextToSpeech(config)
        tts._token = "token"
        consumed_text: list[str] = []

        def stream_tts(**kwargs):
            consumed_text.extend(list(kwargs["text_chunks"]))
            return iter([b"\x00\x00"])

        with patch("voiceui.tts._aliyun_stream_input_tts_chunks_from_text_chunks") as stream:
            with patch("voiceui.tts._play_pcm_stream") as play_pcm_stream:
                stream.side_effect = stream_tts
                play_pcm_stream.side_effect = lambda chunks, **_kwargs: sum(1 for _ in chunks)

                with contextlib.redirect_stdout(io.StringIO()):
                    text = tts.speak_text_stream(iter(["你好，", "我在。"]))

        self.assertEqual(text, "你好，我在。")
        self.assertEqual(consumed_text, ["你好，", "我在。"])
        self.assertEqual(stream.call_args.kwargs["config"], config)
        self.assertEqual(stream.call_args.kwargs["token"], "token")
        self.assertEqual(play_pcm_stream.call_args.kwargs["sample_rate"], 24000)

    def test_aliyun_stream_input_tts_waits_for_first_text_before_start(self) -> None:
        events: list[str] = []
        first_text_ready = threading.Event()
        received_chunks: list[bytes] = []
        errors: list[Exception] = []

        class FakeSynthesizer:
            def __init__(self, **kwargs):
                self.on_data = kwargs["on_data"]
                events.append("created")

            def startStreamInputTts(self, **_kwargs) -> None:
                events.append("start")

            def sendStreamInputTts(self, text: str) -> None:
                events.append(f"send:{text}")
                self.on_data(b"\x00\x00")

            def stopStreamInputTts(self) -> None:
                events.append("stop")

            def shutdown(self) -> None:
                events.append("shutdown")

        def delayed_text_chunks():
            events.append("generator_started")
            first_text_ready.wait(timeout=1.0)
            events.append("text_yielded")
            yield "hello?"

        def consume() -> None:
            try:
                received_chunks.extend(
                    _aliyun_stream_input_tts_chunks_from_text_chunks(
                        config=TtsConfig(
                            provider="aliyun_nls",
                            endpoint="wss://nls-gateway.example/ws/v1",
                            app_key_env="ALIYUN_NLS_APPKEY",
                            timeout_seconds=1,
                        ),
                        token="token",
                        text_chunks=delayed_text_chunks(),
                    )
                )
            except Exception as exc:
                errors.append(exc)

        fake_nls = types.SimpleNamespace(NlsStreamInputTtsSynthesizer=FakeSynthesizer)
        with patch.dict(sys.modules, {"nls": fake_nls}):
            with patch.dict("os.environ", {"ALIYUN_NLS_APPKEY": "appkey"}):
                thread = threading.Thread(target=consume)
                thread.start()
                time.sleep(0.05)
                self.assertIn("generator_started", events)
                self.assertNotIn("start", events)

                first_text_ready.set()
                thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(received_chunks, [b"\x00\x00"])
        self.assertLess(events.index("text_yielded"), events.index("start"))
        self.assertEqual(
            [event for event in events if event in ("start", "send:hello?", "stop", "shutdown")],
            ["start", "send:hello?", "stop", "shutdown"],
        )

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
        with patch("voiceui.tts._play_pcm_stream") as play_pcm_stream:
            _play_audio_bytes(b"\x00\x00\x00\x40", audio_format="pcm", sample_rate=24000)

        self.assertEqual(play_pcm_stream.call_args.kwargs["sample_rate"], 24000)

    def test_play_pcm_stream_resamples_and_outputs_stereo(self) -> None:
        written: list[bytes] = []
        stream_settings: dict[str, object] = {}

        class FakeStream:
            def __init__(self, **kwargs):
                stream_settings.update(kwargs)

            def start(self):
                return None

            def write(self, data: bytes):
                written.append(data)

            def stop(self):
                return None

            def close(self):
                return None

        def check_output_settings(**kwargs):
            if kwargs["samplerate"] != 16000 or kwargs["channels"] != 2:
                raise ValueError("unsupported")

        with patch("sounddevice.RawOutputStream", FakeStream):
            with patch("sounddevice.check_output_settings", side_effect=check_output_settings):
                with patch(
                    "sounddevice.query_devices",
                    return_value={"default_samplerate": 16000, "max_output_channels": 2},
                ):
                    chunks = iter([b"\x00\x00" * 240])
                    with contextlib.redirect_stdout(io.StringIO()):
                        count = _play_pcm_stream(
                            chunks,
                            sample_rate=24000,
                            source_channels=1,
                            device=22,
                            playback_sample_rate=16000,
                            playback_channels=2,
                        )

        self.assertEqual(count, 1)
        self.assertEqual(stream_settings["samplerate"], 16000)
        self.assertEqual(stream_settings["channels"], 2)
        self.assertEqual(stream_settings["device"], 22)
        self.assertEqual(len(written), 1)
        self.assertEqual(len(written[0]), 160 * 2 * 2)

    def test_play_pcm_stream_limiter_scales_pcm16_before_write(self) -> None:
        written: list[bytes] = []

        class FakeStream:
            def __init__(self, **kwargs):
                pass

            def start(self):
                return None

            def write(self, data: bytes):
                written.append(data)

            def stop(self):
                return None

            def close(self):
                return None

        samples = [32767, -32768, 0]
        pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)

        with patch("sounddevice.RawOutputStream", FakeStream):
            with patch("sounddevice.check_output_settings", return_value=None):
                with patch(
                    "sounddevice.query_devices",
                    return_value={"default_samplerate": 24000, "max_output_channels": 1},
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        count = _play_pcm_stream(
                            iter([pcm]),
                            sample_rate=24000,
                            source_channels=1,
                            limiter_enabled=True,
                            limiter_threshold=0.5,
                        )

        limited = [
            int.from_bytes(written[0][index : index + 2], "little", signed=True)
            for index in range(0, len(written[0]), 2)
        ]
        self.assertEqual(count, 1)
        self.assertLessEqual(max(abs(sample) for sample in limited), 16384)


    def test_aliyun_non_stream_synthesize_uses_plain_synthesizer(self) -> None:
        created: list[object] = []

        class FakePlainSynthesizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.start_kwargs = {}
                self.closed = False
                created.append(self)

            def start(self, **kwargs):
                self.start_kwargs = kwargs
                self.kwargs["on_data"](b"pcm")
                return True

        setattr(FakePlainSynthesizer, "shut" + "down", lambda self: setattr(self, "closed", True))

        fake_nls = types.SimpleNamespace(NlsSpeechSynthesizer=FakePlainSynthesizer)
        config = TtsConfig(
            provider="aliyun_nls",
            endpoint="wss://nls-gateway.example/ws/v1",
            app_key_env="ALIYUN_NLS_APPKEY",
            voice="longxiaochun",
            audio_format="pcm",
            sample_rate=24000,
            stream=False,
            volume=100,
        )
        tts = AliyunNlsTextToSpeech(config)
        tts._token = "token"

        with patch.dict(sys.modules, {"nls": fake_nls}):
            with patch.dict("os.environ", {"ALIYUN_NLS_APPKEY": "appkey"}):
                audio = tts.synthesize("你好")

        self.assertEqual(audio.data, b"pcm")
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)
        self.assertEqual(created[0].start_kwargs["text"], "你好")
        self.assertEqual(created[0].start_kwargs["volume"], 100)

    def test_play_pcm_stream_applies_positive_gain_before_limiter(self) -> None:
        written: list[bytes] = []

        class FakeStream:
            def __init__(self, **kwargs):
                pass

            def start(self):
                return None

            def write(self, data: bytes):
                written.append(data)

            def stop(self):
                return None

            def close(self):
                return None

        pcm = (1000).to_bytes(2, "little", signed=True)

        with patch("sounddevice.RawOutputStream", FakeStream):
            with patch("sounddevice.check_output_settings", return_value=None):
                with patch(
                    "sounddevice.query_devices",
                    return_value={"default_samplerate": 24000, "max_output_channels": 1},
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        _play_pcm_stream(
                            iter([pcm]),
                            sample_rate=24000,
                            source_channels=1,
                            playback_gain_db=6.0,
                            limiter_enabled=False,
                        )

        amplified = int.from_bytes(written[0], "little", signed=True)
        self.assertGreater(amplified, 1900)

if __name__ == "__main__":
    unittest.main()
