from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from voiceui.models import SttConfig, Utterance
from voiceui.stt import (
    AliyunNlsSpeechToText,
    MimoAudioUnderstandingSpeechToText,
    _ensure_pcm16_sample_rate,
    _extract_aliyun_result,
    _prepend_pcm16_silence,
)


class SttTests(unittest.TestCase):
    def test_mimo_stt_sends_input_audio_base64_payload(self) -> None:
        config = SttConfig(
            provider="mify",
            endpoint="https://api.xiaomimimo.com/v1",
            api_key_env="MIFY_API_KEY",
            model="xiaomi/mimo-v2.5",
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
        self.assertEqual(payload["model"], "xiaomi/mimo-v2.5")
        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "input_audio")
        self.assertTrue(user_content[0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))
        self.assertEqual(post_json.call_args.kwargs["headers"], {"api-key": "test-token"})

    def test_mimo_stt_falls_back_to_reasoning_content(self) -> None:
        config = SttConfig(provider="mify", endpoint="https://api.xiaomimimo.com/v1", model="xiaomi/mimo-v2.5")
        stt = MimoAudioUnderstandingSpeechToText(config)
        utterance = Utterance(pcm=b"\x00\x00" * 160, sample_rate=16000, duration_ms=10)

        with patch("voiceui.stt._post_json") as post_json:
            post_json.return_value = {
                "choices": [{"message": {"content": "", "reasoning_content": "转写文本"}}]
            }
            transcript = stt.transcribe(utterance)

        self.assertEqual(transcript, "转写文本")

    def test_aliyun_stt_uses_env_credentials_and_pcm(self) -> None:
        config = SttConfig(
            provider="aliyun_nls",
            endpoint="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1",
            access_key_id_env="ALIYUN_AccessKeyId",
            access_key_secret_env="ALIYUN_AccessKeySecret",
            app_key_env="ALIYUN_NLS_APPKEY",
            timeout_seconds=20,
        )
        stt = AliyunNlsSpeechToText(config)
        utterance = Utterance(pcm=b"\x00\x00" * 160, sample_rate=16000, duration_ms=10)

        with patch.dict(
            "os.environ",
            {
                "ALIYUN_AccessKeyId": "ak",
                "ALIYUN_AccessKeySecret": "secret",
                "ALIYUN_NLS_APPKEY": "appkey",
            },
        ):
            with patch("voiceui.stt._get_aliyun_nls_token", return_value="token") as get_token:
                with patch(
                    "voiceui.stt._run_aliyun_speech_recognizer", return_value="你好"
                ) as recognizer:
                    transcript = stt.transcribe(utterance)

        self.assertEqual(transcript, "你好")
        get_token.assert_called_once_with("ak", "secret")
        self.assertEqual(recognizer.call_args.kwargs["url"], config.endpoint)
        self.assertEqual(recognizer.call_args.kwargs["token"], "token")
        self.assertEqual(recognizer.call_args.kwargs["app_key"], "appkey")
        self.assertEqual(recognizer.call_args.kwargs["pcm"], utterance.pcm)
        self.assertEqual(recognizer.call_args.kwargs["sample_rate"], 16000)

    def test_aliyun_stt_can_prepend_leading_silence(self) -> None:
        config = SttConfig(
            provider="aliyun_nls",
            endpoint="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1",
            access_key_id_env="ALIYUN_AccessKeyId",
            access_key_secret_env="ALIYUN_AccessKeySecret",
            app_key_env="ALIYUN_NLS_APPKEY",
            timeout_seconds=20,
            leading_silence_ms=100,
        )
        stt = AliyunNlsSpeechToText(config)
        utterance = Utterance(pcm=b"\x01\x00" * 160, sample_rate=16000, duration_ms=10)

        with patch.dict(
            "os.environ",
            {
                "ALIYUN_AccessKeyId": "ak",
                "ALIYUN_AccessKeySecret": "secret",
                "ALIYUN_NLS_APPKEY": "appkey",
            },
        ):
            with patch("voiceui.stt._get_aliyun_nls_token", return_value="token"):
                with patch(
                    "voiceui.stt._run_aliyun_speech_recognizer", return_value="浣犲ソ"
                ) as recognizer:
                    stt.transcribe(utterance)

        silence_bytes = b"\x00\x00" * 1600
        self.assertEqual(recognizer.call_args.kwargs["pcm"], silence_bytes + utterance.pcm)

    def test_aliyun_streaming_stt_sends_audio_incrementally(self) -> None:
        created: list[object] = []

        class FakeRecognizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.sent: list[bytes] = []
                self.started = False
                self.stopped = False
                self.shutdown_called = False
                created.append(self)

            def start(self, **kwargs):
                self.start_kwargs = kwargs
                self.started = True
                return True

            def send_audio(self, chunk: bytes):
                self.sent.append(chunk)

            def stop(self, timeout: int):
                self.stopped = True
                self.stop_timeout = timeout
                self.kwargs["on_completed"]('{"payload":{"result":"你好"}}')

            def shutdown(self):
                self.shutdown_called = True

        fake_nls = types.SimpleNamespace(NlsSpeechRecognizer=FakeRecognizer)
        config = SttConfig(
            provider="aliyun_nls",
            endpoint="wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1",
            access_key_id_env="ALIYUN_AccessKeyId",
            access_key_secret_env="ALIYUN_AccessKeySecret",
            app_key_env="ALIYUN_NLS_APPKEY",
            timeout_seconds=20,
            leading_silence_ms=20,
        )
        stt = AliyunNlsSpeechToText(config)

        with patch.dict(sys.modules, {"nls": fake_nls}):
            with patch.dict(
                "os.environ",
                {
                    "ALIYUN_AccessKeyId": "ak",
                    "ALIYUN_AccessKeySecret": "secret",
                    "ALIYUN_NLS_APPKEY": "appkey",
                },
            ):
                with patch("voiceui.stt._get_aliyun_nls_token", return_value="token"):
                    session = stt.start_streaming(sample_rate=16000)
                    session.write(b"\x01\x00" * 320)
                    transcript = session.finish()

        self.assertEqual(transcript, "你好")
        self.assertEqual(len(created), 1)
        recognizer = created[0]
        self.assertTrue(recognizer.started)
        self.assertTrue(recognizer.stopped)
        self.assertTrue(recognizer.shutdown_called)
        self.assertEqual(recognizer.kwargs["url"], config.endpoint)
        self.assertEqual(recognizer.kwargs["token"], "token")
        self.assertEqual(recognizer.kwargs["appkey"], "appkey")
        self.assertEqual(recognizer.sent[0], b"\x00\x00" * 320)
        self.assertEqual(recognizer.sent[1], b"\x01\x00" * 320)

    def test_extract_aliyun_result(self) -> None:
        message = '{"payload":{"result":"second time时间"}}'

        self.assertEqual(_extract_aliyun_result(message), "second time时间")

    def test_ensure_pcm16_sample_rate_keeps_matching_rate(self) -> None:
        pcm = b"\x01\x00\x02\x00"

        self.assertIs(_ensure_pcm16_sample_rate(pcm, 16000, 16000), pcm)

    def test_prepend_pcm16_silence(self) -> None:
        self.assertEqual(
            _prepend_pcm16_silence(b"\x01\x00", sample_rate=16000, silence_ms=1),
            b"\x00\x00" * 16 + b"\x01\x00",
        )


if __name__ == "__main__":
    unittest.main()
