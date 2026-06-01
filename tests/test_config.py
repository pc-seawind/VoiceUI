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
        self.assertEqual(config.audio.input_gain_db, 0.0)

    def test_mify_config_loads_backend_values(self) -> None:
        config = load_config("config.mify.example.yaml")

        self.assertEqual(config.stt.provider, "mify")
        self.assertEqual(config.llm.provider, "mify")
        self.assertEqual(config.stt.api_key_env, "MIFY_API_KEY")
        self.assertEqual(config.llm.api_key_env, "MIFY_API_KEY")
        self.assertEqual(config.stt.model, "xiaomi/mimo-v2.5")
        self.assertEqual(config.llm.model, "xiaomi/mimo-v2.5")

    def test_demo_configs_load(self) -> None:
        mock_config = load_config("config.demo.mock.yaml")
        mify_config = load_config("config.demo.mify.yaml")
        wake_config = load_config("config.demo.wake.yaml")
        local_tts_config = load_config("config.demo.wake.local-tts.yaml")
        aliyun_asr_config = load_config("config.demo.aliyun-asr.yaml")

        self.assertEqual(mock_config.wake.engine, "manual")
        self.assertEqual(mock_config.tts.provider, "system")
        self.assertEqual(mock_config.audio.input_gain_db, 0.0)
        self.assertEqual(mock_config.vad.engine, "silero")
        self.assertEqual(mock_config.vad.threshold, 0.6)
        self.assertEqual(mify_config.wake.engine, "manual")
        self.assertEqual(mify_config.vad.engine, "silero")
        self.assertEqual(mify_config.vad.threshold, 0.6)
        self.assertEqual(mify_config.stt.provider, "mify")
        self.assertEqual(mify_config.stt.model, "xiaomi/mimo-v2.5")
        self.assertEqual(mify_config.llm.model, "xiaomi/mimo-v2.5")
        self.assertEqual(mify_config.tts.provider, "mify")
        self.assertEqual(mify_config.tts.model, "xiaomi/mimo-v2-tts")
        self.assertEqual(mify_config.tts.audio_format, "pcm16")
        self.assertTrue(mify_config.tts.stream)
        self.assertTrue(mify_config.debug.enabled)
        self.assertEqual(wake_config.wake.engine, "openwakeword")
        self.assertEqual(wake_config.wake.model, "hey_jarvis")
        self.assertEqual(wake_config.wake.threshold, 0.35)
        self.assertEqual(wake_config.wake.inference_framework, "onnx")
        self.assertTrue(wake_config.wake_ack.enabled)
        self.assertEqual(wake_config.wake_ack.wav_path, "default")
        self.assertEqual(wake_config.vad.engine, "silero")
        self.assertEqual(wake_config.vad.threshold, 0.6)
        self.assertEqual(wake_config.stt.model, "xiaomi/mimo-v2.5")
        self.assertEqual(wake_config.tts.model, "xiaomi/mimo-v2-tts")
        self.assertTrue(wake_config.tts.stream)
        self.assertEqual(local_tts_config.tts.provider, "openai_speech")
        self.assertEqual(local_tts_config.tts.endpoint, "http://127.0.0.1:8000")
        self.assertEqual(local_tts_config.tts.audio_format, "pcm")
        self.assertTrue(local_tts_config.tts.stream)
        self.assertEqual(aliyun_asr_config.stt.provider, "aliyun_nls")
        self.assertEqual(aliyun_asr_config.stt.app_key_env, "ALIYUN_NLS_APPKEY")
        self.assertEqual(
            aliyun_asr_config.stt.endpoint,
            "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1",
        )


if __name__ == "__main__":
    unittest.main()
