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
        self.assertEqual(config.wake.debug_audio_seconds, 5.0)

    def test_example_config_loads_nested_values(self) -> None:
        config = load_config("config.example.yaml")

        self.assertIsInstance(config.stt, SttConfig)
        self.assertIsInstance(config.llm, LlmConfig)
        self.assertEqual(config.input.mode, "audio")
        self.assertEqual(config.audio.channels, 2)
        self.assertEqual(config.audio.device, 24)
        self.assertEqual(config.audio.wake_stream_channel, 1)
        self.assertEqual(config.audio.input_gain_db, 0.0)
        self.assertEqual(config.wake_ack.playback_device, 22)
        self.assertEqual(config.tts.sample_rate, 24000)
        self.assertEqual(config.tts.playback_sample_rate, 16000)
        self.assertEqual(config.tts.playback_channels, 2)
        self.assertEqual(config.tts.playback_device, 22)
        self.assertEqual(config.vad.trailing_silence_trim_ms, 500)

    def test_mify_config_loads_backend_values(self) -> None:
        config = load_config("config.mify.example.yaml")

        self.assertEqual(config.stt.provider, "mify")
        self.assertEqual(config.llm.provider, "bailian")
        self.assertEqual(config.stt.api_key_env, "MIFY_API_KEY")
        self.assertEqual(config.llm.api_key_env, "BAILIAN_API_KEY")
        self.assertEqual(config.stt.model, "xiaomi/mimo-v2.5")
        self.assertEqual(config.llm.model, "qwen3.6-flash")
        self.assertEqual(config.llm.extra_body, {"enable_thinking": False})
        self.assertEqual(config.tts.sample_rate, 24000)
        self.assertEqual(config.tts.playback_sample_rate, 16000)
        self.assertEqual(config.tts.playback_channels, 2)

    def test_demo_configs_load(self) -> None:
        mock_config = load_config("config.demo.mock.yaml")
        mify_config = load_config("config.demo.mify.yaml")
        wake_config = load_config("config.demo.wake.yaml")
        local_tts_config = load_config("config.demo.wake.local-tts.yaml")
        aliyun_asr_config = load_config("config.demo.aliyun-asr.yaml")
        aliyun_tts_config = load_config("config.demo.aliyun-tts.yaml")
        aliyun_wake_config = load_config("config.demo.wake.aliyun.yaml")

        for demo_config in (
            mock_config,
            mify_config,
            wake_config,
            local_tts_config,
            aliyun_asr_config,
            aliyun_tts_config,
            aliyun_wake_config,
        ):
            self.assertEqual(demo_config.audio.device, 24)
            self.assertEqual(demo_config.audio.channels, 2)
            self.assertEqual(demo_config.audio.wake_stream_channel, 1)
            self.assertEqual(demo_config.audio.command_stream_channel, 0)
            self.assertEqual(demo_config.vad.trailing_silence_trim_ms, 500)
            if demo_config.tts.playback_device == 22:
                self.assertEqual(demo_config.tts.sample_rate, 24000)
                self.assertEqual(demo_config.tts.playback_sample_rate, 16000)
                self.assertEqual(demo_config.tts.playback_channels, 2)

        self.assertEqual(mock_config.wake.engine, "manual")
        self.assertEqual(mock_config.tts.provider, "system")
        self.assertEqual(mock_config.audio.device, 24)
        self.assertEqual(mock_config.audio.input_gain_db, 0.0)
        self.assertEqual(mock_config.vad.engine, "silero")
        self.assertEqual(mock_config.vad.threshold, 0.6)
        self.assertEqual(mock_config.vad.pre_roll_ms, 640)
        self.assertTrue(mock_config.vad.debug)
        self.assertEqual(mify_config.wake.engine, "manual")
        self.assertEqual(mify_config.vad.engine, "silero")
        self.assertEqual(mify_config.vad.threshold, 0.6)
        self.assertEqual(mify_config.vad.pre_roll_ms, 640)
        self.assertTrue(mify_config.vad.debug)
        self.assertEqual(mify_config.audio.device, 24)
        self.assertEqual(mify_config.stt.provider, "mify")
        self.assertEqual(mify_config.stt.model, "xiaomi/mimo-v2.5")
        self.assertEqual(mify_config.llm.provider, "bailian")
        self.assertEqual(mify_config.llm.api_key_env, "BAILIAN_API_KEY")
        self.assertEqual(mify_config.llm.model, "qwen3.6-flash")
        self.assertEqual(mify_config.llm.extra_body, {"enable_thinking": False})
        self.assertTrue(mify_config.llm.stream)
        self.assertEqual(mify_config.tts.provider, "mify")
        self.assertEqual(mify_config.tts.model, "xiaomi/mimo-v2-tts")
        self.assertEqual(mify_config.tts.audio_format, "pcm16")
        self.assertTrue(mify_config.tts.stream)
        self.assertEqual(mify_config.tts.playback_device, 22)
        self.assertTrue(mify_config.debug.enabled)
        self.assertEqual(wake_config.wake.engine, "openwakeword")
        self.assertEqual(wake_config.wake.model, "alexa")
        self.assertEqual(wake_config.wake.threshold, 0.5)
        self.assertEqual(wake_config.wake.inference_framework, "onnx")
        self.assertTrue(wake_config.wake.debug)
        self.assertTrue(wake_config.wake_ack.enabled)
        self.assertEqual(wake_config.wake_ack.wav_path, "default")
        self.assertEqual(wake_config.wake_ack.playback_device, 22)
        self.assertEqual(wake_config.audio.device, 24)
        self.assertEqual(wake_config.vad.engine, "silero")
        self.assertEqual(wake_config.vad.threshold, 0.6)
        self.assertEqual(wake_config.vad.pre_roll_ms, 640)
        self.assertTrue(wake_config.vad.debug)
        self.assertTrue(wake_config.conversation.barge_in_enabled)
        self.assertEqual(wake_config.stt.model, "xiaomi/mimo-v2.5")
        self.assertEqual(wake_config.llm.provider, "bailian")
        self.assertEqual(wake_config.llm.model, "qwen3.6-flash")
        self.assertEqual(wake_config.llm.extra_body, {"enable_thinking": False})
        self.assertTrue(wake_config.llm.stream)
        self.assertEqual(wake_config.tts.model, "xiaomi/mimo-v2-tts")
        self.assertTrue(wake_config.tts.stream)
        self.assertEqual(wake_config.tts.playback_device, 22)
        self.assertEqual(local_tts_config.tts.provider, "openai_speech")
        self.assertEqual(local_tts_config.tts.endpoint, "http://127.0.0.1:8000")
        self.assertEqual(local_tts_config.tts.audio_format, "pcm")
        self.assertTrue(local_tts_config.tts.stream)
        self.assertEqual(local_tts_config.tts.playback_device, 22)
        self.assertEqual(local_tts_config.wake.model, "alexa")
        self.assertEqual(local_tts_config.wake.threshold, 0.5)
        self.assertTrue(local_tts_config.wake.debug)
        self.assertTrue(local_tts_config.conversation.barge_in_enabled)
        self.assertEqual(local_tts_config.llm.provider, "bailian")
        self.assertEqual(local_tts_config.llm.model, "qwen3.6-flash")
        self.assertTrue(local_tts_config.llm.stream)
        self.assertEqual(aliyun_asr_config.stt.provider, "aliyun_nls")
        self.assertEqual(aliyun_asr_config.stt.app_key_env, "ALIYUN_NLS_APPKEY")
        self.assertEqual(aliyun_asr_config.stt.leading_silence_ms, 200)
        self.assertTrue(aliyun_asr_config.stt.debug)
        self.assertEqual(
            aliyun_asr_config.stt.endpoint,
            "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1",
        )
        self.assertEqual(aliyun_tts_config.tts.provider, "aliyun_nls")
        self.assertEqual(aliyun_tts_config.tts.voice, "longxiaochun")
        self.assertEqual(aliyun_tts_config.audio.device, 24)
        self.assertEqual(aliyun_tts_config.tts.playback_device, 22)
        self.assertEqual(
            aliyun_tts_config.tts.endpoint,
            "wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1",
        )
        self.assertEqual(aliyun_wake_config.wake.engine, "openwakeword")
        self.assertTrue(aliyun_wake_config.audio.debug)
        self.assertEqual(aliyun_wake_config.audio.device, 24)
        self.assertEqual(aliyun_wake_config.wake.model, "alexa")
        self.assertEqual(aliyun_wake_config.wake.threshold, 0.5)
        self.assertTrue(aliyun_wake_config.wake.debug)
        self.assertTrue(aliyun_wake_config.wake_ack.enabled)
        self.assertEqual(aliyun_wake_config.wake_ack.playback_device, 22)
        self.assertEqual(aliyun_wake_config.vad.engine, "silero")
        self.assertEqual(aliyun_wake_config.vad.threshold, 0.6)
        self.assertEqual(aliyun_wake_config.vad.pre_roll_ms, 640)
        self.assertTrue(aliyun_wake_config.vad.debug)
        self.assertEqual(aliyun_wake_config.stt.provider, "aliyun_nls")
        self.assertEqual(aliyun_wake_config.stt.leading_silence_ms, 200)
        self.assertTrue(aliyun_wake_config.stt.debug)
        self.assertEqual(aliyun_wake_config.llm.provider, "bailian")
        self.assertEqual(aliyun_wake_config.llm.api_key_env, "BAILIAN_API_KEY")
        self.assertEqual(aliyun_wake_config.llm.model, "qwen3.6-flash")
        self.assertEqual(aliyun_wake_config.llm.extra_body, {"enable_thinking": False})
        self.assertTrue(aliyun_wake_config.llm.stream)
        self.assertEqual(aliyun_wake_config.tts.provider, "aliyun_nls")
        self.assertTrue(aliyun_wake_config.tts.stream)
        self.assertEqual(aliyun_wake_config.tts.playback_device, 22)
        self.assertTrue(aliyun_wake_config.conversation.barge_in_enabled)


if __name__ == "__main__":
    unittest.main()
