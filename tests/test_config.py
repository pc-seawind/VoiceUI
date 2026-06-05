from __future__ import annotations

import unittest

from voiceui.config import load_config
from voiceui.models import (
    AudioConfig,
    CronConfig,
    CronJobConfig,
    InputConfig,
    LlmConfig,
    LoggingConfig,
    MusicConfig,
    SearchConfig,
    SttConfig,
    XiaomiMiotConfig,
)

XVF_INPUT_DEVICE = (
    "回音消除话筒 (reSpeaker XVF3800 4-Mic Array), Windows WASAPI (2 in, 0 out)"
)
XVF_OUTPUT_DEVICE = (
    "回音消除话筒 (reSpeaker XVF3800 4-Mic Array), Windows WASAPI (0 in, 2 out)"
)
XVF2_INPUT_DEVICE = (
    "回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array), Windows WASAPI (2 in, 0 out)"
)
XVF2_OUTPUT_DEVICE = (
    "回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array), Windows WASAPI (0 in, 2 out)"
)
XVF_INPUT_DEVICES = {XVF_INPUT_DEVICE, XVF2_INPUT_DEVICE}
XVF_OUTPUT_DEVICES = {XVF_OUTPUT_DEVICE, XVF2_OUTPUT_DEVICE}


class ConfigTests(unittest.TestCase):
    def test_default_config_uses_nested_dataclasses(self) -> None:
        config = load_config()

        self.assertIsInstance(config.input, InputConfig)
        self.assertIsInstance(config.audio, AudioConfig)
        self.assertEqual(config.input.mode, "text")
        self.assertEqual(config.wake.debug_audio_seconds, 5.0)
        self.assertIsInstance(config.cron, CronConfig)
        self.assertFalse(config.cron.enabled)
        self.assertIsInstance(config.logging, LoggingConfig)
        self.assertTrue(config.logging.enabled)

    def test_example_config_loads_nested_values(self) -> None:
        config = load_config("config.example.yaml")

        self.assertIsInstance(config.stt, SttConfig)
        self.assertIsInstance(config.llm, LlmConfig)
        self.assertEqual(config.input.mode, "audio")
        self.assertEqual(config.audio.channels, 2)
        self.assertEqual(config.audio.device, XVF_INPUT_DEVICE)
        self.assertEqual(config.audio.wake_stream_channel, 0)
        self.assertEqual(config.audio.input_gain_db, 0.0)
        self.assertEqual(config.wake_ack.playback_device, XVF_OUTPUT_DEVICE)
        self.assertEqual(config.tts.sample_rate, 24000)
        self.assertEqual(config.tts.playback_sample_rate, 16000)
        self.assertEqual(config.tts.playback_channels, 2)
        self.assertEqual(config.tts.playback_device, XVF_OUTPUT_DEVICE)
        self.assertEqual(config.vad.trailing_silence_trim_ms, 500)
        self.assertFalse(config.tools.enabled)
        self.assertTrue(config.tools.allow_time)
        self.assertTrue(config.tools.allow_weather)
        self.assertEqual(config.tools.default_weather_location, "")
        self.assertFalse(config.tools.allow_volume)
        self.assertFalse(config.tools.allow_music)
        self.assertFalse(config.tools.allow_miot)
        self.assertFalse(config.tools.allow_search)
        self.assertIsInstance(config.cron, CronConfig)
        self.assertFalse(config.cron.enabled)
        self.assertEqual(len(config.cron.jobs), 1)
        self.assertIsInstance(config.cron.jobs[0], CronJobConfig)
        self.assertEqual(config.cron.jobs[0].id, "morning_weather")
        self.assertEqual(config.cron.jobs[0].schedule, "0 7 * * *")
        self.assertFalse(config.cron.jobs[0].enabled)
        self.assertIsInstance(config.music, MusicConfig)
        self.assertEqual(config.music.provider, "disabled")
        self.assertEqual(config.music.server, "netease")
        self.assertIsInstance(config.search, SearchConfig)
        self.assertEqual(config.search.provider, "auto")
        self.assertEqual(config.search.tavily_api_key_env, "TAVILY_API_KEY")
        self.assertTrue(config.search.baidu_ai_enabled)
        self.assertEqual(config.search.baidu_ai_api_key_env, "QIANFAN_API_KEY")
        self.assertEqual(
            config.search.baidu_ai_endpoint,
            "https://qianfan.baidubce.com/v2/ai_search/chat/completions",
        )
        self.assertEqual(config.search.baidu_ai_model, "deepseek-v3")
        self.assertEqual(config.search.baidu_ai_search_mode, "required")
        self.assertFalse(config.search.baidu_ai_deep_search)
        self.assertTrue(config.search.baidu_ai_fallback_to_html)
        self.assertEqual(config.search.baidu_endpoint, "https://www.baidu.com/baidu")
        self.assertIsInstance(config.xiaomi_miot, XiaomiMiotConfig)
        self.assertFalse(config.xiaomi_miot.enabled)
        self.assertEqual(config.xiaomi_miot.cloud_server, "cn")
        self.assertTrue(config.xiaomi_miot.control_verify)
        self.assertEqual(config.xiaomi_miot.control_verify_delay_seconds, 0.8)
        self.assertIsInstance(config.logging, LoggingConfig)
        self.assertFalse(config.logging.events["audio.stream_opened"])
        self.assertFalse(config.logging.continuous["wake.score"])
        self.assertTrue(config.debug.system_input_dump_enabled)
        self.assertEqual(config.debug.system_input_dump_segment_seconds, 30)
        self.assertTrue(config.debug.voice_path_dump_enabled)

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
            self.assertIn(demo_config.audio.device, XVF_INPUT_DEVICES)
            self.assertEqual(demo_config.audio.channels, 2)
            self.assertEqual(demo_config.audio.wake_stream_channel, 0)
            self.assertEqual(demo_config.audio.command_stream_channel, 0)
            self.assertEqual(demo_config.vad.trailing_silence_trim_ms, 500)
            if demo_config.tts.playback_device in XVF_OUTPUT_DEVICES:
                self.assertEqual(demo_config.tts.sample_rate, 24000)
                self.assertEqual(demo_config.tts.playback_sample_rate, 16000)
                self.assertEqual(demo_config.tts.playback_channels, 2)

        self.assertEqual(mock_config.wake.engine, "manual")
        self.assertEqual(mock_config.tts.provider, "system")
        self.assertIn(mock_config.audio.device, XVF_INPUT_DEVICES)
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
        self.assertEqual(mify_config.audio.device, XVF_INPUT_DEVICE)
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
        self.assertEqual(mify_config.tts.playback_device, XVF_OUTPUT_DEVICE)
        self.assertTrue(mify_config.debug.enabled)
        self.assertEqual(wake_config.wake.engine, "openwakeword")
        self.assertEqual(wake_config.wake.model, "alexa")
        self.assertEqual(wake_config.wake.threshold, 0.5)
        self.assertEqual(wake_config.wake.inference_framework, "onnx")
        self.assertTrue(wake_config.wake.debug)
        self.assertTrue(wake_config.wake_ack.enabled)
        self.assertEqual(wake_config.wake_ack.wav_path, "default")
        self.assertIn(wake_config.wake_ack.playback_device, XVF_OUTPUT_DEVICES)
        self.assertIn(wake_config.audio.device, XVF_INPUT_DEVICES)
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
        self.assertIn(wake_config.tts.playback_device, XVF_OUTPUT_DEVICES)
        self.assertTrue(wake_config.debug.system_input_dump_enabled)
        self.assertEqual(wake_config.debug.system_input_dump_segment_seconds, 30)
        self.assertTrue(wake_config.debug.voice_path_dump_enabled)
        self.assertEqual(local_tts_config.tts.provider, "openai_speech")
        self.assertEqual(local_tts_config.tts.endpoint, "http://127.0.0.1:8000")
        self.assertEqual(local_tts_config.tts.audio_format, "pcm")
        self.assertTrue(local_tts_config.tts.stream)
        self.assertEqual(local_tts_config.tts.playback_device, XVF_OUTPUT_DEVICE)
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
        self.assertEqual(aliyun_tts_config.audio.device, XVF_INPUT_DEVICE)
        self.assertEqual(aliyun_tts_config.tts.playback_device, XVF_OUTPUT_DEVICE)
        self.assertEqual(
            aliyun_tts_config.tts.endpoint,
            "wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1",
        )
        self.assertEqual(aliyun_wake_config.wake.engine, "openwakeword")
        self.assertTrue(aliyun_wake_config.audio.debug)
        self.assertEqual(aliyun_wake_config.audio.device, XVF_INPUT_DEVICE)
        self.assertEqual(aliyun_wake_config.wake.model, "alexa")
        self.assertEqual(aliyun_wake_config.wake.threshold, 0.5)
        self.assertTrue(aliyun_wake_config.wake.debug)
        self.assertTrue(aliyun_wake_config.wake_ack.enabled)
        self.assertEqual(
            aliyun_wake_config.wake_ack.playback_device,
            aliyun_wake_config.tts.playback_device,
        )
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
        self.assertIsNotNone(aliyun_wake_config.tts.playback_device)
        self.assertEqual(aliyun_wake_config.tts.volume, 50)
        self.assertTrue(aliyun_wake_config.tts.limiter_enabled)
        self.assertEqual(aliyun_wake_config.tts.limiter_threshold, 0.92)
        self.assertTrue(aliyun_wake_config.debug.system_input_dump_enabled)
        self.assertEqual(aliyun_wake_config.debug.system_input_dump_segment_seconds, 30)
        self.assertTrue(aliyun_wake_config.debug.voice_path_dump_enabled)
        self.assertTrue(aliyun_wake_config.tools.allow_volume)
        self.assertTrue(aliyun_wake_config.tools.allow_music)
        self.assertTrue(aliyun_wake_config.tools.allow_miot)
        self.assertTrue(aliyun_wake_config.tools.allow_search)
        self.assertEqual(aliyun_wake_config.tools.default_weather_location, "北京昌平")
        self.assertEqual(aliyun_wake_config.music.provider, "meting")
        self.assertEqual(aliyun_wake_config.music.server, "netease")
        self.assertEqual(aliyun_wake_config.music.playback_device, XVF_OUTPUT_DEVICE)
        self.assertEqual(aliyun_wake_config.music.playback_sample_rate, 16000)
        self.assertEqual(aliyun_wake_config.music.playback_channels, 2)
        self.assertEqual(aliyun_wake_config.music.playback_volume, 1.0)
        self.assertEqual(aliyun_wake_config.music.ducking_volume_factor, 0.2)
        self.assertTrue(aliyun_wake_config.music.limiter_enabled)
        self.assertEqual(aliyun_wake_config.music.limiter_threshold, 0.92)
        self.assertEqual(aliyun_wake_config.search.provider, "auto")
        self.assertEqual(aliyun_wake_config.search.tavily_api_key_env, "TAVILY_API_KEY")
        self.assertTrue(aliyun_wake_config.search.baidu_ai_enabled)
        self.assertEqual(
            aliyun_wake_config.search.baidu_ai_api_key_env,
            "QIANFAN_API_KEY",
        )
        self.assertTrue(aliyun_wake_config.xiaomi_miot.enabled)
        self.assertEqual(aliyun_wake_config.xiaomi_miot.token_file, ".voiceui/miot_token.json")
        self.assertTrue(aliyun_wake_config.xiaomi_miot.control_verify)
        self.assertTrue(aliyun_wake_config.conversation.barge_in_enabled)


if __name__ == "__main__":
    unittest.main()
