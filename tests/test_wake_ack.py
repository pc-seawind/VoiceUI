from __future__ import annotations

import sys
import types
import unittest
import wave
from unittest.mock import patch

from voiceui.models import WakeAckConfig
from voiceui.wake_ack import (
    DisabledWakeAckPlayer,
    WavWakeAckPlayer,
    _convert_pcm16_channels,
    _play_pcm16,
    _resample_pcm16,
    _select_output_format,
    create_wake_ack_player,
    resolve_wake_ack_path,
)


class WakeAckTests(unittest.TestCase):
    def test_disabled_ack_player(self) -> None:
        player = create_wake_ack_player(WakeAckConfig(enabled=False))

        self.assertIsInstance(player, DisabledWakeAckPlayer)

    def test_enabled_ack_player_uses_wav(self) -> None:
        player = create_wake_ack_player(WakeAckConfig(enabled=True))

        self.assertIsInstance(player, WavWakeAckPlayer)

    def test_default_ack_resource_exists(self) -> None:
        path = resolve_wake_ack_path()

        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        with wave.open(str(path), "rb") as wav:
            self.assertEqual(wav.getframerate(), 24000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)

    def test_select_output_format_falls_back_to_device_default_stereo(self) -> None:
        class FakeSoundDevice:
            def check_output_settings(self, **kwargs):
                if kwargs["samplerate"] != 16000 or kwargs["channels"] != 2:
                    raise ValueError("unsupported")

            def query_devices(self, _device, _kind):
                return {"default_samplerate": 16000, "max_output_channels": 2}

        sample_rate, channels = _select_output_format(
            FakeSoundDevice(),
            device=22,
            requested_sample_rate=24000,
            source_channels=1,
        )

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(channels, 2)

    def test_resample_pcm16_changes_sample_count(self) -> None:
        pcm = b"\x00\x00" * 240

        converted, resampler = _resample_pcm16(
            pcm,
            source_rate=24000,
            target_rate=16000,
            channels=1,
        )

        self.assertEqual(len(converted), 160 * 2)
        self.assertIn(resampler, {"scipy", "audioop"})

    def test_convert_pcm16_channels_duplicates_mono_to_stereo(self) -> None:
        pcm = b"\x01\x00\x02\x00"

        converted = _convert_pcm16_channels(
            pcm,
            source_channels=1,
            target_channels=2,
        )

        self.assertEqual(converted, b"\x01\x00\x01\x00\x02\x00\x02\x00")

    def test_play_pcm16_applies_limiter_safe_gain(self) -> None:
        written: list[bytes] = []

        class FakeStream:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def write(self, data: bytes):
                written.append(data)

        fake_sd = types.SimpleNamespace(
            RawOutputStream=FakeStream,
            check_output_settings=lambda **_kwargs: None,
            query_devices=lambda *_args, **_kwargs: {
                "default_samplerate": 24000,
                "max_output_channels": 1,
            },
        )
        samples = [17344, -17344, 1000]
        pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)

        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            _play_pcm16(
                pcm,
                sample_rate=24000,
                channels=1,
                playback_gain_db=12.0,
                limiter_enabled=True,
                limiter_threshold=0.92,
            )

        played = [
            int.from_bytes(written[0][index : index + 2], "little", signed=True)
            for index in range(0, len(written[0]), 2)
        ]
        self.assertLessEqual(max(abs(sample) for sample in played), 30146)
        self.assertLess(abs(played[2]), 2000)


if __name__ == "__main__":
    unittest.main()
