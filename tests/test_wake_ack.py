from __future__ import annotations

import unittest
import wave

from voiceui.models import WakeAckConfig
from voiceui.wake_ack import (
    DisabledWakeAckPlayer,
    WavWakeAckPlayer,
    create_wake_ack_player,
    resolve_wake_ack_path,
    _resample_pcm16,
    _select_output_sample_rate,
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

    def test_select_output_sample_rate_falls_back_to_device_default(self) -> None:
        class FakeSoundDevice:
            def check_output_settings(self, **kwargs):
                if kwargs["samplerate"] != 16000:
                    raise ValueError("unsupported")

            def query_devices(self, _device, _kind):
                return {"default_samplerate": 16000}

        sample_rate = _select_output_sample_rate(
            FakeSoundDevice(),
            device=22,
            requested_sample_rate=24000,
            channels=1,
        )

        self.assertEqual(sample_rate, 16000)

    def test_resample_pcm16_changes_sample_count(self) -> None:
        pcm = b"\x00\x00" * 240

        converted = _resample_pcm16(
            pcm,
            source_rate=24000,
            target_rate=16000,
            channels=1,
        )

        self.assertEqual(len(converted), 160 * 2)


if __name__ == "__main__":
    unittest.main()
