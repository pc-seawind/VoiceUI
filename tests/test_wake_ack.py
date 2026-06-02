from __future__ import annotations

import unittest
import wave

from voiceui.models import WakeAckConfig
from voiceui.wake_ack import (
    DisabledWakeAckPlayer,
    WavWakeAckPlayer,
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
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)


if __name__ == "__main__":
    unittest.main()
