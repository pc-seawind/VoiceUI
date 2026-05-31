from __future__ import annotations

import unittest

from voiceui.models import TtsConfig
from voiceui.tts import ConsoleTextToSpeech, SystemTextToSpeech, create_tts


class TtsTests(unittest.TestCase):
    def test_create_console_tts(self) -> None:
        tts = create_tts(TtsConfig(provider="console"))

        self.assertIsInstance(tts, ConsoleTextToSpeech)

    def test_create_system_tts(self) -> None:
        tts = create_tts(TtsConfig(provider="system"))

        self.assertIsInstance(tts, SystemTextToSpeech)


if __name__ == "__main__":
    unittest.main()
