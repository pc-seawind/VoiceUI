from __future__ import annotations

import unittest

from voiceui.models import WakeConfig
from voiceui.wake import DisabledWakeDetector, ManualWakeDetector, create_wake_detector


class WakeTests(unittest.TestCase):
    def test_create_manual_wake_detector(self) -> None:
        detector = create_wake_detector(WakeConfig(engine="manual"))

        self.assertIsInstance(detector, ManualWakeDetector)

    def test_create_disabled_wake_detector(self) -> None:
        detector = create_wake_detector(WakeConfig(engine="disabled"))

        self.assertIsInstance(detector, DisabledWakeDetector)


if __name__ == "__main__":
    unittest.main()
