from __future__ import annotations

import unittest

from voiceui.models import WakeConfig
from voiceui.wake import (
    DisabledWakeDetector,
    ManualWakeDetector,
    OpenWakeWordDetector,
    _best_prediction,
    _normalize_openwakeword_label,
    _resolve_openwakeword_models,
    create_wake_detector,
)


class WakeTests(unittest.TestCase):
    def test_create_manual_wake_detector(self) -> None:
        detector = create_wake_detector(WakeConfig(engine="manual"))

        self.assertIsInstance(detector, ManualWakeDetector)

    def test_create_disabled_wake_detector(self) -> None:
        detector = create_wake_detector(WakeConfig(engine="disabled"))

        self.assertIsInstance(detector, DisabledWakeDetector)

    def test_create_openwakeword_detector_does_not_load_model(self) -> None:
        detector = create_wake_detector(WakeConfig(engine="openwakeword"))

        self.assertIsInstance(detector, OpenWakeWordDetector)

    def test_best_prediction_casts_confidence_to_float(self) -> None:
        label, confidence = _best_prediction({"hey_jarvis": 0.2, "alexa": 0.7})

        self.assertEqual(label, "alexa")
        self.assertIsInstance(confidence, float)
        self.assertEqual(confidence, 0.7)

    def test_openwakeword_model_name_normalization(self) -> None:
        self.assertEqual(_normalize_openwakeword_label("Hey Jarvis"), "hey_jarvis")
        self.assertEqual(_normalize_openwakeword_label("hey-jarvis"), "hey_jarvis")

    def test_resolve_openwakeword_builtin_model(self) -> None:
        models = _resolve_openwakeword_models(
            "hey jarvis",
            available_models=["alexa", "hey_jarvis"],
        )

        self.assertEqual(models, ["hey_jarvis"])

    def test_resolve_openwakeword_any_model(self) -> None:
        models = _resolve_openwakeword_models(
            "any",
            available_models=["alexa", "hey_jarvis"],
        )

        self.assertEqual(models, [])

    def test_resolve_openwakeword_rejects_unknown_model(self) -> None:
        with self.assertRaises(ValueError):
            _resolve_openwakeword_models(
                "not_a_wake_word",
                available_models=["alexa", "hey_jarvis"],
            )


if __name__ == "__main__":
    unittest.main()
