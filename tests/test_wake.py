from __future__ import annotations

import contextlib
import io
import time
import unittest

from voiceui.logs import configure_logging, reset_logging
from voiceui.models import LoggingConfig, WakeConfig
from voiceui.wake import (
    DisabledWakeDetector,
    ManualWakeDetector,
    OpenWakeWordDetector,
    _best_prediction,
    _format_predictions,
    _normalize_openwakeword_label,
    _PcmRingBuffer,
    _resolve_openwakeword_models,
    create_wake_detector,
)


class WakeTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_logging()

    def test_create_manual_wake_detector(self) -> None:
        detector = create_wake_detector(WakeConfig(engine="manual"))

        self.assertIsInstance(detector, ManualWakeDetector)

    def test_create_disabled_wake_detector(self) -> None:
        detector = create_wake_detector(WakeConfig(engine="disabled"))

        self.assertIsInstance(detector, DisabledWakeDetector)

    def test_create_openwakeword_detector_does_not_load_model(self) -> None:
        detector = create_wake_detector(WakeConfig(engine="openwakeword"))

        self.assertIsInstance(detector, OpenWakeWordDetector)

    def test_openwakeword_resets_model_state_before_wait(self) -> None:
        class FakeAudio:
            sample_rate = 16000

            def chunks(self):
                yield b"\x00\x00" * 1280

        class FakeModel:
            def __init__(self):
                self.reset_calls = 0
                self.predict_calls = 0

            def reset(self):
                self.reset_calls += 1

            def predict(self, _samples):
                self.predict_calls += 1
                return {"alexa": 0.9}

        detector = OpenWakeWordDetector(WakeConfig(engine="openwakeword", threshold=0.5))
        fake_model = FakeModel()
        detector._model = fake_model

        event = detector.wait(FakeAudio())

        self.assertEqual(event.label, "alexa")
        self.assertEqual(fake_model.reset_calls, 1)
        self.assertEqual(fake_model.predict_calls, 1)

    def test_openwakeword_requires_consecutive_trigger_hits(self) -> None:
        class FakeAudio:
            sample_rate = 16000

            def chunks(self):
                for _index in range(4):
                    yield b"\x00\x00" * 1280

        class FakeModel:
            def __init__(self):
                self.predict_calls = 0
                self.scores = [0.9, 0.0, 0.8, 0.85]

            def reset(self):
                return None

            def predict(self, _samples):
                score = self.scores[self.predict_calls]
                self.predict_calls += 1
                return {"alexa": score}

        detector = OpenWakeWordDetector(
            WakeConfig(engine="openwakeword", threshold=0.5, trigger_level=2)
        )
        fake_model = FakeModel()
        detector._model = fake_model

        event = detector.wait(FakeAudio())

        self.assertEqual(event.label, "alexa")
        self.assertEqual(fake_model.predict_calls, 4)

    def test_wake_debug_does_not_print_score_without_continuous_switch(self) -> None:
        class FakeAudio:
            sample_rate = 16000
            block_ms = 80

            def chunks(self):
                yield b"\x00\x00" * 1280
                time.sleep(0.12)
                yield b"\x00\x00" * 1280

        class FakeModel:
            def __init__(self):
                self.predict_calls = 0

            def reset(self):
                return None

            def predict(self, _samples):
                self.predict_calls += 1
                return {"alexa": 0.0 if self.predict_calls == 1 else 0.9}

        configure_logging(LoggingConfig())
        detector = OpenWakeWordDetector(
            WakeConfig(
                engine="openwakeword",
                threshold=0.5,
                debug=True,
                debug_interval_seconds=0.01,
            )
        )
        detector._model = FakeModel()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            detector.wait(FakeAudio())

        self.assertIn("module=wake | event=debug_enabled", output.getvalue())
        self.assertNotIn("module=wake | event=score", output.getvalue())

    def test_wake_score_prints_when_continuous_switch_is_enabled(self) -> None:
        class FakeAudio:
            sample_rate = 16000
            block_ms = 80

            def chunks(self):
                yield b"\x00\x00" * 1280
                time.sleep(0.12)
                yield b"\x00\x00" * 1280

        class FakeModel:
            def __init__(self):
                self.predict_calls = 0

            def reset(self):
                return None

            def predict(self, _samples):
                self.predict_calls += 1
                return {"alexa": 0.0 if self.predict_calls == 1 else 0.9}

        configure_logging(LoggingConfig(continuous={"wake.score": True}))
        detector = OpenWakeWordDetector(
            WakeConfig(
                engine="openwakeword",
                threshold=0.5,
                debug=True,
                debug_interval_seconds=0.01,
            )
        )
        detector._model = FakeModel()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            detector.wait(FakeAudio())

        self.assertIn("module=wake | event=score", output.getvalue())

    def test_best_prediction_casts_confidence_to_float(self) -> None:
        label, confidence = _best_prediction({"hey_jarvis": 0.2, "alexa": 0.7})

        self.assertEqual(label, "alexa")
        self.assertIsInstance(confidence, float)
        self.assertEqual(confidence, 0.7)

    def test_format_predictions_orders_top_scores(self) -> None:
        self.assertEqual(
            _format_predictions({"low": 0.1, "high": 0.9, "mid": 0.5}, limit=2),
            "high:0.900,mid:0.500",
        )

    def test_pcm_ring_buffer_keeps_recent_audio(self) -> None:
        buffer = _PcmRingBuffer(max_bytes=6)

        buffer.append(b"aa")
        buffer.append(b"bb")
        buffer.append(b"cc")
        buffer.append(b"dd")

        self.assertEqual(buffer.pcm(), b"bbccdd")

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
