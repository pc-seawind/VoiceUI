from __future__ import annotations

import time

from voiceui.audio import AudioInput
from voiceui.models import WakeConfig, WakeEvent


class WakeDetector:
    def wait(self, audio: AudioInput) -> WakeEvent:
        raise NotImplementedError


class DisabledWakeDetector(WakeDetector):
    def wait(self, audio: AudioInput) -> WakeEvent:
        return WakeEvent(engine="disabled", confidence=1.0, label="disabled")


class ManualWakeDetector(WakeDetector):
    def wait(self, audio: AudioInput) -> WakeEvent:
        input("wake> press Enter, then speak your command...")
        return WakeEvent(engine="manual", confidence=1.0, label="enter")


class OpenWakeWordDetector(WakeDetector):
    def __init__(self, config: WakeConfig):
        self.config = config
        self._model = None

    def wait(self, audio: AudioInput) -> WakeEvent:
        try:
            import numpy as np  # type: ignore[import-untyped]
            from openwakeword.model import Model  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord requires optional wake dependencies. "
                "Install with: pip install -e \".[wake]\""
            ) from exc

        if self._model is None:
            self._model = Model(wakeword_models=[self.config.model])

        started = time.monotonic()
        for chunk in audio.chunks():
            predictions = self._model.predict(np.frombuffer(chunk, dtype=np.int16))
            label, confidence = _best_prediction(predictions)
            if confidence >= self.config.threshold:
                return WakeEvent(engine="openwakeword", confidence=confidence, label=label)
            if self.config.max_wait_seconds > 0:
                elapsed = time.monotonic() - started
                if elapsed >= self.config.max_wait_seconds:
                    raise TimeoutError("Timed out waiting for wake word.")

        raise RuntimeError("Audio stream ended while waiting for wake word.")


class SherpaOnnxDetector(WakeDetector):
    def __init__(self, config: WakeConfig):
        self.config = config

    def wait(self, audio: AudioInput) -> WakeEvent:
        raise NotImplementedError(
            "sherpa-onnx KWS is planned for Chinese custom wake words. "
            "Use openwakeword or disabled for the current MVP."
        )


def create_wake_detector(config: WakeConfig) -> WakeDetector:
    if config.engine == "disabled":
        return DisabledWakeDetector()
    if config.engine == "manual":
        return ManualWakeDetector()
    if config.engine == "openwakeword":
        return OpenWakeWordDetector(config)
    if config.engine == "sherpa_onnx":
        return SherpaOnnxDetector(config)
    raise ValueError(f"Unsupported wake engine: {config.engine}")


def _best_prediction(predictions: dict[str, float]) -> tuple[str, float]:
    if not predictions:
        return "", 0.0
    return max(predictions.items(), key=lambda item: item[1])
