from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from voiceui.audio import AudioInput
from voiceui.models import WakeConfig, WakeEvent

_OPENWAKEWORD_ANY_MODELS = {"", "*", "all", "any"}
_OPENWAKEWORD_MODEL_SUFFIXES = {".onnx", ".tflite"}


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

    def _load_model(self):
        if self.config.inference_framework == "onnx":
            missing_dependency = "openWakeWord and onnxruntime"
            install_hint = 'pip install -e ".[wake]"'
        else:
            missing_dependency = "openWakeWord and tflite-runtime"
            install_hint = "pip install openwakeword tflite-runtime"

        try:
            import openwakeword  # type: ignore[import-untyped]
            from openwakeword.model import Model  # type: ignore[import-untyped]
            from openwakeword.utils import download_models  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                f"{missing_dependency} are required for wake detection. "
                f"Install with: {install_hint}"
            ) from exc

        model_args = _resolve_openwakeword_models(
            self.config.model,
            available_models=getattr(openwakeword, "MODELS", {}).keys(),
        )
        _download_openwakeword_models(download_models, model_args)
        loaded = "all built-ins" if not model_args else ", ".join(model_args)
        print(
            "wake> loading openWakeWord "
            f"model={loaded} framework={self.config.inference_framework}"
        )
        try:
            return Model(
                wakeword_models=model_args,
                inference_framework=self.config.inference_framework,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize openWakeWord "
                f"model={self.config.model!r} framework={self.config.inference_framework}: {exc}"
            ) from exc

    def wait(self, audio: AudioInput) -> WakeEvent:
        if audio.sample_rate != 16000:
            raise RuntimeError(
                f"openWakeWord requires 16 kHz audio, got sample_rate={audio.sample_rate}"
            )

        try:
            import numpy as np  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord requires numpy. Install with: pip install -e \".[wake]\""
            ) from exc

        if self._model is None:
            self._model = self._load_model()

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
    label, confidence = max(predictions.items(), key=lambda item: item[1])
    return label, float(confidence)


def _normalize_openwakeword_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_openwakeword_models(model: str, available_models: Iterable[str]) -> list[str]:
    requested = model.strip()
    if _normalize_openwakeword_label(requested) in _OPENWAKEWORD_ANY_MODELS:
        return []

    available_model_list = list(available_models)
    requested_path = Path(requested).expanduser()
    if requested_path.exists():
        return [str(requested_path)]
    if requested_path.suffix.lower() in _OPENWAKEWORD_MODEL_SUFFIXES:
        raise FileNotFoundError(f"openWakeWord model path does not exist: {requested}")

    by_normalized_name = {
        _normalize_openwakeword_label(name): name for name in available_model_list
    }
    normalized = _normalize_openwakeword_label(requested)
    if normalized in by_normalized_name:
        return [by_normalized_name[normalized]]

    built_ins = ", ".join(sorted(available_model_list))
    raise ValueError(
        f"Unknown openWakeWord model {model!r}. Use one of [{built_ins}], "
        "use 'any' to load all built-ins, or set a .onnx/.tflite model path."
    )


def _download_openwakeword_models(download_models, model_args: list[str]) -> None:
    if not model_args:
        download_models(model_names=[])
        return

    built_in_models = [model for model in model_args if not Path(model).exists()]
    if built_in_models:
        download_models(model_names=built_in_models)
