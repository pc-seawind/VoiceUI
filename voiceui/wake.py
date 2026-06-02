from __future__ import annotations

import math
import time
from collections import deque
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
        debug_enabled = self.config.debug
        debug_interval = max(0.1, self.config.debug_interval_seconds)
        next_debug_at = started + debug_interval
        debug_window = _WakeDebugWindow()
        audio_buffer = _PcmRingBuffer(
            max_bytes=int(
                audio.sample_rate * 2 * max(0.0, self.config.debug_audio_seconds)
            )
        )
        if debug_enabled:
            print(
                "wake_debug> enabled "
                f"{_describe_audio_input(audio)} "
                f"model={self.config.model} threshold={self.config.threshold:.3f} "
                f"framework={self.config.inference_framework}"
            )

        for chunk in audio.chunks():
            audio_buffer.append(chunk)
            samples = np.frombuffer(chunk, dtype=np.int16)
            predict_started = time.monotonic()
            predictions = self._model.predict(samples)
            predict_ms = (time.monotonic() - predict_started) * 1000
            label, confidence = _best_prediction(predictions)
            if debug_enabled:
                now = time.monotonic()
                debug_window.update(
                    samples=samples,
                    predictions=predictions,
                    label=label,
                    confidence=confidence,
                    predict_ms=predict_ms,
                    audio_ms=int(len(samples) / audio.sample_rate * 1000),
                )
                if now >= next_debug_at:
                    elapsed = now - started
                    print(
                        "wake_debug> "
                        f"elapsed_s={elapsed:.1f} "
                        f"{debug_window.format(self.config.threshold, self.config.debug_top_predictions)}"
                    )
                    debug_window.reset()
                    next_debug_at = now + debug_interval
            if confidence >= self.config.threshold:
                if debug_enabled:
                    elapsed = time.monotonic() - started
                    print(
                        "wake_debug> detected "
                        f"elapsed_s={elapsed:.2f} label={label} confidence={confidence:.3f} "
                        f"threshold={self.config.threshold:.3f}"
                    )
                pcm = audio_buffer.pcm()
                return WakeEvent(
                    engine="openwakeword",
                    confidence=confidence,
                    label=label,
                    pcm=pcm,
                    sample_rate=audio.sample_rate,
                    duration_ms=_pcm_duration_ms(pcm, audio.sample_rate),
                )
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


def list_openwakeword_models() -> list[str]:
    try:
        import openwakeword  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "openWakeWord is required to list wake models. "
            "Install with: pip install -e \".[wake]\""
        ) from exc
    return sorted(getattr(openwakeword, "MODELS", {}).keys())


def _best_prediction(predictions: dict[str, float]) -> tuple[str, float]:
    if not predictions:
        return "", 0.0
    label, confidence = max(predictions.items(), key=lambda item: item[1])
    return label, float(confidence)


class _WakeDebugWindow:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.chunks = 0
        self.audio_ms = 0
        self.predict_total_ms = 0.0
        self.predict_max_ms = 0.0
        self.best_label = ""
        self.best_confidence = 0.0
        self.last_label = ""
        self.last_confidence = 0.0
        self.last_predictions: dict[str, float] = {}
        self.last_stats = {
            "rms": 0.0,
            "peak": 0,
            "dbfs": -120.0,
            "near_zero_pct": 100.0,
            "clipped_pct": 0.0,
        }

    def update(
        self,
        *,
        samples,
        predictions: dict[str, float],
        label: str,
        confidence: float,
        predict_ms: float,
        audio_ms: int,
    ) -> None:
        self.chunks += 1
        self.audio_ms += audio_ms
        self.predict_total_ms += predict_ms
        self.predict_max_ms = max(self.predict_max_ms, predict_ms)
        self.last_label = label
        self.last_confidence = confidence
        self.last_predictions = predictions
        if confidence > self.best_confidence:
            self.best_label = label
            self.best_confidence = confidence
        self.last_stats = _pcm16_stats(samples)

    def format(self, threshold: float, top_predictions: int) -> str:
        avg_predict_ms = self.predict_total_ms / self.chunks if self.chunks else 0.0
        stats = self.last_stats
        return (
            f"chunks={self.chunks} audio_ms={self.audio_ms} "
            f"rms={stats['rms']:.1f} peak={stats['peak']} dbfs={stats['dbfs']:.1f} "
            f"near_zero_pct={stats['near_zero_pct']:.1f} clipped_pct={stats['clipped_pct']:.2f} "
            f"last={self.last_label}:{self.last_confidence:.3f} "
            f"best_window={self.best_label}:{self.best_confidence:.3f} "
            f"threshold={threshold:.3f} "
            f"top={_format_predictions(self.last_predictions, top_predictions)} "
            f"predict_avg_ms={avg_predict_ms:.1f} predict_max_ms={self.predict_max_ms:.1f}"
        )


class _PcmRingBuffer:
    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, max_bytes)
        self._chunks: deque[bytes] = deque()
        self._size = 0

    def append(self, chunk: bytes) -> None:
        if self.max_bytes <= 0 or not chunk:
            return
        self._chunks.append(chunk)
        self._size += len(chunk)
        while self._size > self.max_bytes and self._chunks:
            removed = self._chunks.popleft()
            self._size -= len(removed)

    def pcm(self) -> bytes:
        return b"".join(self._chunks)


def _pcm_duration_ms(pcm: bytes, sample_rate: int) -> int:
    if sample_rate <= 0:
        return 0
    return int(len(pcm) / 2 / sample_rate * 1000)


def _pcm16_stats(samples) -> dict[str, float | int]:
    if len(samples) == 0:
        return {
            "rms": 0.0,
            "peak": 0,
            "dbfs": -120.0,
            "near_zero_pct": 100.0,
            "clipped_pct": 0.0,
        }
    int_samples = samples.astype("int32")
    abs_samples = abs(int_samples)
    peak = int(abs_samples.max())
    rms = float(math.sqrt(float((int_samples.astype("float64") ** 2).mean())))
    dbfs = 20.0 * math.log10(max(rms, 1.0) / 32768.0)
    near_zero_pct = float((abs_samples <= 3).mean() * 100.0)
    clipped_pct = float((abs_samples >= 32760).mean() * 100.0)
    return {
        "rms": rms,
        "peak": peak,
        "dbfs": dbfs,
        "near_zero_pct": near_zero_pct,
        "clipped_pct": clipped_pct,
    }


def _format_predictions(predictions: dict[str, float], limit: int) -> str:
    if not predictions:
        return "-"
    top_items = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    top_items = top_items[: max(1, limit)]
    return ",".join(f"{label}:{float(confidence):.3f}" for label, confidence in top_items)


def _describe_audio_input(audio: AudioInput) -> str:
    config = getattr(audio, "config", None)
    selected_channel = getattr(audio, "selected_channel", "?")
    if config is None:
        return (
            f"sample_rate={audio.sample_rate} block_ms={audio.block_ms} "
            f"selected_channel={selected_channel}"
        )
    return (
        f"device={getattr(config, 'device', None)} "
        f"channels={getattr(config, 'channels', '?')} "
        f"selected_channel={selected_channel} "
        f"input_gain_db={getattr(config, 'input_gain_db', 0.0)} "
        f"sample_rate={audio.sample_rate} block_ms={audio.block_ms}"
    )


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
