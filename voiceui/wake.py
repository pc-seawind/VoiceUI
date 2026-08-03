from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from voiceui.audio import AudioInput
from voiceui.logs import is_log_enabled, log_continuous, log_event
from voiceui.models import WakeConfig, WakeEvent

_OPENWAKEWORD_ANY_MODELS = {"", "*", "all", "any"}
_OPENWAKEWORD_MODEL_SUFFIXES = {".onnx", ".tflite"}


class WakeDetector:
    def warm_up(self) -> bool:
        return False

    def wait(self, audio: AudioInput) -> WakeEvent:
        raise NotImplementedError


class DisabledWakeDetector(WakeDetector):
    def wait(self, audio: AudioInput) -> WakeEvent:
        return WakeEvent(engine="disabled", confidence=1.0, label="disabled")


class ManualWakeDetector(WakeDetector):
    def wait(self, audio: AudioInput) -> WakeEvent:
        input("Press Enter, then speak your command...")
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
        log_event(
            "wake",
            "loading",
            log_id="wake.loading",
            engine="openwakeword",
            model=loaded,
            framework=self.config.inference_framework,
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

    def warm_up(self) -> bool:
        if self._model is None:
            self._model = self._load_model()
        return True

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
        self._reset_model_state()

        started = time.monotonic()
        score_log_enabled = is_log_enabled(
            "wake.score",
            kind="continuous",
        )
        debug_enabled_log_enabled = is_log_enabled(
            "wake.debug_enabled",
            default_enabled=self.config.debug,
        )
        detected_debug_log_enabled = is_log_enabled(
            "wake.detected_debug",
            default_enabled=self.config.debug,
        )
        debug_enabled = (
            self.config.debug
            or score_log_enabled
            or debug_enabled_log_enabled
            or detected_debug_log_enabled
        )
        debug_interval = max(0.1, self.config.debug_interval_seconds)
        next_debug_at = started + debug_interval
        debug_window = _WakeDebugWindow()
        trigger_level = max(1, int(self.config.trigger_level))
        trigger_hits = 0
        trigger_label = ""
        audio_buffer = _PcmRingBuffer(
            max_bytes=int(
                audio.sample_rate * 2 * max(0.0, self.config.debug_audio_seconds)
            )
        )
        if debug_enabled_log_enabled:
            log_event(
                "wake",
                "debug_enabled",
                log_id="wake.debug_enabled",
                default_enabled=self.config.debug,
                **_audio_input_params(audio),
                model=self.config.model,
                threshold=f"{self.config.threshold:.3f}",
                framework=self.config.inference_framework,
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
                    log_continuous(
                        "wake",
                        "score",
                        log_id="wake.score",
                        elapsed_s=f"{elapsed:.1f}",
                        **debug_window.snapshot(
                            self.config.threshold,
                            self.config.debug_top_predictions,
                        ),
                    )
                    debug_window.reset()
                    next_debug_at = now + debug_interval
            if confidence >= self.config.threshold:
                if label == trigger_label:
                    trigger_hits += 1
                else:
                    trigger_label = label
                    trigger_hits = 1
            else:
                trigger_label = ""
                trigger_hits = 0

            if trigger_hits >= trigger_level:
                if detected_debug_log_enabled:
                    elapsed = time.monotonic() - started
                    log_event(
                        "wake",
                        "detected_debug",
                        log_id="wake.detected_debug",
                        default_enabled=self.config.debug,
                        elapsed_s=f"{elapsed:.2f}",
                        label=label,
                        confidence=f"{confidence:.3f}",
                        threshold=f"{self.config.threshold:.3f}",
                        trigger_level=trigger_level,
                        trigger_hits=trigger_hits,
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

    def _reset_model_state(self) -> None:
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            reset()


class SherpaOnnxDetector(WakeDetector):
    def __init__(self, config: WakeConfig):
        self.config = config

    def wait(self, audio: AudioInput) -> WakeEvent:
        raise NotImplementedError(
            "sherpa-onnx KWS is planned for Chinese custom wake words. "
            "Use openwakeword or disabled for the current MVP."
        )


class WekwsMhaDetector(WakeDetector):
    def __init__(self, config: WakeConfig):
        self.config = config
        self._runtime = None

    def _load_runtime(self):
        from voiceui.wekws_onnx import LeelaMhaOnnxRuntime

        log_event(
            "wake",
            "loading",
            log_id="wake.loading",
            engine="wekws_mha",
            model=self.config.model,
            window_seconds=f"{self.config.wekws_window_seconds:.1f}",
            hop_frames=max(1, int(self.config.wekws_hop_frames)),
        )
        return LeelaMhaOnnxRuntime(self.config.model)

    def warm_up(self) -> bool:
        if self._runtime is None:
            self._runtime = self._load_runtime()
        return True

    def wait(self, audio: AudioInput) -> WakeEvent:
        if self._runtime is None:
            self._runtime = self._load_runtime()
        if audio.sample_rate != self._runtime.sample_rate:
            raise RuntimeError(
                "Leela MHA requires "
                f"{self._runtime.sample_rate} Hz audio, got sample_rate={audio.sample_rate}"
            )

        started = time.monotonic()
        score_log_enabled = is_log_enabled("wake.score", kind="continuous")
        debug_enabled_log_enabled = is_log_enabled(
            "wake.debug_enabled",
            default_enabled=self.config.debug,
        )
        detected_debug_log_enabled = is_log_enabled(
            "wake.detected_debug",
            default_enabled=self.config.debug,
        )
        debug_enabled = (
            self.config.debug
            or score_log_enabled
            or debug_enabled_log_enabled
            or detected_debug_log_enabled
        )
        debug_interval = max(0.1, self.config.debug_interval_seconds)
        next_debug_at = started + debug_interval
        debug_window = _WakeDebugWindow()
        trigger_level = max(1, int(self.config.trigger_level))
        trigger_hits = 0
        trigger_label = ""
        window_seconds = max(0.1, self.config.wekws_window_seconds)
        hop_frames = max(1, int(self.config.wekws_hop_frames))
        window_bytes = _pcm16_bytes_for_seconds(audio.sample_rate, window_seconds)
        model_audio = _PcmRingBuffer(max_bytes=window_bytes)
        has_scored_window = False
        frames_since_prediction = 0
        if debug_enabled_log_enabled:
            log_event(
                "wake",
                "debug_enabled",
                log_id="wake.debug_enabled",
                default_enabled=self.config.debug,
                **_audio_input_params(audio),
                model=self.config.model,
                threshold=f"{self.config.threshold:.3f}",
                framework=getattr(self._runtime, "backend", "wekws_mha"),
                window_seconds=f"{window_seconds:.1f}",
                hop_frames=hop_frames,
                hop_ms=hop_frames * audio.block_ms,
            )

        for chunk in audio.chunks():
            model_audio.append(chunk)
            if model_audio.size < window_bytes:
                if (
                    self.config.max_wait_seconds > 0
                    and time.monotonic() - started >= self.config.max_wait_seconds
                ):
                    raise TimeoutError("Timed out waiting for wake word.")
                continue
            if has_scored_window:
                frames_since_prediction += 1
                if frames_since_prediction < hop_frames:
                    if (
                        self.config.max_wait_seconds > 0
                        and time.monotonic() - started >= self.config.max_wait_seconds
                    ):
                        raise TimeoutError("Timed out waiting for wake word.")
                    continue
                frames_since_prediction = 0
            else:
                has_scored_window = True

            model_pcm = model_audio.pcm()
            predict_started = time.monotonic()
            predictions = self._runtime.score_pcm(model_pcm, audio.sample_rate)
            predict_ms = (time.monotonic() - predict_started) * 1000
            label, confidence = _best_prediction(predictions)
            threshold = self.config.wekws_label_thresholds.get(label, self.config.threshold)
            if debug_enabled:
                try:
                    import numpy as np  # type: ignore[import-untyped]
                except ImportError as exc:
                    raise RuntimeError(
                        "WeKWS MHA wake detection requires numpy. "
                        'Install with: pip install -e ".[wake]"'
                    ) from exc
                samples = np.frombuffer(chunk, dtype=np.int16)
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
                    log_continuous(
                        "wake",
                        "score",
                        log_id="wake.score",
                        elapsed_s=f"{now - started:.1f}",
                        **debug_window.snapshot(
                            threshold, self.config.debug_top_predictions
                        ),
                    )
                    debug_window.reset()
                    next_debug_at = now + debug_interval

            if confidence >= threshold:
                if label == trigger_label:
                    trigger_hits += 1
                else:
                    trigger_label = label
                    trigger_hits = 1
            else:
                trigger_label = ""
                trigger_hits = 0

            if trigger_hits >= trigger_level:
                if detected_debug_log_enabled:
                    log_event(
                        "wake",
                        "detected_debug",
                        log_id="wake.detected_debug",
                        default_enabled=self.config.debug,
                        elapsed_s=f"{time.monotonic() - started:.2f}",
                        label=label,
                        confidence=f"{confidence:.3f}",
                        threshold=f"{threshold:.3f}",
                        trigger_level=trigger_level,
                        trigger_hits=trigger_hits,
                    )
                return WakeEvent(
                    engine="wekws_mha",
                    confidence=confidence,
                    label=label,
                    pcm=model_pcm,
                    sample_rate=audio.sample_rate,
                    duration_ms=_pcm_duration_ms(model_pcm, audio.sample_rate),
                )
            elapsed = time.monotonic() - started
            if self.config.max_wait_seconds > 0 and elapsed >= self.config.max_wait_seconds:
                raise TimeoutError("Timed out waiting for wake word.")

        raise RuntimeError("Audio stream ended while waiting for wake word.")


def create_wake_detector(config: WakeConfig) -> WakeDetector:
    if config.engine == "disabled":
        return DisabledWakeDetector()
    if config.engine == "manual":
        return ManualWakeDetector()
    if config.engine == "openwakeword":
        return OpenWakeWordDetector(config)
    if config.engine == "wekws_mha":
        return WekwsMhaDetector(config)
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
        params = self.snapshot(threshold, top_predictions)
        return " ".join(f"{key}={value}" for key, value in params.items())

    def snapshot(self, threshold: float, top_predictions: int) -> dict[str, int | str]:
        avg_predict_ms = self.predict_total_ms / self.chunks if self.chunks else 0.0
        stats = self.last_stats
        return {
            "chunks": self.chunks,
            "audio_ms": self.audio_ms,
            "rms": f"{stats['rms']:.1f}",
            "peak": int(stats["peak"]),
            "dbfs": f"{stats['dbfs']:.1f}",
            "near_zero_pct": f"{stats['near_zero_pct']:.1f}",
            "clipped_pct": f"{stats['clipped_pct']:.2f}",
            "last": f"{self.last_label}:{self.last_confidence:.3f}",
            "best_window": f"{self.best_label}:{self.best_confidence:.3f}",
            "threshold": f"{threshold:.3f}",
            "top": _format_predictions(self.last_predictions, top_predictions),
            "predict_avg_ms": f"{avg_predict_ms:.1f}",
            "predict_max_ms": f"{self.predict_max_ms:.1f}",
        }


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

    @property
    def size(self) -> int:
        return self._size


def _pcm16_bytes_for_seconds(sample_rate: int, seconds: float) -> int:
    byte_count = max(2, int(sample_rate * 2 * seconds))
    return byte_count - (byte_count % 2)


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
    return " ".join(f"{key}={value}" for key, value in _audio_input_params(audio).items())


def _audio_input_params(audio: AudioInput) -> dict[str, object]:
    config = getattr(audio, "config", None)
    selected_channel = getattr(audio, "selected_channel", "?")
    if config is None:
        return {
            "sample_rate": audio.sample_rate,
            "block_ms": audio.block_ms,
            "selected_channel": selected_channel,
        }
    return {
        "device": getattr(config, "device", None),
        "channels": getattr(config, "channels", "?"),
        "selected_channel": selected_channel,
        "input_gain_db": getattr(config, "input_gain_db", 0.0),
        "sample_rate": audio.sample_rate,
        "block_ms": audio.block_ms,
    }


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
