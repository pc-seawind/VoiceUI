from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterator

from voiceui.audio import AudioInput, pcm16_rms
from voiceui.models import Utterance, VadConfig


class SpeechStartTimeoutError(TimeoutError):
    pass


class VadRecorder:
    def warm_up(self) -> bool:
        return False

    def record(
        self,
        audio: AudioInput,
        start_timeout_seconds: float = 0.0,
        stop_event: threading.Event | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_audio: Callable[[bytes], None] | None = None,
    ) -> Utterance:
        raise NotImplementedError


class EnergyVadRecorder(VadRecorder):
    def __init__(self, config: VadConfig):
        self.config = config

    def record(
        self,
        audio: AudioInput,
        start_timeout_seconds: float = 0.0,
        stop_event: threading.Event | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_audio: Callable[[bytes], None] | None = None,
    ) -> Utterance:
        chunk_ms = audio.block_ms
        pre_roll_chunks = max(1, self.config.pre_roll_ms // chunk_ms)
        min_speech_chunks = max(1, self.config.min_speech_ms // chunk_ms)
        start_buffer_chunks = pre_roll_chunks + min_speech_chunks
        silence_chunks = max(1, self.config.silence_ms // chunk_ms)
        max_chunks = max(1, self.config.max_speech_ms // chunk_ms)
        start_timeout_ms = max(0, int(start_timeout_seconds * 1000))

        pre_roll: deque[bytes] = deque(maxlen=start_buffer_chunks)
        recorded: list[bytes] = []
        speech_chunks = 0
        trailing_silence = 0
        is_recording = False
        waited_ms = 0

        for chunk in audio.chunks():
            if _stop_requested(stop_event):
                raise SpeechStartTimeoutError("Stopped waiting for speech.")
            rms = pcm16_rms(chunk)
            speech = rms >= self.config.threshold
            if not is_recording:
                waited_ms += chunk_ms

            if not is_recording:
                pre_roll.append(chunk)
                if speech:
                    speech_chunks += 1
                else:
                    speech_chunks = 0

                if speech_chunks >= min_speech_chunks:
                    is_recording = True
                    recorded.extend(pre_roll)
                    _print_vad_start(
                        self.config,
                        engine="energy",
                        waited_ms=waited_ms,
                        buffered_ms=len(pre_roll) * chunk_ms,
                        score=rms,
                        threshold=self.config.threshold,
                    )
                    _notify_speech_start(on_speech_start)
                    _notify_speech_audio(on_speech_audio, b"".join(pre_roll))
                elif start_timeout_ms and waited_ms >= start_timeout_ms and speech_chunks == 0:
                    raise SpeechStartTimeoutError("Timed out waiting for speech.")
                continue

            recorded.append(chunk)
            _notify_speech_audio(on_speech_audio, chunk)
            if speech:
                trailing_silence = 0
            else:
                trailing_silence += 1

            if trailing_silence >= silence_chunks or len(recorded) >= max_chunks:
                reason = "max_duration" if len(recorded) >= max_chunks else "silence"
                utterance, trimmed_ms = _build_utterance(
                    recorded,
                    sample_rate=audio.sample_rate,
                    frame_duration_ms=chunk_ms,
                    trailing_silence_ms=trailing_silence * chunk_ms,
                    trim_ms=self.config.trailing_silence_trim_ms,
                )
                _print_vad_stop(
                    self.config,
                    engine="energy",
                    reason=reason,
                    duration_ms=utterance.duration_ms,
                    trailing_silence_ms=trailing_silence * chunk_ms,
                    trimmed_silence_ms=trimmed_ms,
                )
                return utterance

        raise RuntimeError("Audio stream ended during VAD recording.")


class SileroVadRecorder(VadRecorder):
    _VALID_SAMPLE_RATES = {8000, 16000}

    def __init__(self, config: VadConfig):
        self.config = config
        self._model = None
        self._torch = None

    def warm_up(self) -> bool:
        self._load_model()
        return True

    def record(
        self,
        audio: AudioInput,
        start_timeout_seconds: float = 0.0,
        stop_event: threading.Event | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_audio: Callable[[bytes], None] | None = None,
    ) -> Utterance:
        if audio.sample_rate not in self._VALID_SAMPLE_RATES:
            raise RuntimeError(
                "silero VAD requires sample_rate to be 8000 or 16000, "
                f"got {audio.sample_rate}"
            )

        model, torch = self._load_model()
        if hasattr(model, "reset_states"):
            model.reset_states()

        window_samples = 512 if audio.sample_rate == 16000 else 256
        window_ms = int(window_samples * 1000 / audio.sample_rate)
        threshold = float(self.config.threshold)
        neg_threshold = max(0.01, threshold - 0.15)
        pre_roll_frames = max(1, self.config.pre_roll_ms // window_ms)
        min_speech_frames = max(1, self.config.min_speech_ms // window_ms)
        start_buffer_frames = pre_roll_frames + min_speech_frames
        silence_frames = max(1, self.config.silence_ms // window_ms)
        max_frames = max(1, self.config.max_speech_ms // window_ms)
        start_timeout_ms = max(0, int(start_timeout_seconds * 1000))

        pre_roll: deque[bytes] = deque(maxlen=start_buffer_frames)
        recorded: list[bytes] = []
        speech_frames = 0
        trailing_silence = 0
        is_recording = False
        waited_ms = 0

        for frame in _pcm16_sample_windows(audio, window_samples=window_samples):
            if _stop_requested(stop_event):
                raise SpeechStartTimeoutError("Stopped waiting for speech.")
            probability = _silero_speech_probability(model, torch, frame, audio.sample_rate)
            if not is_recording:
                waited_ms += window_ms

            if not is_recording:
                pre_roll.append(frame)
                if probability >= threshold:
                    speech_frames += 1
                else:
                    speech_frames = 0

                if speech_frames >= min_speech_frames:
                    is_recording = True
                    recorded.extend(pre_roll)
                    _print_vad_start(
                        self.config,
                        engine="silero",
                        waited_ms=waited_ms,
                        buffered_ms=len(pre_roll) * window_ms,
                        score=probability,
                        threshold=threshold,
                    )
                    _notify_speech_start(on_speech_start)
                    _notify_speech_audio(on_speech_audio, b"".join(pre_roll))
                elif start_timeout_ms and waited_ms >= start_timeout_ms and speech_frames == 0:
                    raise SpeechStartTimeoutError("Timed out waiting for speech.")
                continue

            recorded.append(frame)
            _notify_speech_audio(on_speech_audio, frame)
            if probability >= neg_threshold:
                trailing_silence = 0
            else:
                trailing_silence += 1

            if trailing_silence >= silence_frames or len(recorded) >= max_frames:
                reason = "max_duration" if len(recorded) >= max_frames else "silence"
                utterance, trimmed_ms = _build_utterance(
                    recorded,
                    sample_rate=audio.sample_rate,
                    frame_duration_ms=window_ms,
                    trailing_silence_ms=trailing_silence * window_ms,
                    trim_ms=self.config.trailing_silence_trim_ms,
                )
                _print_vad_stop(
                    self.config,
                    engine="silero",
                    reason=reason,
                    duration_ms=utterance.duration_ms,
                    trailing_silence_ms=trailing_silence * window_ms,
                    trimmed_silence_ms=trimmed_ms,
                )
                return utterance

        raise RuntimeError("Audio stream ended during VAD recording.")

    def _load_model(self):
        if self._model is not None and self._torch is not None:
            return self._model, self._torch

        try:
            import torch  # type: ignore[import-untyped]
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "silero VAD requires silero-vad. Install with: "
                "pip install -e \".[demo]\" or pip install -e \".[vad]\""
            ) from exc

        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(1)
        self._model = load_silero_vad()
        self._torch = torch
        return self._model, self._torch


class WebRtcVadRecorder(VadRecorder):
    _VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}
    _VALID_FRAME_MS = {10, 20, 30}

    def __init__(self, config: VadConfig):
        self.config = config
        if config.frame_ms not in self._VALID_FRAME_MS:
            raise ValueError("webrtc VAD frame_ms must be one of 10, 20, or 30.")
        if config.webrtc_mode < 0 or config.webrtc_mode > 3:
            raise ValueError("webrtc VAD mode must be between 0 and 3.")

    def record(
        self,
        audio: AudioInput,
        start_timeout_seconds: float = 0.0,
        stop_event: threading.Event | None = None,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_audio: Callable[[bytes], None] | None = None,
    ) -> Utterance:
        if audio.sample_rate not in self._VALID_SAMPLE_RATES:
            raise RuntimeError(
                "webrtc VAD requires sample_rate to be one of "
                f"{sorted(self._VALID_SAMPLE_RATES)}, got {audio.sample_rate}"
            )

        try:
            import webrtcvad  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "webrtc VAD requires webrtcvad-wheels. Install with: pip install -e \".[vad]\""
            ) from exc

        detector = webrtcvad.Vad(self.config.webrtc_mode)
        frame_ms = self.config.frame_ms
        pre_roll_frames = max(1, self.config.pre_roll_ms // frame_ms)
        min_speech_frames = max(1, self.config.min_speech_ms // frame_ms)
        start_buffer_frames = pre_roll_frames + min_speech_frames
        silence_frames = max(1, self.config.silence_ms // frame_ms)
        max_frames = max(1, self.config.max_speech_ms // frame_ms)
        start_timeout_ms = max(0, int(start_timeout_seconds * 1000))

        pre_roll: deque[bytes] = deque(maxlen=start_buffer_frames)
        recorded: list[bytes] = []
        speech_frames = 0
        trailing_silence = 0
        is_recording = False
        waited_ms = 0

        for frame in _pcm16_frames(audio, frame_ms=frame_ms):
            if _stop_requested(stop_event):
                raise SpeechStartTimeoutError("Stopped waiting for speech.")
            speech = detector.is_speech(frame, audio.sample_rate)
            if not is_recording:
                waited_ms += frame_ms

            if not is_recording:
                pre_roll.append(frame)
                if speech:
                    speech_frames += 1
                else:
                    speech_frames = 0

                if speech_frames >= min_speech_frames:
                    is_recording = True
                    recorded.extend(pre_roll)
                    _print_vad_start(
                        self.config,
                        engine="webrtc",
                        waited_ms=waited_ms,
                        buffered_ms=len(pre_roll) * frame_ms,
                        score=1.0,
                        threshold=float(self.config.webrtc_mode),
                    )
                    _notify_speech_start(on_speech_start)
                    _notify_speech_audio(on_speech_audio, b"".join(pre_roll))
                elif start_timeout_ms and waited_ms >= start_timeout_ms and speech_frames == 0:
                    raise SpeechStartTimeoutError("Timed out waiting for speech.")
                continue

            recorded.append(frame)
            _notify_speech_audio(on_speech_audio, frame)
            if speech:
                trailing_silence = 0
            else:
                trailing_silence += 1

            if trailing_silence >= silence_frames or len(recorded) >= max_frames:
                reason = "max_duration" if len(recorded) >= max_frames else "silence"
                utterance, trimmed_ms = _build_utterance(
                    recorded,
                    sample_rate=audio.sample_rate,
                    frame_duration_ms=frame_ms,
                    trailing_silence_ms=trailing_silence * frame_ms,
                    trim_ms=self.config.trailing_silence_trim_ms,
                )
                _print_vad_stop(
                    self.config,
                    engine="webrtc",
                    reason=reason,
                    duration_ms=utterance.duration_ms,
                    trailing_silence_ms=trailing_silence * frame_ms,
                    trimmed_silence_ms=trimmed_ms,
                )
                return utterance

        raise RuntimeError("Audio stream ended during VAD recording.")


def create_vad_recorder(config: VadConfig) -> VadRecorder:
    if config.engine == "energy":
        return EnergyVadRecorder(config)
    if config.engine == "silero":
        return SileroVadRecorder(config)
    if config.engine == "webrtc":
        return WebRtcVadRecorder(config)
    raise ValueError(f"Unsupported VAD engine: {config.engine}")


def _build_utterance(
    recorded: list[bytes],
    *,
    sample_rate: int,
    frame_duration_ms: int,
    trailing_silence_ms: int,
    trim_ms: int,
) -> tuple[Utterance, int]:
    pcm = b"".join(recorded)
    duration_ms = len(recorded) * frame_duration_ms
    trimmed_ms = min(max(0, trim_ms), max(0, trailing_silence_ms))
    if trimmed_ms > 0 and sample_rate > 0:
        trim_bytes = int(sample_rate * trimmed_ms / 1000) * 2
        trim_bytes -= trim_bytes % 2
        if trim_bytes > 0:
            trim_bytes = min(trim_bytes, max(0, len(pcm) - 2))
            pcm = pcm[: len(pcm) - trim_bytes]
            duration_ms = int(len(pcm) / 2 / sample_rate * 1000)
            trimmed_ms = int(trim_bytes / 2 / sample_rate * 1000)
    return Utterance(pcm=pcm, sample_rate=sample_rate, duration_ms=duration_ms), trimmed_ms


def _pcm16_frames(audio: AudioInput, frame_ms: int) -> Iterator[bytes]:
    frame_bytes = int(audio.sample_rate * frame_ms / 1000) * 2
    yield from _pcm16_windows(audio, frame_bytes=frame_bytes)


def _pcm16_sample_windows(audio: AudioInput, window_samples: int) -> Iterator[bytes]:
    yield from _pcm16_windows(audio, frame_bytes=window_samples * 2)


def _pcm16_windows(audio: AudioInput, frame_bytes: int) -> Iterator[bytes]:
    buffer = bytearray()
    for chunk in audio.chunks():
        buffer.extend(chunk)
        while len(buffer) >= frame_bytes:
            yield bytes(buffer[:frame_bytes])
            del buffer[:frame_bytes]


def _silero_speech_probability(model, torch, pcm: bytes, sample_rate: int) -> float:
    samples = [
        int.from_bytes(pcm[index : index + 2], "little", signed=True) / 32768.0
        for index in range(0, len(pcm) - 1, 2)
    ]
    dtype = getattr(torch, "float32", None)
    tensor = torch.tensor(samples, dtype=dtype) if dtype is not None else torch.tensor(samples)
    prediction = model(tensor, sample_rate)
    if hasattr(prediction, "item"):
        return float(prediction.item())
    return float(prediction)


def _notify_speech_start(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _notify_speech_audio(callback: Callable[[bytes], None] | None, pcm: bytes) -> None:
    if callback is not None and pcm:
        callback(pcm)


def _stop_requested(stop_event: threading.Event | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


def _print_vad_start(
    config: VadConfig,
    *,
    engine: str,
    waited_ms: int,
    buffered_ms: int,
    score: float,
    threshold: float,
) -> None:
    if not config.debug:
        return
    effective_start_ms = max(0, waited_ms - buffered_ms)
    print(
        "vad_debug> start "
        f"engine={engine} waited_ms={waited_ms} buffered_ms={buffered_ms} "
        f"effective_start_ms={effective_start_ms} pre_roll_ms={config.pre_roll_ms} "
        f"min_speech_ms={config.min_speech_ms} score={score:.3f} threshold={threshold:.3f}"
    )


def _print_vad_stop(
    config: VadConfig,
    *,
    engine: str,
    reason: str,
    duration_ms: int,
    trailing_silence_ms: int,
    trimmed_silence_ms: int,
) -> None:
    if not config.debug:
        return
    print(
        "vad_debug> stop "
        f"engine={engine} reason={reason} duration_ms={duration_ms} "
        f"trailing_silence_ms={trailing_silence_ms} "
        f"trimmed_silence_ms={trimmed_silence_ms}"
    )
