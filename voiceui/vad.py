from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from voiceui.audio import AudioInput, pcm16_rms
from voiceui.models import Utterance, VadConfig


class SpeechStartTimeoutError(TimeoutError):
    pass


class VadRecorder:
    def record(self, audio: AudioInput, start_timeout_seconds: float = 0.0) -> Utterance:
        raise NotImplementedError


class EnergyVadRecorder(VadRecorder):
    def __init__(self, config: VadConfig):
        self.config = config

    def record(self, audio: AudioInput, start_timeout_seconds: float = 0.0) -> Utterance:
        chunk_ms = audio.block_ms
        pre_roll_chunks = max(1, self.config.pre_roll_ms // chunk_ms)
        min_speech_chunks = max(1, self.config.min_speech_ms // chunk_ms)
        silence_chunks = max(1, self.config.silence_ms // chunk_ms)
        max_chunks = max(1, self.config.max_speech_ms // chunk_ms)
        start_timeout_ms = max(0, int(start_timeout_seconds * 1000))

        pre_roll: deque[bytes] = deque(maxlen=pre_roll_chunks)
        recorded: list[bytes] = []
        speech_chunks = 0
        trailing_silence = 0
        is_recording = False
        waited_ms = 0

        for chunk in audio.chunks():
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
                elif start_timeout_ms and waited_ms >= start_timeout_ms and speech_chunks == 0:
                    raise SpeechStartTimeoutError("Timed out waiting for speech.")
                continue

            recorded.append(chunk)
            if speech:
                trailing_silence = 0
            else:
                trailing_silence += 1

            if trailing_silence >= silence_chunks or len(recorded) >= max_chunks:
                duration_ms = len(recorded) * chunk_ms
                return Utterance(
                    pcm=b"".join(recorded),
                    sample_rate=audio.sample_rate,
                    duration_ms=duration_ms,
                )

        raise RuntimeError("Audio stream ended during VAD recording.")


class SileroVadRecorder(EnergyVadRecorder):
    def __init__(self, config: VadConfig):
        super().__init__(config)


class WebRtcVadRecorder(VadRecorder):
    _VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}
    _VALID_FRAME_MS = {10, 20, 30}

    def __init__(self, config: VadConfig):
        self.config = config
        if config.frame_ms not in self._VALID_FRAME_MS:
            raise ValueError("webrtc VAD frame_ms must be one of 10, 20, or 30.")
        if config.webrtc_mode < 0 or config.webrtc_mode > 3:
            raise ValueError("webrtc VAD mode must be between 0 and 3.")

    def record(self, audio: AudioInput, start_timeout_seconds: float = 0.0) -> Utterance:
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
        silence_frames = max(1, self.config.silence_ms // frame_ms)
        max_frames = max(1, self.config.max_speech_ms // frame_ms)
        start_timeout_ms = max(0, int(start_timeout_seconds * 1000))

        pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        recorded: list[bytes] = []
        speech_frames = 0
        trailing_silence = 0
        is_recording = False
        waited_ms = 0

        for frame in _pcm16_frames(audio, frame_ms=frame_ms):
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
                elif start_timeout_ms and waited_ms >= start_timeout_ms and speech_frames == 0:
                    raise SpeechStartTimeoutError("Timed out waiting for speech.")
                continue

            recorded.append(frame)
            if speech:
                trailing_silence = 0
            else:
                trailing_silence += 1

            if trailing_silence >= silence_frames or len(recorded) >= max_frames:
                return Utterance(
                    pcm=b"".join(recorded),
                    sample_rate=audio.sample_rate,
                    duration_ms=len(recorded) * frame_ms,
                )

        raise RuntimeError("Audio stream ended during VAD recording.")


def create_vad_recorder(config: VadConfig) -> VadRecorder:
    if config.engine == "energy":
        return EnergyVadRecorder(config)
    if config.engine == "silero":
        return SileroVadRecorder(config)
    if config.engine == "webrtc":
        return WebRtcVadRecorder(config)
    raise ValueError(f"Unsupported VAD engine: {config.engine}")


def _pcm16_frames(audio: AudioInput, frame_ms: int) -> Iterator[bytes]:
    frame_bytes = int(audio.sample_rate * frame_ms / 1000) * 2
    buffer = bytearray()
    for chunk in audio.chunks():
        buffer.extend(chunk)
        while len(buffer) >= frame_bytes:
            yield bytes(buffer[:frame_bytes])
            del buffer[:frame_bytes]
