from __future__ import annotations

from collections import deque

from voiceui.audio import AudioInput, pcm16_rms
from voiceui.models import Utterance, VadConfig


class VadRecorder:
    def record(self, audio: AudioInput) -> Utterance:
        raise NotImplementedError


class EnergyVadRecorder(VadRecorder):
    def __init__(self, config: VadConfig):
        self.config = config

    def record(self, audio: AudioInput) -> Utterance:
        chunk_ms = audio.block_ms
        pre_roll_chunks = max(1, self.config.pre_roll_ms // chunk_ms)
        min_speech_chunks = max(1, self.config.min_speech_ms // chunk_ms)
        silence_chunks = max(1, self.config.silence_ms // chunk_ms)
        max_chunks = max(1, self.config.max_speech_ms // chunk_ms)

        pre_roll: deque[bytes] = deque(maxlen=pre_roll_chunks)
        recorded: list[bytes] = []
        speech_chunks = 0
        trailing_silence = 0
        is_recording = False

        for chunk in audio.chunks():
            rms = pcm16_rms(chunk)
            speech = rms >= self.config.threshold

            if not is_recording:
                pre_roll.append(chunk)
                if speech:
                    speech_chunks += 1
                else:
                    speech_chunks = 0

                if speech_chunks >= min_speech_chunks:
                    is_recording = True
                    recorded.extend(pre_roll)
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


def create_vad_recorder(config: VadConfig) -> VadRecorder:
    if config.engine == "energy":
        return EnergyVadRecorder(config)
    if config.engine == "silero":
        return SileroVadRecorder(config)
    raise ValueError(f"Unsupported VAD engine: {config.engine}")
