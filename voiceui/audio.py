from __future__ import annotations

import math
import time
import wave
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from voiceui.models import AudioConfig


class AudioInput(Protocol):
    sample_rate: int
    block_ms: int

    def chunks(self) -> Iterator[bytes]:
        """Yield little-endian signed 16-bit PCM chunks."""


class NullAudioInput:
    sample_rate = 16000
    block_ms = 80

    def chunks(self) -> Iterator[bytes]:
        raise RuntimeError("Audio input is not configured. Use --text or input.mode=text.")


class RecordingAudioInput:
    def __init__(self, audio: AudioInput, max_seconds: float | None = None):
        self.audio = audio
        self.config = getattr(audio, "config", None)
        self.selected_channel = getattr(audio, "selected_channel", "?")
        self.sample_rate = audio.sample_rate
        self.block_ms = audio.block_ms
        self.max_bytes = (
            int(self.sample_rate * 2 * max_seconds)
            if max_seconds is not None and max_seconds > 0
            else 0
        )
        self._chunks: deque[bytes] = deque()
        self._size = 0

    def chunks(self) -> Iterator[bytes]:
        for chunk in self.audio.chunks():
            self._append(chunk)
            yield chunk

    def pcm(self) -> bytes:
        return b"".join(self._chunks)

    def duration_ms(self) -> int:
        if self.sample_rate <= 0:
            return 0
        return int(self._size / 2 / self.sample_rate * 1000)

    def _append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._chunks.append(chunk)
        self._size += len(chunk)
        while self.max_bytes > 0 and self._size > self.max_bytes and self._chunks:
            removed = self._chunks.popleft()
            self._size -= len(removed)


class SoundDeviceAudioInput:
    def __init__(self, config: AudioConfig, selected_channel: int = 0):
        self.config = config
        self.selected_channel = selected_channel
        self.sample_rate = config.sample_rate
        self.block_ms = config.block_ms

    def chunks(self) -> Iterator[bytes]:
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Audio capture requires sounddevice. Install with: pip install -e \".[audio]\""
            ) from exc

        frames = max(1, int(self.config.sample_rate * self.config.block_ms / 1000))
        kwargs = {
            "samplerate": self.config.sample_rate,
            "channels": self.config.channels,
            "dtype": "int16",
            "blocksize": frames,
        }
        if self.config.device not in (None, "default"):
            kwargs["device"] = self.config.device

        stream_started = time.monotonic()
        with sd.RawInputStream(**kwargs) as stream:
            if self.config.debug:
                latency_ms = int((time.monotonic() - stream_started) * 1000)
                print(
                    "audio_debug> stream_opened "
                    f"device={self.config.device} channels={self.config.channels} "
                    f"selected_channel={self.selected_channel} "
                    f"sample_rate={self.sample_rate} block_ms={self.block_ms} "
                    f"latency_ms={latency_ms}"
                )
            first_chunk = True
            while True:
                read_started = time.monotonic()
                data, overflowed = stream.read(frames)
                if self.config.debug and first_chunk:
                    read_ms = int((time.monotonic() - read_started) * 1000)
                    print(
                        "audio_debug> first_chunk "
                        f"selected_channel={self.selected_channel} read_ms={read_ms} "
                        f"overflowed={bool(overflowed)}"
                    )
                    first_chunk = False
                if overflowed:
                    continue
                chunk = bytes(data)
                if self.config.channels > 1:
                    chunk = select_pcm16_channel(
                        chunk,
                        channels=self.config.channels,
                        selected_channel=self.selected_channel,
                    )
                chunk = apply_pcm16_gain_db(chunk, self.config.input_gain_db)
                yield chunk


def create_audio_input(
    config: AudioConfig,
    enabled: bool,
    selected_channel: int = 0,
) -> AudioInput:
    if not enabled:
        return NullAudioInput()
    return SoundDeviceAudioInput(config, selected_channel=selected_channel)


def list_audio_devices() -> str:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Audio device listing requires sounddevice. Install with: pip install -e \".[audio]\""
        ) from exc

    return str(sd.query_devices())


def pcm16_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    sample_count = len(pcm) // 2
    if sample_count == 0:
        return 0.0
    total = 0
    for index in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        total += sample * sample
    return math.sqrt(total / sample_count)


def apply_pcm16_gain_db(pcm: bytes, gain_db: float) -> bytes:
    if not pcm or gain_db == 0:
        return pcm

    multiplier = math.pow(10.0, gain_db / 20.0)
    output = bytearray(len(pcm))
    for index in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        amplified = int(round(sample * multiplier))
        clipped = min(32767, max(-32768, amplified))
        output[index : index + 2] = clipped.to_bytes(2, "little", signed=True)
    if len(pcm) % 2:
        output[-1] = pcm[-1]
    return bytes(output)


def write_pcm16_wav(path: str | Path, pcm: bytes, sample_rate: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def read_pcm16_wav(path: str | Path, selected_channel: int = 0) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample_width={sample_width}")
    if channels > 1:
        frames = select_pcm16_channel(frames, channels=channels, selected_channel=selected_channel)
    return frames, sample_rate


def select_pcm16_channel(pcm: bytes, channels: int, selected_channel: int) -> bytes:
    if channels <= 1:
        return pcm
    if selected_channel < 0 or selected_channel >= channels:
        raise ValueError(
            f"selected_channel={selected_channel} is outside available channels={channels}"
        )

    frame_width = channels * 2
    output = bytearray(len(pcm) // channels)
    write_index = 0
    for frame_index in range(0, len(pcm) - frame_width + 1, frame_width):
        sample_index = frame_index + selected_channel * 2
        output[write_index : write_index + 2] = pcm[sample_index : sample_index + 2]
        write_index += 2
    return bytes(output[:write_index])
