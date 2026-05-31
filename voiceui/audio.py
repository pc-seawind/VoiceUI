from __future__ import annotations

import math
import wave
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

        with sd.RawInputStream(**kwargs) as stream:
            while True:
                data, overflowed = stream.read(frames)
                if overflowed:
                    continue
                chunk = bytes(data)
                if self.config.channels > 1:
                    chunk = select_pcm16_channel(
                        chunk,
                        channels=self.config.channels,
                        selected_channel=self.selected_channel,
                    )
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
