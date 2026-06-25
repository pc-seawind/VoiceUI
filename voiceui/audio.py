from __future__ import annotations

import array
import math
import re
import sys
import threading
import time
import wave
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from voiceui.logs import is_log_enabled, log_event
from voiceui.models import AudioConfig

_DEFAULT_HOSTAPI = "Windows WASAPI"
_DEVICE_DISPLAY_PATTERN = re.compile(
    r"^(?P<name>.+),\s*(?P<hostapi>.+?)\s*"
    r"\((?P<input>\d+)\s+in,\s*(?P<output>\d+)\s+out\)$"
)


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


class RawAudioRecording:
    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        max_seconds: float | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_bytes = (
            int(sample_rate * channels * 2 * max_seconds)
            if max_seconds is not None and max_seconds > 0
            else 0
        )
        self._chunks: deque[bytes] = deque()
        self._size = 0
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            self._chunks.append(chunk)
            self._size += len(chunk)
            while self.max_bytes > 0 and self._size > self.max_bytes and self._chunks:
                removed = self._chunks.popleft()
                self._size -= len(removed)

    def pcm(self) -> bytes:
        with self._lock:
            return b"".join(self._chunks)

    def duration_ms(self) -> int:
        if self.sample_rate <= 0 or self.channels <= 0:
            return 0
        with self._lock:
            return int(self._size / 2 / self.channels / self.sample_rate * 1000)


class SoundDeviceAudioInput:
    def __init__(self, config: AudioConfig, selected_channel: int = 0):
        self.config = config
        self.selected_channel = selected_channel
        self.sample_rate = config.sample_rate
        self.block_ms = config.block_ms
        self._raw_recordings: list[RawAudioRecording] = []
        self._raw_recordings_lock = threading.Lock()

    def start_raw_recording(self, max_seconds: float | None = None) -> RawAudioRecording:
        recording = RawAudioRecording(
            sample_rate=self.sample_rate,
            channels=self.config.channels,
            max_seconds=max_seconds,
        )
        with self._raw_recordings_lock:
            self._raw_recordings.append(recording)
        return recording

    def stop_raw_recording(self, recording: RawAudioRecording) -> None:
        with self._raw_recordings_lock:
            if recording in self._raw_recordings:
                self._raw_recordings.remove(recording)

    def chunks(self) -> Iterator[bytes]:
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Audio capture requires sounddevice. Install with: pip install -e \".[audio]\""
            ) from exc

        device = resolve_sounddevice_device(sd, self.config.device, kind="input")
        frames = max(1, int(self.config.sample_rate * self.config.block_ms / 1000))
        kwargs = {
            "samplerate": self.config.sample_rate,
            "channels": self.config.channels,
            "dtype": "int16",
            "blocksize": frames,
        }
        if device is not None:
            kwargs["device"] = device

        stream_started = time.monotonic()
        with sd.RawInputStream(**kwargs) as stream:
            if is_log_enabled("audio.stream_opened", default_enabled=self.config.debug):
                latency_ms = int((time.monotonic() - stream_started) * 1000)
                log_event(
                    "audio",
                    "stream_opened",
                    log_id="audio.stream_opened",
                    default_enabled=self.config.debug,
                    device=self.config.device,
                    resolved_device=device,
                    channels=self.config.channels,
                    selected_channel=self.selected_channel,
                    sample_rate=self.sample_rate,
                    block_ms=self.block_ms,
                    latency_ms=latency_ms,
                )
            first_chunk = True
            while True:
                read_started = time.monotonic()
                data, overflowed = stream.read(frames)
                if first_chunk:
                    if is_log_enabled("audio.first_chunk", default_enabled=self.config.debug):
                        read_ms = int((time.monotonic() - read_started) * 1000)
                        log_event(
                            "audio",
                            "first_chunk",
                            log_id="audio.first_chunk",
                            default_enabled=self.config.debug,
                            selected_channel=self.selected_channel,
                            read_ms=read_ms,
                            overflowed=bool(overflowed),
                        )
                    first_chunk = False
                if overflowed:
                    continue
                chunk = bytes(data)
                self._append_raw_recording_chunk(chunk)
                if self.config.channels > 1:
                    chunk = select_pcm16_channel(
                        chunk,
                        channels=self.config.channels,
                        selected_channel=self.selected_channel,
                    )
                chunk = apply_pcm16_gain_db(chunk, self.config.input_gain_db)
                yield chunk

    def _append_raw_recording_chunk(self, chunk: bytes) -> None:
        with self._raw_recordings_lock:
            recordings = list(self._raw_recordings)
        for recording in recordings:
            recording.append(chunk)


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


def resolve_sounddevice_device(
    sd,
    device: str | int | None,
    *,
    kind: str,
    preferred_hostapi: str = _DEFAULT_HOSTAPI,
) -> int | None:
    """Resolve a stable device name/display string to a current sounddevice index."""

    if device is None or device == "default":
        return None
    if isinstance(device, int):
        return device

    requested = str(device).strip()
    if not requested or requested == "default":
        return None
    if requested.isdigit():
        return int(requested)
    if kind not in {"input", "output"}:
        raise ValueError(f"Unsupported sounddevice kind: {kind}")

    requested_name, requested_hostapi = _parse_sounddevice_device_request(requested)
    direction_candidates = [
        candidate
        for candidate in _sounddevice_candidates(sd)
        if candidate[_channel_key(kind)] > 0
    ]
    if requested_hostapi:
        candidates = [
            candidate
            for candidate in direction_candidates
            if _same_device_text(candidate["hostapi_name"], requested_hostapi)
        ]
    else:
        candidates = _prefer_hostapi(direction_candidates, preferred_hostapi)

    matches = [
        candidate
        for candidate in candidates
        if _same_device_text(candidate["name"], requested_name)
        or _same_device_text(candidate["display"], requested)
    ]
    if not matches:
        requested_norm = _normalize_device_text(requested_name)
        matches = [
            candidate
            for candidate in candidates
            if requested_norm and requested_norm in _normalize_device_text(candidate["name"])
        ]

    if len(matches) == 1:
        return int(matches[0]["index"])

    available = "; ".join(candidate["display"] for candidate in candidates) or "<none>"
    if not matches:
        raise RuntimeError(
            f"Could not resolve {kind} audio device {device!r}. "
            f"Available {kind} devices: {available}"
        )
    matched = "; ".join(candidate["display"] for candidate in matches)
    raise RuntimeError(
        f"Audio device {device!r} is ambiguous for {kind}. "
        f"Matched devices: {matched}"
    )


def _parse_sounddevice_device_request(requested: str) -> tuple[str, str | None]:
    match = _DEVICE_DISPLAY_PATTERN.match(requested)
    if match is None:
        return requested, None
    return match.group("name").strip(), match.group("hostapi").strip()


def _sounddevice_candidates(sd) -> list[dict[str, object]]:
    devices = list(sd.query_devices())
    hostapi_names = _sounddevice_hostapi_names(sd)
    candidates: list[dict[str, object]] = []
    for index, info in enumerate(devices):
        name = str(info.get("name") or "").strip()
        if not name:
            continue
        hostapi_index = int(info.get("hostapi", -1))
        hostapi_name = hostapi_names.get(hostapi_index, str(hostapi_index))
        input_channels = int(info.get("max_input_channels") or 0)
        output_channels = int(info.get("max_output_channels") or 0)
        candidates.append(
            {
                "index": index,
                "name": name,
                "hostapi_name": hostapi_name,
                "max_input_channels": input_channels,
                "max_output_channels": output_channels,
                "display": (
                    f"{name}, {hostapi_name} "
                    f"({input_channels} in, {output_channels} out)"
                ),
            }
        )
    return candidates


def _sounddevice_hostapi_names(sd) -> dict[int, str]:
    try:
        hostapis = list(sd.query_hostapis())
    except Exception:
        return {}
    names: dict[int, str] = {}
    for index, info in enumerate(hostapis):
        names[index] = str(info.get("name") or index).strip()
    return names


def _prefer_hostapi(
    candidates: list[dict[str, object]],
    preferred_hostapi: str,
) -> list[dict[str, object]]:
    preferred = [
        candidate
        for candidate in candidates
        if _same_device_text(candidate["hostapi_name"], preferred_hostapi)
    ]
    return preferred or candidates


def _channel_key(kind: str) -> str:
    return "max_input_channels" if kind == "input" else "max_output_channels"


def _same_device_text(left: object, right: object) -> bool:
    return _normalize_device_text(str(left)) == _normalize_device_text(str(right))


def _normalize_device_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


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


def apply_pcm16_gain_db_limited(
    pcm: bytes,
    gain_db: float = 0.0,
    threshold: float = 0.92,
) -> tuple[bytes, float, float]:
    """Apply gain and peak limiting before the final int16 clip.

    Returns (processed_pcm, pre_limiter_peak, limiter_gain).  The peak is
    normalized against int16 full scale after the requested gain, before the
    limiter gain is applied.  Unlike apply_pcm16_gain_db(), this avoids
    clipping first and trying to limit an already flattened waveform later.
    """

    if not pcm:
        return pcm, 0.0, 1.0

    playable_len = len(pcm) - (len(pcm) % 2)
    if playable_len <= 0:
        return pcm, 0.0, 1.0

    samples = array.array("h")
    samples.frombytes(pcm[:playable_len])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return pcm, 0.0, 1.0

    gain_multiplier = math.pow(10.0, gain_db / 20.0) if gain_db else 1.0
    scaled_peak = max((abs(sample * gain_multiplier) for sample in samples), default=0.0)
    if scaled_peak <= 0.0:
        return pcm, 0.0, 1.0

    limit = _normalized_pcm16_limiter_threshold(threshold)
    limit_sample = max(1, int(round(32767 * limit)))
    limiter_gain = min(1.0, limit_sample / scaled_peak)
    total_gain = gain_multiplier * limiter_gain

    if gain_db == 0 and limiter_gain >= 0.999999:
        return pcm, scaled_peak / 32768.0, 1.0

    for index, sample in enumerate(samples):
        processed = int(round(sample * total_gain))
        samples[index] = max(-32768, min(32767, processed))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes() + pcm[playable_len:], scaled_peak / 32768.0, limiter_gain


def _normalized_pcm16_limiter_threshold(threshold: float) -> float:
    return max(0.05, min(1.0, float(threshold)))


def write_pcm16_wav(
    path: str | Path,
    pcm: bytes,
    sample_rate: int,
    channels: int = 1,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(channels)
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
