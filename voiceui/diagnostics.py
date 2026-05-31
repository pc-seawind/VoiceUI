from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from voiceui.audio import create_audio_input, pcm16_rms, write_pcm16_wav
from voiceui.models import AssistantConfig


@dataclass(slots=True)
class RmsSummary:
    chunks: int
    duration_seconds: float
    min: float
    max: float
    mean: float
    p50: float
    p90: float
    p95: float
    recommended_vad_threshold: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=True)


def resolve_audio_channel(config: AssistantConfig, purpose: str, override: int | None = None) -> int:
    if override is not None:
        return override
    if purpose == "wake":
        return config.audio.wake_stream_channel
    if purpose == "command":
        return config.audio.command_stream_channel
    raise ValueError(f"Unsupported audio purpose: {purpose}")


def record_wav(
    config: AssistantConfig,
    output_path: str | Path,
    seconds: float,
    purpose: str = "command",
    channel_override: int | None = None,
) -> Path:
    channel = resolve_audio_channel(config, purpose, channel_override)
    audio = create_audio_input(config.audio, enabled=True, selected_channel=channel)
    pcm = _capture_pcm(audio, seconds)
    path = Path(output_path)
    write_pcm16_wav(path, pcm, sample_rate=audio.sample_rate)
    return path


def calibrate_vad(
    config: AssistantConfig,
    seconds: float,
    purpose: str = "command",
    channel_override: int | None = None,
) -> RmsSummary:
    channel = resolve_audio_channel(config, purpose, channel_override)
    audio = create_audio_input(config.audio, enabled=True, selected_channel=channel)
    rms_values: list[float] = []
    chunks_needed = _chunks_needed(seconds, audio.block_ms)
    chunk_iter = audio.chunks()
    for _ in range(chunks_needed):
        rms_values.append(pcm16_rms(next(chunk_iter)))
    return summarize_rms(rms_values, duration_seconds=chunks_needed * audio.block_ms / 1000)


def summarize_rms(rms_values: list[float], duration_seconds: float) -> RmsSummary:
    if not rms_values:
        return RmsSummary(
            chunks=0,
            duration_seconds=duration_seconds,
            min=0.0,
            max=0.0,
            mean=0.0,
            p50=0.0,
            p90=0.0,
            p95=0.0,
            recommended_vad_threshold=450,
        )

    sorted_values = sorted(rms_values)
    p50 = _percentile(sorted_values, 50)
    p90 = _percentile(sorted_values, 90)
    p95 = _percentile(sorted_values, 95)
    mean = sum(sorted_values) / len(sorted_values)
    recommended = max(300, math.ceil(p95 * 1.8), math.ceil(mean * 2.5))
    return RmsSummary(
        chunks=len(sorted_values),
        duration_seconds=duration_seconds,
        min=sorted_values[0],
        max=sorted_values[-1],
        mean=mean,
        p50=p50,
        p90=p90,
        p95=p95,
        recommended_vad_threshold=recommended,
    )


def _capture_pcm(audio, seconds: float) -> bytes:
    chunks_needed = _chunks_needed(seconds, audio.block_ms)
    chunks: list[bytes] = []
    chunk_iter = audio.chunks()
    for _ in range(chunks_needed):
        chunks.append(next(chunk_iter))
    return b"".join(chunks)


def _chunks_needed(seconds: float, block_ms: int) -> int:
    if seconds <= 0:
        raise ValueError("seconds must be greater than 0")
    return max(1, math.ceil(seconds * 1000 / block_ms))


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
