#!/usr/bin/env python3
"""Linux XVF3800 playback/record loopback diagnostic.

This utility intentionally records the raw mic channel (usually ch1) while playing
through the configured output device. The real assistant path should stay on the
AEC/EC channel (usually ch0); ch1 is only for loopback validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop
from dataclasses import asdict, dataclass
from pathlib import Path


def _ensure_local_portaudio() -> None:
    """Re-exec with the repo-local PortAudio library if sounddevice needs it."""

    repo_root = Path(__file__).resolve().parents[1]
    local_lib = repo_root / ".venv/local/portaudio/usr/lib/x86_64-linux-gnu"
    if not local_lib.exists():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(":") if entry]
    if str(local_lib) in entries:
        return
    os.environ["LD_LIBRARY_PATH"] = f"{local_lib}:{current}" if current else str(local_lib)
    os.execv(sys.executable, [sys.executable, *sys.argv])


_ensure_local_portaudio()

import sounddevice as sd  # type: ignore[import-untyped]  # noqa: E402

from voiceui.audio import (  # noqa: E402
    pcm16_rms,
    resolve_sounddevice_device,
    select_pcm16_channel,
    write_pcm16_wav,
)
from voiceui.config import load_config  # noqa: E402
from voiceui.models import Utterance  # noqa: E402
from voiceui.stt import create_stt  # noqa: E402
from voiceui.tts import create_tts  # noqa: E402
from voiceui.wake_ack import _convert_pcm16_channels, _resample_pcm16  # noqa: E402


@dataclass(slots=True)
class ChannelSummary:
    channel: int
    label: str
    wav_path: str
    rms: float
    max_100ms_rms: float
    peak: int
    asr_text: str = ""
    asr_latency_ms: int = 0
    asr_error: str = ""


@dataclass(slots=True)
class LoopbackSummary:
    config: str
    text: str
    output_dir: str
    input_device_index: int | None
    output_device_index: int | None
    formal_command_channel: int
    loopback_channel: int
    playback_ref_wav: str
    record_stereo_wav: str
    record_duration_ms: int
    channels: list[ChannelSummary]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="auto", help="VoiceUI config path or auto")
    parser.add_argument("--text", default="回环测试，音量已经降低。")
    parser.add_argument("--output-dir", default="/tmp/voiceui-loopback-diag")
    parser.add_argument(
        "--playback-gain",
        type=float,
        default=0.25,
        help="Digital playback gain multiplier. Default 0.25 (-12 dB).",
    )
    parser.add_argument("--record-seconds", type=float, default=3.0)
    parser.add_argument("--loopback-channel", type=int, default=1)
    parser.add_argument("--no-asr", action="store_true", help="Skip Aliyun ASR validation")
    args = parser.parse_args(argv)

    summary = run_loopback_diag(
        config_path=args.config,
        text=args.text,
        output_dir=Path(args.output_dir),
        playback_gain=args.playback_gain,
        record_seconds=args.record_seconds,
        loopback_channel=args.loopback_channel,
        run_asr=not args.no_asr,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


def run_loopback_diag(
    *,
    config_path: str,
    text: str,
    output_dir: Path,
    playback_gain: float,
    record_seconds: float,
    loopback_channel: int,
    run_asr: bool,
) -> LoopbackSummary:
    cfg = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts = create_tts(cfg.tts)
    synthesized = tts.synthesize(text)
    mono_16k, _resampler = _resample_pcm16(
        synthesized.data,
        source_rate=cfg.tts.sample_rate,
        target_rate=cfg.tts.playback_sample_rate or cfg.audio.sample_rate,
        channels=1,
    )
    playback_rate = cfg.tts.playback_sample_rate or cfg.audio.sample_rate
    playback_channels = cfg.tts.playback_channels or cfg.audio.channels
    playback_pcm = _convert_pcm16_channels(
        mono_16k,
        source_channels=1,
        target_channels=playback_channels,
    )
    playback_pcm = _scale_pcm16(playback_pcm, playback_gain)

    playback_ref = output_dir / "playback_ref.wav"
    write_pcm16_wav(playback_ref, playback_pcm, playback_rate, channels=playback_channels)

    input_device = resolve_sounddevice_device(sd, cfg.audio.device, kind="input")
    output_device = resolve_sounddevice_device(sd, cfg.tts.playback_device, kind="output")
    recorded = _play_and_record(
        playback_pcm=playback_pcm,
        playback_rate=playback_rate,
        playback_channels=playback_channels,
        input_device=input_device,
        output_device=output_device,
        input_channels=cfg.audio.channels,
        record_seconds=record_seconds,
    )

    record_stereo = output_dir / "record_stereo.wav"
    write_pcm16_wav(record_stereo, recorded, cfg.audio.sample_rate, channels=cfg.audio.channels)

    channel_summaries: list[ChannelSummary] = []
    stt = create_stt(cfg.stt) if run_asr else None
    labels = {cfg.audio.command_stream_channel: "formal_ec", loopback_channel: "loopback_raw"}
    for channel in sorted(set(labels)):
        pcm = select_pcm16_channel(recorded, channels=cfg.audio.channels, selected_channel=channel)
        path = output_dir / f"record_ch{channel}_{labels[channel]}.wav"
        write_pcm16_wav(path, pcm, cfg.audio.sample_rate)
        channel_summary = ChannelSummary(
            channel=channel,
            label=labels[channel],
            wav_path=str(path),
            rms=round(pcm16_rms(pcm), 1),
            max_100ms_rms=round(_max_window_rms(pcm, cfg.audio.sample_rate, 100), 1),
            peak=audioop.max(pcm, 2) if pcm else 0,
        )
        if stt is not None and channel == loopback_channel:
            _fill_asr_summary(channel_summary, stt, pcm, cfg.audio.sample_rate)
        channel_summaries.append(channel_summary)

    return LoopbackSummary(
        config=config_path,
        text=text,
        output_dir=str(output_dir),
        input_device_index=input_device,
        output_device_index=output_device,
        formal_command_channel=cfg.audio.command_stream_channel,
        loopback_channel=loopback_channel,
        playback_ref_wav=str(playback_ref),
        record_stereo_wav=str(record_stereo),
        record_duration_ms=int(
            len(recorded) / 2 / cfg.audio.channels / cfg.audio.sample_rate * 1000
        ),
        channels=channel_summaries,
    )


def _play_and_record(
    *,
    playback_pcm: bytes,
    playback_rate: int,
    playback_channels: int,
    input_device: int | None,
    output_device: int | None,
    input_channels: int,
    record_seconds: float,
) -> bytes:
    block_frames = max(1, playback_rate // 20)
    recorded = bytearray()

    def callback(indata, _frames_count, _time_info, status) -> None:
        if status:
            print(f"input status: {status}", file=sys.stderr)
        recorded.extend(bytes(indata))

    playback_seconds = len(playback_pcm) / 2 / playback_channels / playback_rate
    seconds = max(record_seconds, playback_seconds + 0.8)
    with sd.RawInputStream(
        samplerate=playback_rate,
        channels=input_channels,
        dtype="int16",
        blocksize=block_frames,
        device=input_device,
        callback=callback,
    ):
        time.sleep(0.35)
        with sd.RawOutputStream(
            samplerate=playback_rate,
            channels=playback_channels,
            dtype="int16",
            blocksize=block_frames,
            device=output_device,
        ) as stream:
            frame_bytes = block_frames * playback_channels * 2
            for offset in range(0, len(playback_pcm), frame_bytes):
                stream.write(playback_pcm[offset : offset + frame_bytes])
        time.sleep(max(0.2, seconds - 0.35 - playback_seconds))
    return bytes(recorded)


def _scale_pcm16(pcm: bytes, gain: float) -> bytes:
    gain = max(0.0, min(gain, 1.0))
    if gain == 1.0:
        return pcm
    output = bytearray(len(pcm))
    for index in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        scaled = int(round(sample * gain))
        output[index : index + 2] = scaled.to_bytes(2, "little", signed=True)
    if len(pcm) % 2:
        output[-1] = pcm[-1]
    return bytes(output)


def _max_window_rms(pcm: bytes, sample_rate: int, window_ms: int) -> float:
    window_bytes = max(2, int(sample_rate * 2 * window_ms / 1000))
    hop = max(2, window_bytes // 2)
    best = 0.0
    for offset in range(0, max(1, len(pcm) - window_bytes + 1), hop):
        best = max(best, pcm16_rms(pcm[offset : offset + window_bytes]))
    return best


def _fill_asr_summary(channel: ChannelSummary, stt, pcm: bytes, sample_rate: int) -> None:
    started = time.monotonic()
    try:
        transcript = stt.transcribe(
            Utterance(
                pcm=pcm,
                sample_rate=sample_rate,
                duration_ms=int(len(pcm) / 2 / sample_rate * 1000),
            )
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        channel.asr_error = str(exc)
        return
    channel.asr_text = transcript
    channel.asr_latency_ms = int((time.monotonic() - started) * 1000)


if __name__ == "__main__":
    raise SystemExit(main())
