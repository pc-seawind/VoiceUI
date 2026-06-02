from __future__ import annotations

from importlib import resources
from pathlib import Path
import wave

from voiceui.models import WakeAckConfig

_DEFAULT_ACK_RESOURCE = "wake_ack_wo_zai.wav"


class WakeAckPlayer:
    def play(self) -> None:
        raise NotImplementedError


class DisabledWakeAckPlayer(WakeAckPlayer):
    def play(self) -> None:
        return


class WavWakeAckPlayer(WakeAckPlayer):
    def __init__(self, config: WakeAckConfig, fallback_device: str | int | None = None):
        self.config = config
        self.fallback_device = fallback_device

    def play(self) -> None:
        wav_path = resolve_wake_ack_path(self.config.wav_path)
        frames, sample_rate, channels = _read_pcm16_wav(wav_path)
        _play_pcm16(frames, sample_rate=sample_rate, channels=channels, device=self._device())

    def _device(self) -> str | int | None:
        device = self.config.playback_device
        if device is None:
            device = self.fallback_device
        return None if device == "default" else device


def create_wake_ack_player(
    config: WakeAckConfig,
    fallback_device: str | int | None = None,
) -> WakeAckPlayer:
    if not config.enabled:
        return DisabledWakeAckPlayer()
    return WavWakeAckPlayer(config, fallback_device=fallback_device)


def resolve_wake_ack_path(wav_path: str = "") -> Path:
    if not wav_path or wav_path == "default":
        return Path(str(resources.files("voiceui").joinpath("resources", _DEFAULT_ACK_RESOURCE)))

    path = Path(wav_path).expanduser()
    if path.exists():
        return path
    package_path = resources.files("voiceui").joinpath(wav_path)
    return Path(str(package_path))


def _read_pcm16_wav(path: Path) -> tuple[bytes, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Wake ack WAV does not exist: {path}")
    with wave.open(str(path), "rb") as wav:
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"Wake ack WAV must be 16-bit PCM, got sample_width={sample_width}")
        return wav.readframes(wav.getnframes()), wav.getframerate(), wav.getnchannels()


def _play_pcm16(
    pcm: bytes,
    sample_rate: int,
    channels: int,
    device: str | int | None = None,
) -> None:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Wake ack playback requires sounddevice. Install with: pip install -e \".[audio]\""
        ) from exc

    playback_sample_rate, playback_channels = _select_output_format(
        sd,
        device=device,
        requested_sample_rate=sample_rate,
        source_channels=channels,
    )
    resampler = "none"
    if playback_sample_rate != sample_rate:
        pcm, resampler = _resample_pcm16(
            pcm,
            source_rate=sample_rate,
            target_rate=playback_sample_rate,
            channels=channels,
        )
    if playback_channels != channels:
        pcm = _convert_pcm16_channels(
            pcm,
            source_channels=channels,
            target_channels=playback_channels,
        )
    if playback_sample_rate != sample_rate or playback_channels != channels:
        print(
            "wake_ack> converted "
            f"source_sample_rate={sample_rate} playback_sample_rate={playback_sample_rate} "
            f"source_channels={channels} playback_channels={playback_channels} "
            f"resampler={resampler}"
        )

    with sd.RawOutputStream(
        samplerate=playback_sample_rate,
        channels=playback_channels,
        dtype="int16",
        device=device,
    ) as stream:
        stream.write(pcm)


def _select_output_format(
    sd,
    *,
    device: str | int | None,
    requested_sample_rate: int,
    source_channels: int,
) -> tuple[int, int]:
    requested_error: Exception | None = None
    for sample_rate in _candidate_output_sample_rates(sd, device, requested_sample_rate):
        for channels in _candidate_output_channels(sd, device, source_channels):
            try:
                sd.check_output_settings(
                    device=device,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="int16",
                )
                return sample_rate, channels
            except Exception as exc:
                if sample_rate == requested_sample_rate and channels == source_channels:
                    requested_error = exc
                continue
    if requested_error is not None:
        raise requested_error
    raise RuntimeError(
        "Could not find a supported wake acknowledgement playback format "
        f"for device={device!r} requested_sample_rate={requested_sample_rate} "
        f"source_channels={source_channels}"
    )


def _candidate_output_sample_rates(
    sd,
    device: str | int | None,
    requested_sample_rate: int,
) -> list[int]:
    candidates: list[int] = [requested_sample_rate]
    try:
        info = sd.query_devices(device, "output")
        default_rate = int(round(float(info.get("default_samplerate") or 0)))
        if default_rate > 0:
            candidates.append(default_rate)
    except Exception:
        pass
    for sample_rate in (16000, 24000, 48000, 44100):
        if sample_rate not in candidates:
            candidates.append(sample_rate)
    return candidates


def _candidate_output_channels(
    sd,
    device: str | int | None,
    source_channels: int,
) -> list[int]:
    source_channels = max(1, source_channels)
    max_channels = source_channels
    try:
        info = sd.query_devices(device, "output")
        max_channels = int(info.get("max_output_channels") or source_channels)
    except Exception:
        pass

    candidates: list[int] = []
    if source_channels == 1 and max_channels >= 2:
        candidates.append(2)
    candidates.append(source_channels)
    for channels in (2, 1):
        if channels <= max_channels and channels not in candidates:
            candidates.append(channels)
    return candidates


def _resample_pcm16(
    pcm: bytes,
    *,
    source_rate: int,
    target_rate: int,
    channels: int,
) -> tuple[bytes, str]:
    if source_rate == target_rate or not pcm:
        return pcm, "none"
    try:
        return _resample_pcm16_scipy(
            pcm,
            source_rate=source_rate,
            target_rate=target_rate,
            channels=channels,
        ), "scipy"
    except Exception:
        import audioop

        converted, _state = audioop.ratecv(
            pcm,
            2,
            channels,
            source_rate,
            target_rate,
            None,
            1,
            1,
        )
        return converted, "audioop"


def _resample_pcm16_scipy(
    pcm: bytes,
    *,
    source_rate: int,
    target_rate: int,
    channels: int,
) -> bytes:
    import math

    import numpy as np  # type: ignore[import-untyped]
    from scipy.signal import resample_poly  # type: ignore[import-untyped]

    channels = max(1, channels)
    samples = np.frombuffer(pcm, dtype="<i2")
    frame_count = len(samples) // channels
    if frame_count == 0:
        return b""
    samples = samples[: frame_count * channels].reshape(frame_count, channels).astype(np.float32)
    common = math.gcd(source_rate, target_rate)
    up = target_rate // common
    down = source_rate // common
    converted = resample_poly(samples, up, down, axis=0, window=("kaiser", 8.6))
    converted = np.clip(np.rint(converted), -32768, 32767).astype("<i2")
    return converted.reshape(-1).tobytes()


def _convert_pcm16_channels(
    pcm: bytes,
    *,
    source_channels: int,
    target_channels: int,
) -> bytes:
    source_channels = max(1, source_channels)
    target_channels = max(1, target_channels)
    if source_channels == target_channels or not pcm:
        return pcm
    try:
        import numpy as np  # type: ignore[import-untyped]

        samples = np.frombuffer(pcm, dtype="<i2")
        frame_count = len(samples) // source_channels
        if frame_count == 0:
            return b""
        frames = samples[: frame_count * source_channels].reshape(frame_count, source_channels)
        if source_channels == 1:
            converted = np.repeat(frames, target_channels, axis=1)
        elif target_channels == 1:
            converted = np.rint(frames.astype(np.float32).mean(axis=1, keepdims=True))
        else:
            converted = np.zeros((frame_count, target_channels), dtype=np.int16)
            copy_channels = min(source_channels, target_channels)
            converted[:, :copy_channels] = frames[:, :copy_channels]
        return np.clip(converted, -32768, 32767).astype("<i2").reshape(-1).tobytes()
    except Exception:
        import audioop

        if source_channels == 1 and target_channels == 2:
            return audioop.tostereo(pcm, 2, 1, 1)
        if source_channels == 2 and target_channels == 1:
            return audioop.tomono(pcm, 2, 0.5, 0.5)
        raise
