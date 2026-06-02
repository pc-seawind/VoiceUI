from __future__ import annotations

import audioop
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

    playback_sample_rate = _select_output_sample_rate(
        sd,
        device=device,
        requested_sample_rate=sample_rate,
        channels=channels,
    )
    if playback_sample_rate != sample_rate:
        pcm = _resample_pcm16(
            pcm,
            source_rate=sample_rate,
            target_rate=playback_sample_rate,
            channels=channels,
        )
        print(
            "wake_ack> resampled "
            f"source_sample_rate={sample_rate} playback_sample_rate={playback_sample_rate}"
        )

    with sd.RawOutputStream(
        samplerate=playback_sample_rate,
        channels=channels,
        dtype="int16",
        device=device,
    ) as stream:
        stream.write(pcm)


def _select_output_sample_rate(
    sd,
    *,
    device: str | int | None,
    requested_sample_rate: int,
    channels: int,
) -> int:
    try:
        sd.check_output_settings(
            device=device,
            samplerate=requested_sample_rate,
            channels=channels,
            dtype="int16",
        )
        return requested_sample_rate
    except Exception as requested_error:
        for sample_rate in _candidate_output_sample_rates(sd, device):
            if sample_rate == requested_sample_rate:
                continue
            try:
                sd.check_output_settings(
                    device=device,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="int16",
                )
                return sample_rate
            except Exception:
                continue
        raise requested_error


def _candidate_output_sample_rates(sd, device: str | int | None) -> list[int]:
    candidates: list[int] = []
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


def _resample_pcm16(
    pcm: bytes,
    *,
    source_rate: int,
    target_rate: int,
    channels: int,
) -> bytes:
    if source_rate == target_rate or not pcm:
        return pcm
    converted, _state = audioop.ratecv(
        pcm,
        2,
        channels,
        source_rate,
        target_rate,
        None,
    )
    return converted
