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

    with sd.RawOutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        device=device,
    ) as stream:
        stream.write(pcm)
