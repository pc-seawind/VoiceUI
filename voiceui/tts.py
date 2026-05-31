from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from voiceui.models import TtsConfig


class TextToSpeech:
    def speak(self, text: str) -> None:
        raise NotImplementedError


class ConsoleTextToSpeech(TextToSpeech):
    def speak(self, text: str) -> None:
        print(f"assistant> {text}")


class SystemTextToSpeech(TextToSpeech):
    def speak(self, text: str) -> None:
        print(f"assistant> {text}")
        if sys.platform == "win32":
            _run_tts_command(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        "$s.Speak([Console]::In.ReadToEnd())"
                    ),
                ],
                text,
            )
            return
        if sys.platform == "darwin":
            _run_tts_command(["say"], text)
            return

        try:
            _run_tts_command(["spd-say"], text)
        except RuntimeError:
            _run_tts_command(["espeak"], text)


class PiperHttpTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config

    def speak(self, text: str) -> None:
        url = f"{self.config.piper_url.rstrip('/')}?{urllib.parse.urlencode({'text': text})}"
        with urllib.request.urlopen(url, timeout=30) as response:
            wav_data = response.read()
        _play_wav_bytes(wav_data, self.config.playback_device)


class PiperCliTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config

    def speak(self, text: str) -> None:
        if not self.config.piper_model:
            raise RuntimeError("tts.piper_model is required for piper_cli.")

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "reply.wav"
            command = [
                "piper",
                "--model",
                self.config.piper_model,
                "--output_file",
                str(wav_path),
            ]
            subprocess.run(command, input=text, text=True, check=True)
            _play_wav_bytes(wav_path.read_bytes(), self.config.playback_device)


def create_tts(config: TtsConfig) -> TextToSpeech:
    if config.provider == "console":
        return ConsoleTextToSpeech()
    if config.provider == "system":
        return SystemTextToSpeech()
    if config.provider == "piper_http":
        return PiperHttpTextToSpeech(config)
    if config.provider == "piper_cli":
        return PiperCliTextToSpeech(config)
    raise ValueError(f"Unsupported TTS provider: {config.provider}")


def _run_tts_command(command: list[str], text: str) -> None:
    try:
        subprocess.run(command, input=text, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"System TTS command is not available: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"System TTS command failed: {command[0]}") from exc


def _play_wav_bytes(wav_data: bytes, device: str | int | None) -> None:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Audio playback requires sounddevice and soundfile. "
            "Install with: pip install -e \".[tts]\""
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        temp_file.write(wav_data)
        temp_file.flush()
        data, sample_rate = sf.read(temp_file.name, dtype="float32")
        sd.play(data, sample_rate, device=None if device == "default" else device)
        sd.wait()
