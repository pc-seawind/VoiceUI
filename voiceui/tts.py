from __future__ import annotations

import base64
import io
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from voiceui.http_utils import post_json, require_api_key
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


class MimoTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config

    def speak(self, text: str) -> None:
        print(f"assistant> {text}")
        audio_data = self.synthesize(text)
        _play_audio_bytes(
            audio_data.data,
            audio_format=audio_data.format,
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
        )

    def synthesize(self, text: str) -> "SynthesizedAudio":
        headers = {}
        if self.config.api_key_env:
            api_key = require_api_key(self.config.api_key_env)
            headers["api-key"] = api_key

        messages: list[dict[str, str]] = []
        if self.config.style_prompt:
            messages.append({"role": "user", "content": self.config.style_prompt})
        messages.append({"role": "assistant", "content": text})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "audio": {
                "format": self.config.audio_format,
                "voice": self.config.voice,
            },
        }
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        return _extract_audio(data, fallback_format=self.config.audio_format)


class PiperHttpTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config

    def speak(self, text: str) -> None:
        url = f"{self.config.piper_url.rstrip('/')}?{urllib.parse.urlencode({'text': text})}"
        with urllib.request.urlopen(url, timeout=30) as response:
            wav_data = response.read()
        _play_audio_bytes(wav_data, audio_format="wav", device=self.config.playback_device)


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
            _play_audio_bytes(wav_path.read_bytes(), audio_format="wav", device=self.config.playback_device)


def create_tts(config: TtsConfig) -> TextToSpeech:
    if config.provider == "console":
        return ConsoleTextToSpeech()
    if config.provider == "system":
        return SystemTextToSpeech()
    if config.provider in ("mify", "mimo"):
        return MimoTextToSpeech(config)
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


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict:
    return post_json(
        url,
        payload,
        headers=headers,
        timeout=timeout,
        error_prefix="TTS request failed",
    )


def _chat_completions_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class SynthesizedAudio:
    def __init__(self, data: bytes, audio_format: str):
        self.data = data
        self.format = audio_format


def _extract_audio(data: dict, fallback_format: str) -> SynthesizedAudio:
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("TTS response did not contain choices.")
    message = choices[0].get("message", {})
    audio = message.get("audio") or {}
    encoded = audio.get("data")
    if not encoded:
        raise RuntimeError("TTS response did not contain message.audio.data.")
    audio_format = str(audio.get("format") or fallback_format or "pcm").lower()
    return SynthesizedAudio(base64.b64decode(str(encoded)), audio_format)


def _play_audio_bytes(
    audio_data: bytes,
    audio_format: str,
    device: str | int | None = None,
    sample_rate: int = 24000,
) -> None:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Audio playback requires sounddevice and soundfile. "
            "Install with: pip install -e \".[tts]\""
        ) from exc

    normalized_format = audio_format.lower().lstrip(".")
    playback_device = None if device == "default" else device
    if normalized_format == "pcm":
        import numpy as np  # type: ignore[import-untyped]

        data = np.frombuffer(audio_data, dtype="<i2").astype("float32") / 32768.0
        sd.play(data, sample_rate, device=playback_device)
        sd.wait()
        return

    if normalized_format == "wav" or audio_data.startswith(b"RIFF"):
        data, wav_sample_rate = sf.read(io.BytesIO(audio_data), dtype="float32")
        sd.play(data, wav_sample_rate, device=playback_device)
        sd.wait()
        return

    raise RuntimeError(f"Unsupported TTS audio format: {audio_format}")
