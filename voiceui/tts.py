from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
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


class MimoTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config

    def speak(self, text: str) -> None:
        print(f"assistant> {text}")
        wav_data = self.synthesize(text)
        _play_wav_bytes(wav_data, self.config.playback_device)

    def synthesize(self, text: str) -> bytes:
        headers = {}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if api_key:
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
        return _extract_audio_bytes(data)


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
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"TTS request failed: {url}: {exc}") from exc


def _chat_completions_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_audio_bytes(data: dict) -> bytes:
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("TTS response did not contain choices.")
    message = choices[0].get("message", {})
    audio = message.get("audio") or {}
    encoded = audio.get("data")
    if not encoded:
        raise RuntimeError("TTS response did not contain message.audio.data.")
    return base64.b64decode(str(encoded))


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
