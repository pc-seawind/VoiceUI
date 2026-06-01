from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from voiceui.audio import write_pcm16_wav
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
        if self.config.stream:
            self.speak_streaming(text)
            return

        synth_started = time.monotonic()
        audio_data = self.synthesize(text)
        print(f"tts> synth_latency_ms={int((time.monotonic() - synth_started) * 1000)}")
        playback_started = time.monotonic()
        _play_audio_bytes(
            audio_data.data,
            audio_format=audio_data.format,
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
        )
        print(f"tts> playback_latency_ms={int((time.monotonic() - playback_started) * 1000)}")

    def synthesize(self, text: str) -> "SynthesizedAudio":
        headers = self._headers()
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            self._payload(text, stream=False),
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        return _extract_audio(data, fallback_format=self.config.audio_format)

    def speak_streaming(self, text: str) -> None:
        request_started = time.monotonic()
        first_audio_ms: int | None = None
        chunks = 0
        stream_audio_format = _mimo_audio_format(self.config.audio_format, stream=True)

        def audio_chunks() -> Iterator[bytes]:
            nonlocal first_audio_ms, chunks
            for event in _post_json_stream(
                _chat_completions_url(self.config.endpoint),
                self._payload(text, stream=True),
                headers=self._headers(),
                timeout=self.config.timeout_seconds,
            ):
                audio = _extract_stream_audio(event)
                if not audio:
                    continue
                audio_format = _normalize_audio_format(
                    str(audio.get("format") or stream_audio_format)
                )
                if audio_format not in ("pcm", "pcm16"):
                    raise RuntimeError(f"Streaming TTS requires PCM audio, got {audio_format}")
                encoded = audio.get("data")
                if not encoded:
                    continue
                if first_audio_ms is None:
                    first_audio_ms = int((time.monotonic() - request_started) * 1000)
                chunks += 1
                yield base64.b64decode(str(encoded))

        playback_started = time.monotonic()
        written_chunks = _play_pcm_stream(
            audio_chunks(),
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
        )
        if written_chunks == 0:
            raise RuntimeError("Streaming TTS response did not contain audio chunks.")
        print(
            "tts> stream_first_audio_ms="
            f"{first_audio_ms if first_audio_ms is not None else 0} "
            f"stream_chunks={chunks} "
            f"playback_latency_ms={int((time.monotonic() - playback_started) * 1000)}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.config.api_key_env:
            api_key = require_api_key(self.config.api_key_env)
            headers["api-key"] = api_key
        return headers

    def _payload(self, text: str, stream: bool) -> dict:
        messages: list[dict[str, str]] = []
        if self.config.style_prompt:
            messages.append({"role": "user", "content": self.config.style_prompt})
        messages.append({"role": "assistant", "content": text})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "audio": {
                "format": _mimo_audio_format(self.config.audio_format, stream=stream),
                "voice": self.config.voice,
            },
        }
        if stream:
            payload["stream"] = True
        return payload


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


def synthesize_to_wav(config: TtsConfig, text: str, output_path: str | Path) -> Path:
    tts = create_tts(config)
    synthesize = getattr(tts, "synthesize", None)
    if synthesize is None:
        raise RuntimeError(f"tts.provider={config.provider} does not support offline synthesis.")

    audio_data = synthesize(text)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_format = _normalize_audio_format(audio_data.format)
    if normalized_format == "wav" or audio_data.data.startswith(b"RIFF"):
        path.write_bytes(audio_data.data)
        return path
    if normalized_format in ("pcm", "pcm16"):
        write_pcm16_wav(path, audio_data.data, sample_rate=config.sample_rate)
        return path
    raise RuntimeError(f"Cannot save unsupported TTS audio format as WAV: {audio_data.format}")


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


def _post_json_stream(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> Iterator[dict]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break
                yield json.loads(line)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"TTS streaming request failed: {url}: HTTP {exc.code}: {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"TTS streaming request failed: {url}: {exc}") from exc


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
    audio_format = _normalize_audio_format(str(audio.get("format") or fallback_format or "pcm"))
    return SynthesizedAudio(base64.b64decode(str(encoded)), audio_format)


def _extract_stream_audio(data: dict) -> dict | None:
    choices = data.get("choices", [])
    if not choices:
        return None
    choice = choices[0]
    delta = choice.get("delta") or {}
    message = choice.get("message") or {}
    audio = delta.get("audio") or message.get("audio")
    return audio if isinstance(audio, dict) else None


def _normalize_audio_format(audio_format: str) -> str:
    normalized = audio_format.lower().lstrip(".")
    if normalized in ("pcm_s16le", "s16le"):
        return "pcm16"
    return normalized


def _mimo_audio_format(audio_format: str, stream: bool) -> str:
    if stream:
        return "pcm16"
    return audio_format


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

    normalized_format = _normalize_audio_format(audio_format)
    playback_device = None if device == "default" else device
    if normalized_format in ("pcm", "pcm16"):
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


def _play_pcm_stream(
    chunks: Iterator[bytes],
    sample_rate: int = 24000,
    device: str | int | None = None,
) -> int:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Streaming audio playback requires sounddevice. "
            "Install with: pip install -e \".[tts]\""
        ) from exc

    playback_device = None if device == "default" else device
    written_chunks = 0
    stream = None
    try:
        for chunk in chunks:
            if stream is None:
                stream = sd.RawOutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                    device=playback_device,
                )
                stream.start()
            stream.write(chunk)
            written_chunks += 1
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
    return written_chunks
