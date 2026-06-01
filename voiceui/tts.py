from __future__ import annotations

import base64
import io
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from voiceui.aliyun import get_aliyun_nls_token
from voiceui.audio import write_pcm16_wav
from voiceui.http_utils import post_json, require_api_key
from voiceui.models import TtsConfig


class TextToSpeech:
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        raise NotImplementedError


class ConsoleTextToSpeech(TextToSpeech):
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        print(f"assistant> {text}")


class SystemTextToSpeech(TextToSpeech):
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        print(f"assistant> {text}")
        if _stop_requested(stop_event):
            return
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

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        print(f"assistant> {text}")
        if self.config.stream:
            self.speak_streaming(text, stop_event=stop_event)
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
            stop_event=stop_event,
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

    def speak_streaming(
        self,
        text: str,
        stop_event: threading.Event | None = None,
    ) -> None:
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
                if _stop_requested(stop_event):
                    break
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
            stop_event=stop_event,
        )
        if written_chunks == 0 and not _stop_requested(stop_event):
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


class OpenAISpeechTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        print(f"assistant> {text}")
        if self.config.stream:
            self.speak_streaming(text, stop_event=stop_event)
            return

        request_started = time.monotonic()
        audio_format = _openai_speech_response_format(self.config.audio_format, stream=False)
        audio_data = _post_binary(
            _openai_speech_url(self.config.endpoint),
            self._payload(text, audio_format),
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
            error_prefix="OpenAI-compatible TTS request failed",
        )
        print(f"tts> synth_latency_ms={int((time.monotonic() - request_started) * 1000)}")
        playback_started = time.monotonic()
        _play_audio_bytes(
            audio_data,
            audio_format=audio_format,
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
            stop_event=stop_event,
        )
        print(f"tts> playback_latency_ms={int((time.monotonic() - playback_started) * 1000)}")

    def speak_streaming(
        self,
        text: str,
        stop_event: threading.Event | None = None,
    ) -> None:
        request_started = time.monotonic()
        first_audio_ms: int | None = None
        chunks = 0
        audio_format = _openai_speech_response_format(self.config.audio_format, stream=True)

        def audio_chunks() -> Iterator[bytes]:
            nonlocal first_audio_ms, chunks
            pending = b""
            for chunk in _post_binary_stream(
                _openai_speech_url(self.config.endpoint),
                self._payload(text, audio_format),
                headers=self._headers(),
                timeout=self.config.timeout_seconds,
                error_prefix="OpenAI-compatible streaming TTS request failed",
            ):
                if _stop_requested(stop_event):
                    break
                if not chunk:
                    continue
                data = pending + chunk
                playable_len = len(data) - (len(data) % 2)
                pending = data[playable_len:]
                if not playable_len:
                    continue
                if first_audio_ms is None:
                    first_audio_ms = int((time.monotonic() - request_started) * 1000)
                chunks += 1
                yield data[:playable_len]

        playback_started = time.monotonic()
        written_chunks = _play_pcm_stream(
            audio_chunks(),
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
            stop_event=stop_event,
        )
        if written_chunks == 0 and not _stop_requested(stop_event):
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
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _payload(self, text: str, audio_format: str) -> dict:
        return {
            "model": self.config.model,
            "input": text,
            "voice": self.config.voice,
            "response_format": audio_format,
        }


class AliyunNlsTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config
        self._token: str | None = None

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        print(f"assistant> {text}")
        if self.config.stream:
            self.speak_streaming(text, stop_event=stop_event)
            return

        request_started = time.monotonic()
        audio_data = self.synthesize(text)
        print(f"tts> synth_latency_ms={int((time.monotonic() - request_started) * 1000)}")
        playback_started = time.monotonic()
        _play_audio_bytes(
            audio_data.data,
            audio_format=audio_data.format,
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
            stop_event=stop_event,
        )
        print(f"tts> playback_latency_ms={int((time.monotonic() - playback_started) * 1000)}")

    def synthesize(self, text: str) -> SynthesizedAudio:
        chunks = list(
            _aliyun_stream_input_tts_chunks(
                config=self.config,
                token=self._token_or_create(),
                text=text,
            )
        )
        if not chunks:
            raise RuntimeError("Aliyun NLS TTS response did not contain audio data.")
        return SynthesizedAudio(b"".join(chunks), _aliyun_tts_audio_format(self.config.audio_format))

    def speak_streaming(
        self,
        text: str,
        stop_event: threading.Event | None = None,
    ) -> None:
        request_started = time.monotonic()
        first_audio_ms: int | None = None
        chunks = 0

        def audio_chunks() -> Iterator[bytes]:
            nonlocal first_audio_ms, chunks
            for chunk in _aliyun_stream_input_tts_chunks(
                config=self.config,
                token=self._token_or_create(),
                text=text,
                stop_event=stop_event,
            ):
                if _stop_requested(stop_event):
                    break
                if first_audio_ms is None:
                    first_audio_ms = int((time.monotonic() - request_started) * 1000)
                chunks += 1
                yield chunk

        playback_started = time.monotonic()
        written_chunks = _play_pcm_stream(
            audio_chunks(),
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
            stop_event=stop_event,
        )
        if written_chunks == 0 and not _stop_requested(stop_event):
            raise RuntimeError("Aliyun NLS streaming TTS response did not contain audio chunks.")
        print(
            "tts> stream_first_audio_ms="
            f"{first_audio_ms if first_audio_ms is not None else 0} "
            f"stream_chunks={chunks} "
            f"playback_latency_ms={int((time.monotonic() - playback_started) * 1000)}"
        )

    def _token_or_create(self) -> str:
        if self._token is None:
            access_key_id = require_api_key(
                self.config.access_key_id_env or "ALIYUN_AccessKeyId"
            )
            access_key_secret = require_api_key(
                self.config.access_key_secret_env or "ALIYUN_AccessKeySecret"
            )
            self._token = get_aliyun_nls_token(access_key_id, access_key_secret)
        return self._token


class PiperHttpTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        url = f"{self.config.piper_url.rstrip('/')}?{urllib.parse.urlencode({'text': text})}"
        with urllib.request.urlopen(url, timeout=30) as response:
            wav_data = response.read()
        _play_audio_bytes(
            wav_data,
            audio_format="wav",
            device=self.config.playback_device,
            stop_event=stop_event,
        )


class PiperCliTextToSpeech(TextToSpeech):
    def __init__(self, config: TtsConfig):
        self.config = config

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        if not self.config.piper_model:
            raise RuntimeError("tts.piper_model is required for piper_cli.")
        if _stop_requested(stop_event):
            return

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
            _play_audio_bytes(
                wav_path.read_bytes(),
                audio_format="wav",
                device=self.config.playback_device,
                stop_event=stop_event,
            )


def create_tts(config: TtsConfig) -> TextToSpeech:
    if config.provider == "console":
        return ConsoleTextToSpeech()
    if config.provider == "system":
        return SystemTextToSpeech()
    if config.provider in ("mify", "mimo"):
        return MimoTextToSpeech(config)
    if config.provider in ("openai_speech", "openai_compatible_speech"):
        return OpenAISpeechTextToSpeech(config)
    if config.provider == "aliyun_nls":
        return AliyunNlsTextToSpeech(config)
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


def _post_binary(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    error_prefix: str = "Binary HTTP request failed",
) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{error_prefix}: {url}: HTTP {exc.code}: {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{error_prefix}: {url}: {exc}") from exc


def _post_binary_stream(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    chunk_bytes: int = 1024,
    error_prefix: str = "Streaming binary HTTP request failed",
) -> Iterator[bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            read = getattr(response, "read1", response.read)
            while True:
                chunk = read(chunk_bytes)
                if not chunk:
                    break
                yield chunk
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{error_prefix}: {url}: HTTP {exc.code}: {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{error_prefix}: {url}: {exc}") from exc


def _chat_completions_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _openai_speech_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/audio/speech"):
        return base
    if base.endswith("/v1"):
        return f"{base}/audio/speech"
    return f"{base}/v1/audio/speech"


def _aliyun_stream_input_tts_chunks(
    *,
    config: TtsConfig,
    token: str,
    text: str,
    stop_event: threading.Event | None = None,
) -> Iterator[bytes]:
    try:
        import nls  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Aliyun NLS SDK is not installed. Install with: "
            "pip install git+https://github.com/aliyun/alibabacloud-nls-python-sdk.git"
        ) from exc

    app_key = require_api_key(config.app_key_env or "ALIYUN_NLS_APPKEY")
    audio_format = _aliyun_tts_audio_format(config.audio_format)
    items: queue.Queue[object] = queue.Queue()
    done = object()

    def on_data(data: bytes, *_args: object) -> None:
        if data and not _stop_requested(stop_event):
            items.put(bytes(data))

    def on_error(message: str, *_args: object) -> None:
        items.put(RuntimeError(f"Aliyun NLS TTS failed: {message}"))

    def producer() -> None:
        synthesizer = nls.NlsStreamInputTtsSynthesizer(
            url=config.endpoint,
            token=token,
            appkey=app_key,
            on_data=on_data,
            on_error=on_error,
            callback_args=[],
        )
        try:
            synthesizer.startStreamInputTts(
                voice=config.voice,
                aformat=audio_format,
                sample_rate=config.sample_rate,
                volume=config.volume,
                speech_rate=config.speech_rate,
                pitch_rate=config.pitch_rate,
            )
            for text_chunk in _split_stream_input_text(text):
                if _stop_requested(stop_event):
                    break
                synthesizer.sendStreamInputTts(text_chunk)
                time.sleep(0.05)
            synthesizer.stopStreamInputTts()
        except Exception as exc:
            items.put(exc)
        finally:
            try:
                synthesizer.shutdown()
            except Exception:
                pass
            items.put(done)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    deadline = time.monotonic() + max(1.0, config.timeout_seconds)
    while True:
        if _stop_requested(stop_event) and items.empty():
            break
        timeout = max(0.1, min(1.0, deadline - time.monotonic()))
        try:
            item = items.get(timeout=timeout)
        except queue.Empty:
            if time.monotonic() >= deadline:
                raise RuntimeError("Aliyun NLS TTS timed out.")
            continue
        if item is done:
            break
        if isinstance(item, Exception):
            raise item
        yield item  # type: ignore[misc]


def _aliyun_tts_audio_format(audio_format: str) -> str:
    normalized = _normalize_audio_format(audio_format)
    if normalized in ("pcm", "pcm16"):
        return "pcm"
    if normalized in ("wav", "mp3", "opus"):
        return normalized
    raise RuntimeError(f"Unsupported Aliyun NLS TTS audio format: {audio_format}")


_STREAM_SENTENCE_BREAKS = {
    "\u3002",
    "\uff01",
    "\uff1f",
    "\uff1b",
    "!",
    "?",
    ";",
    "\n",
}


def _split_stream_input_text(text: str, max_chars: int = 40) -> list[str]:
    parts: list[str] = []
    current = ""
    for char in text.strip():
        current += char
        if char in _STREAM_SENTENCE_BREAKS or len(current) >= max_chars:
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [part for part in parts if part.strip()]


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


def _openai_speech_response_format(audio_format: str, stream: bool) -> str:
    normalized = _normalize_audio_format(audio_format)
    if stream:
        return "pcm"
    if normalized in ("pcm", "pcm16"):
        return "pcm"
    return normalized


def _play_audio_bytes(
    audio_data: bytes,
    audio_format: str,
    device: str | int | None = None,
    sample_rate: int = 24000,
    stop_event: threading.Event | None = None,
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
    if _stop_requested(stop_event):
        return
    if normalized_format in ("pcm", "pcm16"):
        if stop_event is not None:
            _play_pcm_stream(
                _iter_pcm_chunks(audio_data, sample_rate=sample_rate),
                sample_rate=sample_rate,
                device=device,
                stop_event=stop_event,
            )
            return

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
    stop_event: threading.Event | None = None,
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
            for playable_chunk in _iter_pcm_chunks(chunk, sample_rate=sample_rate):
                if _stop_requested(stop_event):
                    break
                if stream is None:
                    stream = sd.RawOutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="int16",
                        device=playback_device,
                    )
                    stream.start()
                stream.write(playable_chunk)
                written_chunks += 1
                if _stop_requested(stop_event):
                    break
            if _stop_requested(stop_event):
                break
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
    return written_chunks


def _iter_pcm_chunks(
    audio_data: bytes,
    sample_rate: int = 24000,
    chunk_ms: int = 20,
) -> Iterator[bytes]:
    chunk_bytes = max(2, int(sample_rate * chunk_ms / 1000) * 2)
    chunk_bytes -= chunk_bytes % 2
    playable_len = len(audio_data) - (len(audio_data) % 2)
    for offset in range(0, playable_len, chunk_bytes):
        yield audio_data[offset : offset + chunk_bytes]


def _stop_requested(stop_event: threading.Event | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())
