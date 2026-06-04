from __future__ import annotations

import array
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
from collections.abc import Callable, Iterator
from pathlib import Path

from voiceui.aliyun import get_aliyun_nls_token
from voiceui.audio import resolve_sounddevice_device, write_pcm16_wav
from voiceui.audio_dump import current_audio_dump_manager
from voiceui.http_utils import post_json, require_api_key
from voiceui.logs import log_continuous, log_event
from voiceui.models import TtsConfig
from voiceui.wake_ack import (
    _convert_pcm16_channels,
    _resample_pcm16,
    _select_output_format,
)


class TextToSpeech:
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        raise NotImplementedError

    def speak_text_stream(
        self,
        text_chunks: Iterator[str],
        stop_event: threading.Event | None = None,
    ) -> str:
        full_text_parts: list[str] = []
        tracked_chunks = _track_text_chunks(text_chunks, full_text_parts, stop_event)
        for segment in _iter_stream_input_text(tracked_chunks, max_chars=48, min_chars=16):
            if _stop_requested(stop_event):
                break
            self.speak(segment, stop_event=stop_event)
        return "".join(full_text_parts).strip()


class ConsoleTextToSpeech(TextToSpeech):
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        return

    def speak_text_stream(
        self,
        text_chunks: Iterator[str],
        stop_event: threading.Event | None = None,
    ) -> str:
        full_text_parts: list[str] = []
        for _ in _track_text_chunks(text_chunks, full_text_parts, stop_event):
            pass
        return "".join(full_text_parts).strip()


class SystemTextToSpeech(TextToSpeech):
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
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
        if self.config.stream:
            self.speak_streaming(text, stop_event=stop_event)
            return

        synth_started = time.monotonic()
        audio_data = self.synthesize(text)
        log_event(
            "tts",
            "synthesis_completed",
            log_id="tts.synthesis_completed",
            provider=self.config.provider,
            latency_ms=int((time.monotonic() - synth_started) * 1000),
        )
        playback_started = time.monotonic()
        _play_audio_bytes(
            audio_data.data,
            audio_format=audio_data.format,
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
            playback_sample_rate=self.config.playback_sample_rate,
            playback_channels=self.config.playback_channels,
            limiter_enabled=self.config.limiter_enabled,
            limiter_threshold=self.config.limiter_threshold,
            stop_event=stop_event,
        )
        log_event(
            "tts",
            "playback_completed",
            log_id="tts.playback_completed",
            provider=self.config.provider,
            latency_ms=int((time.monotonic() - playback_started) * 1000),
        )

    def synthesize(self, text: str) -> SynthesizedAudio:
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
            playback_sample_rate=self.config.playback_sample_rate,
            playback_channels=self.config.playback_channels,
            limiter_enabled=self.config.limiter_enabled,
            limiter_threshold=self.config.limiter_threshold,
            stop_event=stop_event,
        )
        if written_chunks == 0 and not _stop_requested(stop_event):
            raise RuntimeError("Streaming TTS response did not contain audio chunks.")
        log_event(
            "tts",
            "stream_completed",
            log_id="tts.stream_completed",
            provider=self.config.provider,
            first_audio_ms=first_audio_ms if first_audio_ms is not None else 0,
            stream_chunks=chunks,
            playback_latency_ms=int((time.monotonic() - playback_started) * 1000),
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
        log_event(
            "tts",
            "synthesis_completed",
            log_id="tts.synthesis_completed",
            provider=self.config.provider,
            latency_ms=int((time.monotonic() - request_started) * 1000),
        )
        playback_started = time.monotonic()
        _play_audio_bytes(
            audio_data,
            audio_format=audio_format,
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
            playback_sample_rate=self.config.playback_sample_rate,
            playback_channels=self.config.playback_channels,
            limiter_enabled=self.config.limiter_enabled,
            limiter_threshold=self.config.limiter_threshold,
            stop_event=stop_event,
        )
        log_event(
            "tts",
            "playback_completed",
            log_id="tts.playback_completed",
            provider=self.config.provider,
            latency_ms=int((time.monotonic() - playback_started) * 1000),
        )

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
            playback_sample_rate=self.config.playback_sample_rate,
            playback_channels=self.config.playback_channels,
            limiter_enabled=self.config.limiter_enabled,
            limiter_threshold=self.config.limiter_threshold,
            stop_event=stop_event,
        )
        if written_chunks == 0 and not _stop_requested(stop_event):
            raise RuntimeError("Streaming TTS response did not contain audio chunks.")
        log_event(
            "tts",
            "stream_completed",
            log_id="tts.stream_completed",
            provider=self.config.provider,
            first_audio_ms=first_audio_ms if first_audio_ms is not None else 0,
            stream_chunks=chunks,
            playback_latency_ms=int((time.monotonic() - playback_started) * 1000),
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
        if self.config.stream:
            self.speak_streaming(text, stop_event=stop_event)
            return

        request_started = time.monotonic()
        audio_data = self.synthesize(text)
        log_event(
            "tts",
            "synthesis_completed",
            log_id="tts.synthesis_completed",
            provider=self.config.provider,
            latency_ms=int((time.monotonic() - request_started) * 1000),
        )
        playback_started = time.monotonic()
        _play_audio_bytes(
            audio_data.data,
            audio_format=audio_data.format,
            sample_rate=self.config.sample_rate,
            device=self.config.playback_device,
            playback_sample_rate=self.config.playback_sample_rate,
            playback_channels=self.config.playback_channels,
            limiter_enabled=self.config.limiter_enabled,
            limiter_threshold=self.config.limiter_threshold,
            stop_event=stop_event,
        )
        log_event(
            "tts",
            "playback_completed",
            log_id="tts.playback_completed",
            provider=self.config.provider,
            latency_ms=int((time.monotonic() - playback_started) * 1000),
        )

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
        return SynthesizedAudio(
            b"".join(chunks),
            _aliyun_tts_audio_format(self.config.audio_format),
        )

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
            playback_sample_rate=self.config.playback_sample_rate,
            playback_channels=self.config.playback_channels,
            limiter_enabled=self.config.limiter_enabled,
            limiter_threshold=self.config.limiter_threshold,
            stop_event=stop_event,
        )
        if written_chunks == 0 and not _stop_requested(stop_event):
            raise RuntimeError("Aliyun NLS streaming TTS response did not contain audio chunks.")
        log_event(
            "tts",
            "stream_completed",
            log_id="tts.stream_completed",
            provider=self.config.provider,
            first_audio_ms=first_audio_ms if first_audio_ms is not None else 0,
            stream_chunks=chunks,
            playback_latency_ms=int((time.monotonic() - playback_started) * 1000),
        )

    def speak_text_stream(
        self,
        text_chunks: Iterator[str],
        stop_event: threading.Event | None = None,
    ) -> str:
        request_started = time.monotonic()
        first_audio_ms: int | None = None
        first_text_segment_ms: int | None = None
        stream_started_ms: int | None = None
        first_text_sent_ms: int | None = None
        first_text_chars = 0
        chunks = 0
        full_text_parts: list[str] = []

        def tracked_chunks() -> Iterator[str]:
            yield from _track_text_chunks(text_chunks, full_text_parts, stop_event)

        def on_tts_event(name: str, fields: dict[str, object]) -> None:
            nonlocal first_text_segment_ms, stream_started_ms, first_text_sent_ms
            nonlocal first_text_chars
            elapsed_ms = int((time.monotonic() - request_started) * 1000)
            if name == "first_text_segment":
                first_text_segment_ms = elapsed_ms
                first_text_chars = int(fields.get("chars") or 0)
                log_event(
                    "tts",
                    "first_text_segment",
                    log_id="tts.first_text_segment",
                    provider=self.config.provider,
                    latency_ms=elapsed_ms,
                    chars=first_text_chars,
                )
            elif name == "stream_started":
                stream_started_ms = elapsed_ms
                log_event(
                    "tts",
                    "stream_started",
                    log_id="tts.stream_started",
                    provider=self.config.provider,
                    latency_ms=elapsed_ms,
                )
            elif name == "first_text_sent":
                first_text_sent_ms = elapsed_ms
                log_event(
                    "tts",
                    "first_text_sent",
                    log_id="tts.first_text_sent",
                    provider=self.config.provider,
                    latency_ms=elapsed_ms,
                )

        def audio_chunks() -> Iterator[bytes]:
            nonlocal first_audio_ms, chunks
            for chunk in _aliyun_stream_input_tts_chunks_from_text_chunks(
                config=self.config,
                token=self._token_or_create(),
                text_chunks=tracked_chunks(),
                stop_event=stop_event,
                on_event=on_tts_event,
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
            playback_sample_rate=self.config.playback_sample_rate,
            playback_channels=self.config.playback_channels,
            limiter_enabled=self.config.limiter_enabled,
            limiter_threshold=self.config.limiter_threshold,
            stop_event=stop_event,
        )
        if written_chunks == 0 and full_text_parts and not _stop_requested(stop_event):
            raise RuntimeError("Aliyun NLS streaming TTS response did not contain audio chunks.")
        log_event(
            "tts",
            "stream_completed",
            log_id="tts.stream_completed",
            provider=self.config.provider,
            first_audio_ms=first_audio_ms if first_audio_ms is not None else 0,
            first_text_segment_ms=(
                first_text_segment_ms if first_text_segment_ms is not None else 0
            ),
            stream_started_ms=stream_started_ms if stream_started_ms is not None else 0,
            first_text_sent_ms=first_text_sent_ms if first_text_sent_ms is not None else 0,
            first_text_chars=first_text_chars,
            stream_chunks=chunks,
            playback_latency_ms=int((time.monotonic() - playback_started) * 1000),
        )
        return "".join(full_text_parts).strip()

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
            playback_sample_rate=self.config.playback_sample_rate,
            playback_channels=self.config.playback_channels,
            limiter_enabled=self.config.limiter_enabled,
            limiter_threshold=self.config.limiter_threshold,
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
                playback_sample_rate=self.config.playback_sample_rate,
                playback_channels=self.config.playback_channels,
                limiter_enabled=self.config.limiter_enabled,
                limiter_threshold=self.config.limiter_threshold,
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
    yield from _aliyun_stream_input_tts_chunks_from_text_chunks(
        config=config,
        token=token,
        text_chunks=iter(_split_stream_input_text(text)),
        stop_event=stop_event,
    )


def _aliyun_stream_input_tts_chunks_from_text_chunks(
    *,
    config: TtsConfig,
    token: str,
    text_chunks: Iterator[str],
    stop_event: threading.Event | None = None,
    on_event: Callable[[str, dict[str, object]], None] | None = None,
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
        synthesizer = None
        try:
            text_segments = _iter_stream_input_text(
                text_chunks,
                max_chars=32,
                min_chars=8,
                max_wait_ms=400,
            )
            first_text_chunk = _next_stream_input_text(text_segments, stop_event)
            if first_text_chunk is None:
                return
            _emit_tts_event(
                on_event,
                "first_text_segment",
                chars=len(first_text_chunk),
                text_preview=first_text_chunk[:16],
            )

            synthesizer = nls.NlsStreamInputTtsSynthesizer(
                url=config.endpoint,
                token=token,
                appkey=app_key,
                on_data=on_data,
                on_error=on_error,
                callback_args=[],
            )
            synthesizer.startStreamInputTts(
                voice=config.voice,
                aformat=audio_format,
                sample_rate=config.sample_rate,
                volume=config.volume,
                speech_rate=config.speech_rate,
                pitch_rate=config.pitch_rate,
            )
            _emit_tts_event(on_event, "stream_started")

            if not _stop_requested(stop_event):
                synthesizer.sendStreamInputTts(first_text_chunk)
                _emit_tts_event(on_event, "first_text_sent", chars=len(first_text_chunk))
                time.sleep(0.02)

            for text_chunk in text_segments:
                if _stop_requested(stop_event):
                    break
                synthesizer.sendStreamInputTts(text_chunk)
                time.sleep(0.02)
            synthesizer.stopStreamInputTts()
        except Exception as exc:
            items.put(exc)
        finally:
            if synthesizer is not None:
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
                raise RuntimeError("Aliyun NLS TTS timed out.") from None
            continue
        if item is done:
            break
        if isinstance(item, Exception):
            raise item
        deadline = time.monotonic() + max(1.0, config.timeout_seconds)
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

_STREAM_SOFT_BREAKS = {
    "\uff0c",
    "\u3001",
    ",",
    "\uff1a",
    ":",
}


def _split_stream_input_text(text: str, max_chars: int = 40) -> list[str]:
    return list(
        _iter_stream_input_text(
            iter([text.strip()]),
            max_chars=max_chars,
            min_chars=max_chars,
        )
    )


def _iter_stream_input_text(
    text_chunks: Iterator[str],
    max_chars: int = 40,
    min_chars: int = 12,
    max_wait_ms: int | None = None,
) -> Iterator[str]:
    current = ""
    current_started: float | None = None
    for chunk in text_chunks:
        for char in chunk:
            if current_started is None:
                current_started = time.monotonic()
            current += char
            waited_ms = (
                int((time.monotonic() - current_started) * 1000)
                if current_started is not None
                else 0
            )
            if (
                char in _STREAM_SENTENCE_BREAKS
                or len(current) >= max_chars
                or (char in _STREAM_SOFT_BREAKS and len(current) >= min_chars)
                or (
                    max_wait_ms is not None
                    and len(current) >= min_chars
                    and waited_ms >= max_wait_ms
                )
            ):
                if current.strip():
                    yield current
                current = ""
                current_started = None
    if current.strip():
        yield current


def _track_text_chunks(
    text_chunks: Iterator[str],
    full_text_parts: list[str],
    stop_event: threading.Event | None = None,
) -> Iterator[str]:
    for chunk in text_chunks:
        if _stop_requested(stop_event):
            break
        if not chunk:
            continue
        full_text_parts.append(chunk)
        yield chunk


def _next_stream_input_text(
    text_segments: Iterator[str],
    stop_event: threading.Event | None = None,
) -> str | None:
    for text_segment in text_segments:
        if _stop_requested(stop_event):
            return None
        if text_segment.strip():
            return text_segment
    return None


def _emit_tts_event(
    on_event: Callable[[str, dict[str, object]], None] | None,
    name: str,
    **fields: object,
) -> None:
    if on_event is not None:
        on_event(name, fields)


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
    source_channels: int = 1,
    playback_sample_rate: int | None = None,
    playback_channels: int | None = None,
    limiter_enabled: bool = False,
    limiter_threshold: float = 0.92,
    stop_event: threading.Event | None = None,
) -> None:
    try:
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Audio playback requires sounddevice and soundfile. "
            "Install with: pip install -e \".[tts]\""
        ) from exc

    normalized_format = _normalize_audio_format(audio_format)
    if _stop_requested(stop_event):
        return
    if normalized_format in ("pcm", "pcm16"):
        _play_pcm_stream(
            iter([audio_data]),
            sample_rate=sample_rate,
            source_channels=source_channels,
            device=device,
            playback_sample_rate=playback_sample_rate,
            playback_channels=playback_channels,
            limiter_enabled=limiter_enabled,
            limiter_threshold=limiter_threshold,
            stop_event=stop_event,
        )
        return

    if normalized_format == "wav" or audio_data.startswith(b"RIFF"):
        data, wav_sample_rate = sf.read(io.BytesIO(audio_data), dtype="float32", always_2d=True)
        if limiter_enabled:
            data, peak, gain = _limit_float_audio(data, limiter_threshold)
            if gain < 0.999:
                log_continuous(
                    "tts",
                    "limiter",
                    log_id="tts.limiter",
                    peak=f"{peak:.3f}",
                    threshold=f"{float(limiter_threshold):.3f}",
                    gain=f"{gain:.3f}",
                )
        pcm, wav_channels = _float_audio_to_pcm16(data)
        _play_pcm_stream(
            iter([pcm]),
            sample_rate=wav_sample_rate,
            source_channels=wav_channels,
            device=device,
            playback_sample_rate=playback_sample_rate,
            playback_channels=playback_channels,
            limiter_enabled=limiter_enabled,
            limiter_threshold=limiter_threshold,
            stop_event=stop_event,
        )
        return

    raise RuntimeError(f"Unsupported TTS audio format: {audio_format}")


def _play_pcm_stream(
    chunks: Iterator[bytes],
    sample_rate: int = 24000,
    source_channels: int = 1,
    device: str | int | None = None,
    playback_sample_rate: int | None = None,
    playback_channels: int | None = None,
    limiter_enabled: bool = False,
    limiter_threshold: float = 0.92,
    stop_event: threading.Event | None = None,
    dump_kind: str = "tts_output",
) -> int:
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Streaming audio playback requires sounddevice. "
            "Install with: pip install -e \".[tts]\""
        ) from exc

    playback_device = resolve_sounddevice_device(sd, device, kind="output")
    requested_playback_rate = playback_sample_rate or sample_rate
    requested_playback_channels = playback_channels or source_channels
    selected_rate, selected_channels = _select_output_format(
        sd,
        device=playback_device,
        requested_sample_rate=requested_playback_rate,
        source_channels=requested_playback_channels,
    )
    should_convert = selected_rate != sample_rate or selected_channels != source_channels
    converter_logged = False
    limiter_logged = False
    dump_manager = current_audio_dump_manager()
    dump_start_ms: int | None = None
    dump_chunks: list[bytes] = []
    written_chunks = 0
    stream = None
    try:
        for chunk in chunks:
            for playable_chunk in _iter_pcm_chunks(
                chunk,
                sample_rate=sample_rate,
                channels=source_channels,
            ):
                if _stop_requested(stop_event):
                    break
                resampler = "none"
                if selected_rate != sample_rate:
                    playable_chunk, resampler = _resample_pcm16(
                        playable_chunk,
                        source_rate=sample_rate,
                        target_rate=selected_rate,
                        channels=source_channels,
                    )
                if selected_channels != source_channels:
                    playable_chunk = _convert_pcm16_channels(
                        playable_chunk,
                        source_channels=source_channels,
                        target_channels=selected_channels,
                    )
                if should_convert and not converter_logged:
                    log_event(
                        "tts",
                        "converted",
                        log_id="tts.converted",
                        source_sample_rate=sample_rate,
                        playback_sample_rate=selected_rate,
                        source_channels=source_channels,
                        playback_channels=selected_channels,
                        resampler=resampler,
                    )
                    converter_logged = True
                if limiter_enabled:
                    playable_chunk, peak, gain = _limit_pcm16_audio(
                        playable_chunk,
                        limiter_threshold,
                    )
                    if gain < 0.999 and not limiter_logged:
                        log_continuous(
                            "tts",
                            "limiter",
                            log_id="tts.limiter",
                            peak=f"{peak:.3f}",
                            threshold=f"{float(limiter_threshold):.3f}",
                            gain=f"{gain:.3f}",
                        )
                        limiter_logged = True
                if stream is None:
                    stream = sd.RawOutputStream(
                        samplerate=selected_rate,
                        channels=selected_channels,
                        dtype="int16",
                        device=playback_device,
                    )
                    stream.start()
                if dump_manager is not None and dump_manager.voice_path_enabled:
                    if dump_start_ms is None:
                        dump_start_ms = dump_manager.elapsed_ms()
                    dump_chunks.append(playable_chunk)
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
        if dump_manager is not None and dump_start_ms is not None and dump_chunks:
            dump_manager.write_voice_path_dump(
                None,
                dump_kind,
                b"".join(dump_chunks),
                sample_rate=selected_rate,
                channels=selected_channels,
                start_ms=dump_start_ms,
                end_ms=dump_manager.elapsed_ms(),
            )
    return written_chunks


def _iter_pcm_chunks(
    audio_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    chunk_ms: int = 20,
) -> Iterator[bytes]:
    frame_bytes = max(2, channels * 2)
    chunk_bytes = max(frame_bytes, int(sample_rate * chunk_ms / 1000) * frame_bytes)
    chunk_bytes -= chunk_bytes % frame_bytes
    playable_len = len(audio_data) - (len(audio_data) % frame_bytes)
    for offset in range(0, playable_len, chunk_bytes):
        yield audio_data[offset : offset + chunk_bytes]


def _float_audio_to_pcm16(data) -> tuple[bytes, int]:
    import numpy as np  # type: ignore[import-untyped]

    channels = int(data.shape[1]) if len(data.shape) > 1 else 1
    clipped = np.clip(data, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    return pcm.reshape(-1).tobytes(), channels


def _limit_float_audio(data, threshold: float = 0.92):
    import numpy as np  # type: ignore[import-untyped]

    limit = _normalized_limiter_threshold(threshold)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= limit or peak <= 0.0:
        return data, peak, 1.0
    gain = limit / peak
    return data * gain, peak, gain


def _limit_pcm16_audio(pcm: bytes, threshold: float = 0.92) -> tuple[bytes, float, float]:
    if not pcm:
        return pcm, 0.0, 1.0

    playable_len = len(pcm) - (len(pcm) % 2)
    if playable_len <= 0:
        return pcm, 0.0, 1.0

    samples = array.array("h")
    samples.frombytes(pcm[:playable_len])
    if sys.byteorder != "little":
        samples.byteswap()
    peak_sample = max((abs(sample) for sample in samples), default=0)
    if peak_sample <= 0:
        return pcm, 0.0, 1.0

    limit = _normalized_limiter_threshold(threshold)
    limit_sample = max(1, int(round(32767 * limit)))
    peak = peak_sample / 32768.0
    if peak_sample <= limit_sample:
        return pcm, peak, 1.0

    gain = limit_sample / peak_sample
    for index, sample in enumerate(samples):
        samples[index] = max(-32768, min(32767, int(sample * gain)))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes() + pcm[playable_len:], peak, gain


def _normalized_limiter_threshold(threshold: float) -> float:
    return max(0.05, min(1.0, float(threshold)))


def _stop_requested(stop_event: threading.Event | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())
