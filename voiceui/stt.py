from __future__ import annotations

import tempfile
import wave
import json
import os
import uuid
from pathlib import Path
import urllib.error
import urllib.request

from voiceui.models import SttConfig, Utterance


class SpeechToText:
    def transcribe(self, utterance: Utterance) -> str:
        raise NotImplementedError


class MockSpeechToText(SpeechToText):
    def __init__(self, config: SttConfig):
        self.config = config

    def transcribe(self, utterance: Utterance) -> str:
        return self.config.mock_text


class FasterWhisperSpeechToText(SpeechToText):
    def __init__(self, config: SttConfig):
        self.config = config
        self._model = None

    def transcribe(self, utterance: Utterance) -> str:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install with: pip install -e \".[stt]\""
            ) from exc

        if self._model is None:
            self._model = WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "utterance.wav"
            _write_wav(wav_path, utterance)
            segments, _info = self._model.transcribe(
                str(wav_path),
                language=self.config.language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            return " ".join(segment.text.strip() for segment in segments).strip()


class OpenAICompatibleSpeechToText(SpeechToText):
    def __init__(self, config: SttConfig):
        self.config = config

    def transcribe(self, utterance: Utterance) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "utterance.wav"
            _write_wav(wav_path, utterance)
            fields = {"model": self.config.model}
            if self.config.language:
                fields["language"] = self.config.language
            headers = {}
            if self.config.api_key_env:
                api_key = os.environ.get(self.config.api_key_env)
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

            data = _post_multipart_json(
                self.config.endpoint,
                fields=fields,
                file_field="file",
                file_name="utterance.wav",
                file_content=wav_path.read_bytes(),
                file_content_type="audio/wav",
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
            return str(data.get("text", "")).strip()


def create_stt(config: SttConfig) -> SpeechToText:
    if config.provider == "mock":
        return MockSpeechToText(config)
    if config.provider == "faster_whisper":
        return FasterWhisperSpeechToText(config)
    if config.provider in ("openai_compatible", "mify"):
        return OpenAICompatibleSpeechToText(config)
    raise ValueError(f"Unsupported STT provider: {config.provider}")


def _write_wav(path: Path, utterance: Utterance) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(utterance.sample_rate)
        wav.writeframes(utterance.pcm)


def _post_multipart_json(
    url: str,
    fields: dict[str, str],
    file_field: str,
    file_name: str,
    file_content: bytes,
    file_content_type: str,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict:
    boundary = f"----voiceui-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'
            f"Content-Type: {file_content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(file_content)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"STT request failed: {url}: {exc}") from exc
