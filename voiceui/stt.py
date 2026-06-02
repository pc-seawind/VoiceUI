from __future__ import annotations

import base64
import json
import math
import tempfile
import time
import wave
import uuid
from pathlib import Path
import urllib.error
import urllib.request

from voiceui.aliyun import get_aliyun_nls_token as _get_aliyun_nls_token
from voiceui.http_utils import post_json, require_api_key
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
                api_key = require_api_key(self.config.api_key_env)
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


class MimoAudioUnderstandingSpeechToText(SpeechToText):
    def __init__(self, config: SttConfig):
        self.config = config

    def transcribe(self, utterance: Utterance) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "utterance.wav"
            _write_wav(wav_path, utterance)
            audio_b64 = base64.b64encode(wav_path.read_bytes()).decode("ascii")

        headers = {}
        if self.config.api_key_env:
            api_key = require_api_key(self.config.api_key_env)
            headers["api-key"] = api_key

        prompt = (
            "请将这段音频逐字转写为简体中文文本。"
            "只输出转写文本，不要解释，不要添加标点之外的额外内容。"
        )
        if self.config.language and self.config.language.lower() not in ("zh", "zh-cn", "chinese"):
            prompt = (
                f"Transcribe this audio in {self.config.language}. "
                "Only output the transcript text."
            )

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise speech transcription assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/wav;base64,{audio_b64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_completion_tokens": 1024,
        }
        data = _post_json(
            _chat_completions_url(self.config.endpoint),
            payload,
            headers=headers,
            timeout=self.config.timeout_seconds,
        )
        return _extract_chat_message_text(data)


class AliyunNlsSpeechToText(SpeechToText):
    def __init__(self, config: SttConfig):
        self.config = config
        self._token: str | None = None

    def transcribe(self, utterance: Utterance) -> str:
        app_key = require_api_key(self.config.app_key_env or "ALIYUN_NLS_APPKEY")
        access_key_id = require_api_key(
            self.config.access_key_id_env or "ALIYUN_AccessKeyId"
        )
        access_key_secret = require_api_key(
            self.config.access_key_secret_env or "ALIYUN_AccessKeySecret"
        )
        if self._token is None:
            self._token = _get_aliyun_nls_token(access_key_id, access_key_secret)

        sample_rate = 16000
        pcm = _ensure_pcm16_sample_rate(utterance.pcm, utterance.sample_rate, sample_rate)
        leading_silence_ms = max(0, self.config.leading_silence_ms)
        if leading_silence_ms:
            pcm = _prepend_pcm16_silence(
                pcm,
                sample_rate=sample_rate,
                silence_ms=leading_silence_ms,
            )
        if self.config.debug:
            sent_audio_ms = int(len(pcm) / 2 / sample_rate * 1000)
            print(
                "stt_debug> "
                f"provider=aliyun_nls utterance_duration_ms={utterance.duration_ms} "
                f"utterance_sample_rate={utterance.sample_rate} sent_sample_rate={sample_rate} "
                f"leading_silence_ms={leading_silence_ms} sent_audio_ms={sent_audio_ms} "
                f"sent_bytes={len(pcm)}"
            )
        return _run_aliyun_speech_recognizer(
            url=self.config.endpoint,
            token=self._token,
            app_key=app_key,
            pcm=pcm,
            sample_rate=sample_rate,
            timeout_seconds=self.config.timeout_seconds,
        ).strip()


def create_stt(config: SttConfig) -> SpeechToText:
    if config.provider == "mock":
        return MockSpeechToText(config)
    if config.provider == "faster_whisper":
        return FasterWhisperSpeechToText(config)
    if config.provider == "openai_compatible":
        return OpenAICompatibleSpeechToText(config)
    if config.provider in ("mify", "mimo"):
        return MimoAudioUnderstandingSpeechToText(config)
    if config.provider == "aliyun_nls":
        return AliyunNlsSpeechToText(config)
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
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"STT request failed: {url}: HTTP {exc.code}: {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"STT request failed: {url}: {exc}") from exc


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
        error_prefix="STT request failed",
    )


def _chat_completions_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_chat_message_text(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = str(message.get("content") or "").strip()
    if content:
        return content
    return str(message.get("reasoning_content") or "").strip()


def _run_aliyun_speech_recognizer(
    *,
    url: str,
    token: str,
    app_key: str,
    pcm: bytes,
    sample_rate: int,
    timeout_seconds: float,
) -> str:
    try:
        import nls  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Aliyun NLS SDK is not installed. Install with: "
            "pip install git+https://github.com/aliyun/alibabacloud-nls-python-sdk.git"
        ) from exc

    results: list[str] = []
    errors: list[str] = []

    def on_completed(message: str, *_args: object) -> None:
        text = _extract_aliyun_result(message)
        if text:
            results.append(text)

    def on_error(message: str, *_args: object) -> None:
        errors.append(message)

    recognizer = nls.NlsSpeechRecognizer(
        url=url,
        token=token,
        appkey=app_key,
        on_completed=on_completed,
        on_error=on_error,
        callback_args=[],
    )
    try:
        start_result = recognizer.start(
            aformat="pcm",
            sample_rate=sample_rate,
            ch=1,
            enable_intermediate_result=False,
            enable_punctuation_prediction=True,
            enable_inverse_text_normalization=True,
            timeout=max(1, math.ceil(timeout_seconds)),
            ping_interval=8,
            ping_timeout=None,
        )
        if start_result is False:
            raise RuntimeError("Aliyun NLS recognizer failed to start.")
        frame_bytes = max(2, int(sample_rate * 2 * 0.02))
        for index in range(0, len(pcm), frame_bytes):
            chunk = pcm[index : index + frame_bytes]
            if chunk:
                recognizer.send_audio(chunk)
                time.sleep(0.01)
        recognizer.stop(timeout=max(1, math.ceil(timeout_seconds)))
    finally:
        recognizer.shutdown()

    if errors:
        raise RuntimeError(f"Aliyun NLS STT failed: {errors[-1]}")
    return results[-1] if results else ""


def _extract_aliyun_result(message: str) -> str:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return ""
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("result") or "").strip()


def _ensure_pcm16_sample_rate(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate:
        return pcm
    try:
        import audioop
    except ImportError as exc:
        raise RuntimeError("Audio resampling requires the Python audioop module.") from exc
    converted, _state = audioop.ratecv(pcm, 2, 1, source_rate, target_rate, None)
    return converted


def _prepend_pcm16_silence(pcm: bytes, sample_rate: int, silence_ms: int) -> bytes:
    if silence_ms <= 0:
        return pcm
    samples = int(sample_rate * silence_ms / 1000)
    return b"\x00\x00" * samples + pcm
