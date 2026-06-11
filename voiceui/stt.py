from __future__ import annotations

import base64
import json
import math
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

from voiceui.aliyun import get_aliyun_nls_token as _get_aliyun_nls_token
from voiceui.http_utils import post_json, require_api_key
from voiceui.logs import log_event
from voiceui.models import SttConfig, Utterance

_ALIYUN_SAMPLE_RATE = 16000


class StreamingSpeechToTextSession:
    def write(self, pcm: bytes) -> None:
        raise NotImplementedError

    def finish(self) -> str:
        raise NotImplementedError

    def abort(self) -> None:
        raise NotImplementedError


class SpeechToText:
    def supports_streaming(self) -> bool:
        return False

    def start_streaming(self, sample_rate: int) -> StreamingSpeechToTextSession:
        raise RuntimeError(f"{type(self).__name__} does not support streaming STT.")

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

    def supports_streaming(self) -> bool:
        return True

    def start_streaming(self, sample_rate: int) -> StreamingSpeechToTextSession:
        if sample_rate != _ALIYUN_SAMPLE_RATE:
            raise RuntimeError(
                "Aliyun NLS streaming STT currently requires 16000 Hz audio, "
                f"got {sample_rate}."
            )
        app_key = require_api_key(self.config.app_key_env or "ALIYUN_NLS_APPKEY")
        token = self._token_or_create()
        leading_silence_ms = max(0, self.config.leading_silence_ms)
        log_event(
            "stt",
            "streaming_config",
            log_id="stt.streaming_config",
            default_enabled=self.config.debug,
            provider="aliyun_nls",
            mode="streaming",
            sent_sample_rate=_ALIYUN_SAMPLE_RATE,
            leading_silence_ms=leading_silence_ms,
        )
        return _AliyunNlsStreamingSession(
            url=self.config.endpoint,
            token=token,
            app_key=app_key,
            sample_rate=_ALIYUN_SAMPLE_RATE,
            timeout_seconds=self.config.timeout_seconds,
            leading_silence_ms=leading_silence_ms,
        )

    def transcribe(self, utterance: Utterance) -> str:
        app_key = require_api_key(self.config.app_key_env or "ALIYUN_NLS_APPKEY")
        token = self._token_or_create()

        sample_rate = _ALIYUN_SAMPLE_RATE
        pcm = _ensure_pcm16_sample_rate(utterance.pcm, utterance.sample_rate, sample_rate)
        leading_silence_ms = max(0, self.config.leading_silence_ms)
        if leading_silence_ms:
            pcm = _prepend_pcm16_silence(
                pcm,
                sample_rate=sample_rate,
                silence_ms=leading_silence_ms,
            )
        sent_audio_ms = int(len(pcm) / 2 / sample_rate * 1000)
        log_event(
            "stt",
            "transcribe_audio",
            log_id="stt.transcribe_audio",
            default_enabled=self.config.debug,
            provider="aliyun_nls",
            utterance_duration_ms=utterance.duration_ms,
            utterance_sample_rate=utterance.sample_rate,
            sent_sample_rate=sample_rate,
            leading_silence_ms=leading_silence_ms,
            sent_audio_ms=sent_audio_ms,
            sent_bytes=len(pcm),
        )
        return _run_aliyun_speech_recognizer(
            url=self.config.endpoint,
            token=token,
            app_key=app_key,
            pcm=pcm,
            sample_rate=sample_rate,
            timeout_seconds=self.config.timeout_seconds,
        ).strip()

    def _token_or_create(self) -> str:
        access_key_id = require_api_key(
            self.config.access_key_id_env or "ALIYUN_AccessKeyId"
        )
        access_key_secret = require_api_key(
            self.config.access_key_secret_env or "ALIYUN_AccessKeySecret"
        )
        if self._token is None:
            self._token = _get_aliyun_nls_token(access_key_id, access_key_secret)
        return self._token


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
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'
            f"Content-Type: {file_content_type}\r\n\r\n"
        ).encode()
    )
    body.extend(file_content)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

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


class _AliyunNlsStreamingSession(StreamingSpeechToTextSession):
    def __init__(
        self,
        *,
        url: str,
        token: str,
        app_key: str,
        sample_rate: int,
        timeout_seconds: float,
        leading_silence_ms: int,
    ):
        try:
            import nls  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Aliyun NLS SDK is not installed. Install with: "
                "pip install git+https://github.com/aliyun/alibabacloud-nls-python-sdk.git"
            ) from exc

        self.sample_rate = sample_rate
        self.timeout_seconds = timeout_seconds
        self.frame_bytes = max(2, int(sample_rate * 2 * 0.02))
        self.results: list[str] = []
        self.errors: list[str] = []
        self.sent_bytes = 0
        self._pending = bytearray()
        self._closed = False

        def on_completed(message: str, *_args: object) -> None:
            text = _extract_aliyun_result(message)
            if text:
                self.results.append(text)

        def on_error(message: str, *_args: object) -> None:
            self.errors.append(message)

        self.recognizer = nls.NlsSpeechRecognizer(
            url=url,
            token=token,
            appkey=app_key,
            on_completed=on_completed,
            on_error=on_error,
            callback_args=[],
        )
        start_result = self.recognizer.start(
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
            self.recognizer.shutdown()
            self._closed = True
            raise RuntimeError("Aliyun NLS recognizer failed to start.")

        if leading_silence_ms > 0:
            self.write(_prepend_pcm16_silence(b"", sample_rate, leading_silence_ms))

    def write(self, pcm: bytes) -> None:
        if self._closed:
            raise RuntimeError("Aliyun NLS streaming session is already closed.")
        if not pcm:
            return
        self._pending.extend(pcm)
        while len(self._pending) >= self.frame_bytes:
            chunk = bytes(self._pending[: self.frame_bytes])
            del self._pending[: self.frame_bytes]
            self.recognizer.send_audio(chunk)
            self.sent_bytes += len(chunk)
            # Aliyun NLS streaming recognition expects audio to be fed at a
            # near-real-time pace. In VoiceUI the recognizer may spend time
            # creating a token/opening the websocket while VAD has already
            # queued speech audio; without pacing that backlog is flushed too
            # quickly and NLS can time out waiting for a final recognition
            # result. Match the non-streaming recognizer's 20 ms frame pacing.
            time.sleep(0.01)

    def finish(self) -> str:
        if self._closed:
            return self.results[-1] if self.results else ""
        try:
            if self._pending:
                chunk = bytes(self._pending)
                self._pending.clear()
                self.recognizer.send_audio(chunk)
                self.sent_bytes += len(chunk)
            self.recognizer.stop(timeout=max(1, math.ceil(self.timeout_seconds)))
        finally:
            self.recognizer.shutdown()
            self._closed = True

        log_event(
            "stt",
            "streaming_finish",
            log_id="stt.streaming_finish",
            default_enabled=True,
            sent_bytes=self.sent_bytes,
            sent_audio_ms=int(self.sent_bytes / 2 / self.sample_rate * 1000),
            results=len(self.results),
            errors=len(self.errors),
        )
        if self.errors:
            raise RuntimeError(f"Aliyun NLS STT failed: {self.errors[-1]}")
        return self.results[-1] if self.results else ""

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self.recognizer.shutdown()
        finally:
            self._closed = True


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
