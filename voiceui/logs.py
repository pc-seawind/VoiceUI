from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

LogKind = Literal["event", "continuous"]


@dataclass(frozen=True, slots=True)
class LogSpec:
    log_id: str
    kind: LogKind
    module: str
    event: str
    default_enabled: bool = True
    description: str = ""


LOG_SPECS: tuple[LogSpec, ...] = (
    LogSpec("assistant.empty_input", "event", "assistant", "empty_input"),
    LogSpec("audio.first_chunk", "event", "audio", "first_chunk", False),
    LogSpec("audio.recorded", "event", "audio", "recorded"),
    LogSpec("audio.stream_opened", "event", "audio", "stream_opened", False),
    LogSpec("audio_dump.error", "event", "audio_dump", "error"),
    LogSpec("audio_dump.segment_saved", "event", "audio_dump", "segment_saved"),
    LogSpec("audio_dump.system_input_started", "event", "audio_dump", "system_input_started"),
    LogSpec("audio_dump.system_input_stopped", "event", "audio_dump", "system_input_stopped"),
    LogSpec("audio_dump.voice_path_saved", "event", "audio_dump", "voice_path_saved"),
    LogSpec("barge_in.captured", "event", "barge_in", "captured"),
    LogSpec("barge_in.config", "event", "barge_in", "config"),
    LogSpec("barge_in.error", "event", "barge_in", "error"),
    LogSpec("barge_in.monitor_saved", "event", "barge_in", "monitor_saved"),
    LogSpec("barge_in.monitor_started", "event", "barge_in", "monitor_started"),
    LogSpec("barge_in.no_speech", "event", "barge_in", "no_speech"),
    LogSpec("barge_in.speech_start", "event", "barge_in", "speech_start"),
    LogSpec("barge_in.timeout", "event", "barge_in", "timeout"),
    LogSpec("debug.saved", "event", "debug", "saved"),
    LogSpec("error.runtime", "event", "error", "runtime"),
    LogSpec("llm.completed", "event", "llm", "completed"),
    LogSpec("llm.first_token", "event", "llm", "first_token"),
    LogSpec("llm.stream_completed", "event", "llm", "stream_completed"),
    LogSpec("music.ducked", "event", "music", "ducked"),
    LogSpec("music.playback_error", "event", "music", "playback_error"),
    LogSpec("music.restored", "event", "music", "restored"),
    LogSpec("music.starting", "event", "music", "starting"),
    LogSpec("music.limiter", "continuous", "music", "limiter", False),
    LogSpec("search.baidu_ai_fallback", "event", "search", "baidu_ai_fallback"),
    LogSpec("search.completed", "event", "search", "completed"),
    LogSpec("session.empty_follow_up", "event", "session", "empty_follow_up"),
    LogSpec("session.follow_up_timeout", "event", "session", "follow_up_timeout"),
    LogSpec("session.listening_for_follow_up", "event", "session", "listening_for_follow_up"),
    LogSpec("session.processing_barge_in", "event", "session", "processing_barge_in"),
    LogSpec("stt.completed", "event", "stt", "completed"),
    LogSpec("stt.streaming_config", "event", "stt", "streaming_config", False),
    LogSpec("stt.streaming_started", "event", "stt", "streaming_started"),
    LogSpec("stt.transcribe_audio", "event", "stt", "transcribe_audio", False),
    LogSpec("tool.executed", "event", "tool", "executed"),
    LogSpec("tools.llm_round", "event", "tools", "llm_round"),
    LogSpec("tools.progress_prompt", "event", "tools", "progress_prompt"),
    LogSpec("tts.completed", "event", "tts", "completed"),
    LogSpec("tts.converted", "event", "tts", "converted"),
    LogSpec("tts.first_text_segment", "event", "tts", "first_text_segment", False),
    LogSpec("tts.first_text_sent", "event", "tts", "first_text_sent", False),
    LogSpec("tts.limiter", "continuous", "tts", "limiter", False),
    LogSpec("tts.playback_completed", "event", "tts", "playback_completed"),
    LogSpec("tts.stream_completed", "event", "tts", "stream_completed"),
    LogSpec("tts.stream_started", "event", "tts", "stream_started", False),
    LogSpec("tts.synthesis_completed", "event", "tts", "synthesis_completed"),
    LogSpec("vad.completed", "event", "vad", "completed"),
    LogSpec("vad.debug_start", "event", "vad", "debug_start", False),
    LogSpec("vad.debug_stop", "event", "vad", "debug_stop", False),
    LogSpec("vad.warm_up_error", "event", "vad", "warm_up_error"),
    LogSpec("vad.warmed_up", "event", "vad", "warmed_up"),
    LogSpec("wake.debug_enabled", "event", "wake", "debug_enabled", False),
    LogSpec("wake.detected", "event", "wake", "detected"),
    LogSpec("wake.detected_debug", "event", "wake", "detected_debug", False),
    LogSpec("wake.loading", "event", "wake", "loading"),
    LogSpec("wake.monitor_done", "event", "wake", "monitor_done"),
    LogSpec("wake.monitor_started", "event", "wake", "monitor_started"),
    LogSpec("wake.score", "continuous", "wake", "score", False),
    LogSpec("wake_ack.converted", "event", "wake_ack", "converted"),
    LogSpec("wake_ack.error", "event", "wake_ack", "error"),
    LogSpec("wake_ack.generated", "event", "wake_ack", "generated"),
    LogSpec("wake_ack.played", "event", "wake_ack", "played"),
    LogSpec("weather.warmed_up", "event", "weather", "warmed_up"),
    LogSpec("weather.warmup_error", "event", "weather", "warmup_error"),
)

_LOG_SPECS_BY_ID = {spec.log_id: spec for spec in LOG_SPECS}
_CONFIG: object | None = None
_TEXT_HIGHLIGHT_MODULES = {"asr", "stt", "tts"}
_TEXT_HIGHLIGHT_KEYS = ("text", "transcript", "reply")
_TEXT_RECORD_MODULES = {"asr", "stt", "tts", "llm"}
_OUTPUT_LOCK = threading.Lock()
_DEBUG_LOG_PATH: Path | None = None
_TEXT_RECORD_DIR: Path | None = None


def configure_logging(config: object | None) -> None:
    global _CONFIG
    _CONFIG = config


def configure_log_files(
    *,
    debug_log_path: str | Path | None = None,
    text_record_dir: str | Path | None = None,
) -> None:
    global _DEBUG_LOG_PATH, _TEXT_RECORD_DIR
    with _OUTPUT_LOCK:
        _DEBUG_LOG_PATH = Path(debug_log_path) if debug_log_path is not None else None
        _TEXT_RECORD_DIR = Path(text_record_dir) if text_record_dir is not None else None
        if _DEBUG_LOG_PATH is not None:
            _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _DEBUG_LOG_PATH.touch(exist_ok=True)
        if _TEXT_RECORD_DIR is not None:
            _TEXT_RECORD_DIR.mkdir(parents=True, exist_ok=True)


def reset_logging() -> None:
    configure_logging(None)
    configure_log_files()


def is_log_enabled(
    log_id: str,
    *,
    kind: LogKind | None = None,
    default_enabled: bool | None = None,
) -> bool:
    spec = _LOG_SPECS_BY_ID.get(log_id)
    resolved_kind = kind or (spec.kind if spec is not None else "event")
    config = _CONFIG
    if config is not None and not bool(getattr(config, "enabled", True)):
        return False
    overrides = _overrides_for_kind(config, resolved_kind)
    if isinstance(overrides, dict) and log_id in overrides:
        return bool(overrides[log_id])
    if default_enabled is not None:
        return bool(default_enabled)
    if spec is not None:
        return spec.default_enabled
    return resolved_kind == "event"


def log_event(
    module: str,
    event: str,
    *,
    log_id: str | None = None,
    default_enabled: bool | None = None,
    **params: Any,
) -> None:
    resolved_log_id = log_id or f"{module}.{event}"
    timestamp = datetime.now()
    _write_text_record_from_params(timestamp, module, event, params)
    if not is_log_enabled(resolved_log_id, kind="event", default_enabled=default_enabled):
        return
    _write_log(module, event, params, timestamp=timestamp)


def log_continuous(
    module: str,
    event: str,
    *,
    log_id: str | None = None,
    default_enabled: bool | None = None,
    **params: Any,
) -> None:
    resolved_log_id = log_id or f"{module}.{event}"
    if not is_log_enabled(resolved_log_id, kind="continuous", default_enabled=default_enabled):
        return
    _write_log(module, event, params, timestamp=datetime.now())


def record_text_event(
    module: str,
    event: str,
    text: str,
    **params: Any,
) -> None:
    _write_text_record(
        datetime.now(),
        module,
        event,
        text_key="text",
        text=text,
        params=params,
    )


def format_log(
    module: str,
    event: str,
    params: dict[str, Any] | None = None,
    *,
    timestamp: datetime | None = None,
) -> str:
    stamp = (timestamp or datetime.now()).isoformat(timespec="milliseconds")
    display_params = dict(params or {})
    highlighted = _pop_highlighted_text(module, display_params)
    line = f"{stamp} | module={module} | event={event} | params={_format_params(display_params)}"
    if highlighted is None:
        return line
    label, text = highlighted
    return f"{line}\n{_format_highlighted_text(label, text)}"


def log_switch_rows(config: object | None = None) -> list[dict[str, object]]:
    logging_config = getattr(config, "logging", config)
    rows: list[dict[str, object]] = []
    for spec in sorted(LOG_SPECS, key=lambda item: item.log_id):
        legacy_default = _legacy_default_enabled(spec.log_id, config)
        rows.append(
            {
                "id": spec.log_id,
                "kind": spec.kind,
                "module": spec.module,
                "event": spec.event,
                "default_enabled": (
                    legacy_default if legacy_default is not None else spec.default_enabled
                ),
                "enabled": _effective_enabled(spec, logging_config, legacy_default),
            }
        )
    return rows


def _effective_enabled(
    spec: LogSpec,
    config: object | None,
    default_enabled: bool | None = None,
) -> bool:
    if config is not None and not bool(getattr(config, "enabled", True)):
        return False
    overrides = _overrides_for_kind(config, spec.kind)
    if isinstance(overrides, dict) and spec.log_id in overrides:
        return bool(overrides[spec.log_id])
    return default_enabled if default_enabled is not None else spec.default_enabled


def _legacy_default_enabled(log_id: str, config: object | None) -> bool | None:
    if config is None or hasattr(config, "events"):
        return None
    if log_id in {"audio.first_chunk", "audio.stream_opened"}:
        return bool(getattr(getattr(config, "audio", None), "debug", False))
    if log_id in {"stt.streaming_config", "stt.transcribe_audio"}:
        return bool(getattr(getattr(config, "stt", None), "debug", False))
    if log_id in {"vad.debug_start", "vad.debug_stop"}:
        return bool(getattr(getattr(config, "vad", None), "debug", False))
    if log_id in {"wake.debug_enabled", "wake.detected_debug"}:
        return bool(getattr(getattr(config, "wake", None), "debug", False))
    return None


def _overrides_for_kind(config: object | None, kind: LogKind) -> object:
    if config is None:
        return {}
    field_name = "events" if kind == "event" else "continuous"
    return getattr(config, field_name, {})


def _write_log(
    module: str,
    event: str,
    params: dict[str, Any],
    *,
    timestamp: datetime,
) -> None:
    line = format_log(module, event, params, timestamp=timestamp)
    with _OUTPUT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        if _DEBUG_LOG_PATH is not None:
            _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return "-"
    return " ".join(f"{key}={_format_value(value)}" for key, value in params.items())


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    if _is_plain_value(text):
        return text
    return json.dumps(text, ensure_ascii=False)


def _is_plain_value(value: str) -> bool:
    if value == "":
        return False
    return all(ch not in value for ch in " \t\r\n|\"'")


def _pop_highlighted_text(module: str, params: dict[str, Any]) -> tuple[str, Any] | None:
    if module not in _TEXT_HIGHLIGHT_MODULES:
        return None
    for key in _TEXT_HIGHLIGHT_KEYS:
        if key in params:
            label = "ASR TEXT" if module in {"asr", "stt"} else "TTS TEXT"
            return label, params.pop(key)
    return None


def _format_highlighted_text(label: str, value: Any) -> str:
    text = "" if value is None else str(value)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        lines = ["<empty>"]
    lines = [line if line else "<empty>" for line in lines]
    rendered = [f"    >>> {label}: {lines[0]}"]
    rendered.extend(f"        {line}" for line in lines[1:])
    return "\n".join(rendered)


def _write_text_record_from_params(
    timestamp: datetime,
    module: str,
    event: str,
    params: dict[str, Any],
) -> None:
    if module not in _TEXT_RECORD_MODULES:
        return
    for key in _TEXT_HIGHLIGHT_KEYS:
        if key in params:
            _write_text_record(
                timestamp,
                module,
                event,
                text_key=key,
                text=params[key],
                params={name: value for name, value in params.items() if name != key},
            )
            return


def _write_text_record(
    timestamp: datetime,
    module: str,
    event: str,
    *,
    text_key: str,
    text: Any,
    params: dict[str, Any],
) -> None:
    if _TEXT_RECORD_DIR is None or module not in _TEXT_RECORD_MODULES:
        return
    record = {
        "timestamp": timestamp.isoformat(timespec="milliseconds"),
        "module": module,
        "event": event,
        "role": _text_record_role(module),
        "text_key": text_key,
        "text": "" if text is None else str(text),
        "params": _json_safe(params),
    }
    path = _TEXT_RECORD_DIR / f"voice_text_{timestamp.date().isoformat()}.jsonl"
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with _OUTPUT_LOCK:
        _TEXT_RECORD_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(encoded + "\n")


def _text_record_role(module: str) -> str:
    if module in {"asr", "stt"}:
        return "user"
    if module in {"llm", "tts"}:
        return "assistant"
    return module


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)
