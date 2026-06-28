from __future__ import annotations

# ruff: noqa: E501 - embedded single-file HTML/CSS/JS intentionally keeps long lines.
import argparse
import errno
import json
import mimetypes
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from voiceui.config import AUTO_CONFIG, load_config
from voiceui.core import VoiceAssistant
from voiceui.env import load_dotenv
from voiceui.logs import (
    configure_logging,
    log_event,
    record_text_event,
    reset_logging,
)
from voiceui.models import AssistantReply

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765
_MAX_BODY_BYTES = 64 * 1024
_SSE_POLL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class WebConsoleSettings:
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    title: str = "VoiceUI Web Console"


class VoiceUiWebConsole:
    def __init__(
        self,
        assistant: object | None,
        *,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        debug_output_dir: str | Path | None = None,
        title: str = "VoiceUI Web Console",
    ):
        self.assistant = assistant
        self.settings = WebConsoleSettings(host=host, port=int(port), title=title)
        assistant_debug = getattr(getattr(assistant, "config", None), "debug", None)
        self.debug_output_dir = Path(
            debug_output_dir
            or (assistant_debug.output_dir if assistant_debug is not None else "debug_sessions")
        )
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        server = self._server
        if server is None:
            return f"http://{self.settings.host}:{self.settings.port}/"
        host, port = server.server_address[:2]
        display_host = "127.0.0.1" if host in {"", "0.0.0.0"} else str(host)
        return f"http://{display_host}:{port}/"

    def start(self) -> None:
        if self._server is not None:
            return

        owner = self

        class Handler(_VoiceUiWebHandler):
            console = owner

        try:
            server = ThreadingHTTPServer((self.settings.host, self.settings.port), Handler)
        except OSError as exc:
            log_event(
                "web",
                "error",
                log_id="web.error",
                host=self.settings.host,
                port=self.settings.port,
                error=str(exc),
            )
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    "VoiceUI web console port is already in use: "
                    f"{self.settings.host}:{self.settings.port}. "
                    "Stop the existing service or pass --web-port to use another port."
                ) from exc
            raise
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="voiceui-web-console",
            daemon=True,
        )
        self._thread.start()
        log_event("web", "started", log_id="web.started", url=self.url)

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        try:
            server.shutdown()
        finally:
            server.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        log_event("web", "stopped", log_id="web.stopped")

    def status(self) -> dict[str, Any]:
        current = self.current_session_dir(create=False)
        latest = latest_debug_session(self.debug_output_dir)
        session = current or latest
        return {
            "ok": True,
            "title": self.settings.title,
            "url": self.url,
            "input_enabled": self.assistant is not None,
            "debug_output_dir": str(self.debug_output_dir),
            "current_session": current.name if current is not None else None,
            "latest_session": latest.name if latest is not None else None,
            "session": session.name if session is not None else None,
            "text_record_dir": str(self.text_record_dir()),
        }

    def current_session_dir(self, *, create: bool) -> Path | None:
        assistant = self.assistant
        manager = getattr(assistant, "audio_dump", None)
        get_dir = getattr(manager, "debug_session_dir", None)
        if callable(get_dir):
            if create:
                return get_dir()
            # Avoid creating a brand-new session just because someone opened the page.
            return getattr(manager, "_session_dir", None)
        return None

    def resolve_session_dir(self, name: str | None) -> Path | None:
        if name in {None, "", "current"}:
            return self.current_session_dir(create=False) or latest_debug_session(
                self.debug_output_dir
            )
        if name == "latest":
            return latest_debug_session(self.debug_output_dir)
        safe_name = Path(name).name
        if safe_name != name:
            return None
        candidate = self.debug_output_dir / safe_name
        try:
            candidate.relative_to(self.debug_output_dir)
        except ValueError:
            return None
        if not candidate.is_dir() or candidate.name == "text_records":
            return None
        return candidate

    def text_record_dir(self) -> Path:
        assistant = self.assistant
        manager = getattr(assistant, "audio_dump", None)
        get_dir = getattr(manager, "text_record_dir", None)
        if callable(get_dir):
            path = get_dir()
            if path is not None:
                return Path(path)
        return self.debug_output_dir / "text_records"

    def logs(self, *, session: str | None = None, tail: int = 500) -> dict[str, Any]:
        path = self.resolve_log_path(session)
        session_dir = self.resolve_session_dir(session) if session not in {None, ""} else None
        lines = tail_lines(path, max_lines=tail) if path is not None else []
        return {
            "session": session_dir.name if session_dir is not None else None,
            "path": str(path) if path is not None else None,
            "lines": lines,
        }

    def resolve_log_path(self, session: str | None) -> Path | None:
        if session in {None, "", "current"}:
            assistant = self.assistant
            manager = getattr(assistant, "audio_dump", None)
            get_path = getattr(manager, "debug_log_path", None)
            if callable(get_path):
                path = get_path()
                if path is not None:
                    return Path(path)
        session_dir = self.resolve_session_dir(session)
        return session_dir / "debug.log" if session_dir is not None else None

    def conversation(self, *, limit: int = 200) -> dict[str, Any]:
        records = read_text_records(self.text_record_dir(), limit=limit)
        return {"records": records}

    def debug_sessions(self) -> dict[str, Any]:
        latest = latest_debug_session(self.debug_output_dir)
        sessions: list[dict[str, Any]] = []
        if self.debug_output_dir.is_dir():
            for path in sorted(self.debug_output_dir.iterdir(), reverse=True):
                if not path.is_dir() or path.name == "text_records":
                    continue
                sessions.append(debug_session_summary(path, latest=latest))
        return {"sessions": sessions}

    def debug_session(self, name: str | None) -> dict[str, Any]:
        path = self.resolve_session_dir(name)
        if path is None:
            raise FileNotFoundError(name or "")
        metadata = read_json_file(path / "metadata.json")
        audio_dir = path / "audio_dumps"
        audio_files = []
        if audio_dir.is_dir():
            for audio_path in sorted(audio_dir.iterdir()):
                if audio_path.is_file():
                    audio_files.append(
                        {
                            "name": audio_path.name,
                            "size": audio_path.stat().st_size,
                            "url": f"debug/audio/{path.name}/{audio_path.name}",
                        }
                    )
        return {
            "session": path.name,
            "path": str(path),
            "metadata": metadata,
            "audio_files": audio_files,
            "log_path": str(path / "debug.log"),
        }

    def audio_file(self, session: str, filename: str) -> Path:
        session_dir = self.resolve_session_dir(session)
        if session_dir is None:
            raise FileNotFoundError(session)
        safe_name = Path(filename).name
        if safe_name != filename:
            raise FileNotFoundError(filename)
        path = session_dir / "audio_dumps" / safe_name
        try:
            path.relative_to(session_dir / "audio_dumps")
        except ValueError as exc:
            raise FileNotFoundError(filename) from exc
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def submit_text(self, text: str) -> dict[str, Any]:
        assistant = self.assistant
        if assistant is None:
            return {"ok": False, "error": "web input is not attached to a running assistant"}
        transcript = text.strip()
        if not transcript:
            return {"ok": False, "error": "empty text"}
        log_event("web", "text_received", log_id="web.text_received", chars=len(transcript))
        record_text_event("stt", "completed", transcript, source="web")
        try:
            reply = assistant.run_text_turn(transcript)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_event("web", "error", log_id="web.error", stage="text_turn", error=exc)
            return {"ok": False, "error": str(exc)}
        if not isinstance(reply, AssistantReply):
            reply_text = str(getattr(reply, "text", ""))
            routed_to = str(getattr(reply, "routed_to", ""))
        else:
            reply_text = reply.text
            routed_to = reply.routed_to
        log_event(
            "web",
            "turn_completed",
            log_id="web.turn_completed",
            routed_to=routed_to or "unknown",
            reply_chars=len(reply_text),
        )
        return {"ok": True, "reply": reply_text, "routed_to": routed_to}

    def stream_events(self, handler: BaseHTTPRequestHandler, session: str | None) -> None:
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()

        log_path = self.resolve_log_path(session)
        log_offset = log_path.stat().st_size if log_path is not None and log_path.exists() else 0
        text_offsets = {
            path: path.stat().st_size for path in sorted(self.text_record_dir().glob("*.jsonl"))
        }
        while True:
            try:
                current_log_path = self.resolve_log_path(session)
                if current_log_path != log_path:
                    log_path = current_log_path
                    log_offset = 0
                if log_path is not None:
                    new_lines, log_offset = read_new_lines(log_path, log_offset)
                    if new_lines:
                        _send_sse(handler, "logs", {"lines": new_lines})

                new_records: list[dict[str, Any]] = []
                for path in sorted(self.text_record_dir().glob("*.jsonl")):
                    offset = text_offsets.get(path, 0)
                    lines, offset = read_new_lines(path, offset)
                    text_offsets[path] = offset
                    for line in lines:
                        record = parse_text_record(line)
                        if record is not None and is_conversation_record(record):
                            new_records.append(record)
                if new_records:
                    _send_sse(handler, "conversation", {"records": new_records})
                _send_sse(handler, "heartbeat", {"time": datetime.now().isoformat()})
                time.sleep(_SSE_POLL_SECONDS)
            except (BrokenPipeError, ConnectionResetError):
                return


class _VoiceUiWebHandler(BaseHTTPRequestHandler):
    console: VoiceUiWebConsole

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/_rpc/"):
            path = "/api/" + path.removeprefix("/_rpc/")
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self._send_html(INDEX_HTML)
            elif path == "/api/status":
                self._send_json(self.console.status())
            elif path == "/api/logs":
                self._send_json(
                    self.console.logs(
                        session=_query_one(query, "session"),
                        tail=_query_int(query, "tail", 500, minimum=1, maximum=5000),
                    )
                )
            elif path == "/api/conversation":
                self._send_json(
                    self.console.conversation(
                        limit=_query_int(query, "limit", 200, minimum=1, maximum=5000)
                    )
                )
            elif path == "/api/debug/sessions":
                self._send_json(self.console.debug_sessions())
            elif path == "/api/debug/session":
                self._send_json(self.console.debug_session(_query_one(query, "name")))
            elif path == "/api/events":
                self.console.stream_events(self, _query_one(query, "session"))
            elif path.startswith("/debug/audio/"):
                _, _, rest = path.partition("/debug/audio/")
                session, _, filename = rest.partition("/")
                self._send_file(self.console.audio_file(unquote(session), unquote(filename)))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_event("web", "error", log_id="web.error", stage="GET", error=exc)
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/_rpc/"):
            path = "/api/" + path.removeprefix("/_rpc/")
        try:
            if path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length > _MAX_BODY_BYTES:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            text = str(payload.get("text") or "")
            result = self.console.submit_text(text)
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
            self._send_json(result, status=status)
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_event("web", "error", log_id="web.error", stage="POST", error=exc)
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html: str) -> None:
        encoded = html.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)


def start_web_console(
    assistant: object | None,
    *,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    debug_output_dir: str | Path | None = None,
    title: str = "VoiceUI Web Console",
) -> VoiceUiWebConsole:
    console = VoiceUiWebConsole(
        assistant,
        host=host,
        port=port,
        debug_output_dir=debug_output_dir,
        title=title,
    )
    console.start()
    return console


def latest_debug_session(root: str | Path) -> Path | None:
    root_path = Path(root)
    if not root_path.is_dir():
        return None
    candidates = [
        path
        for path in root_path.iterdir()
        if path.is_dir() and path.name != "text_records" and not path.name.startswith(".")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def tail_lines(path: str | Path, *, max_lines: int) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return []
    # Debug logs are usually small per run; keep implementation simple and robust.
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]


def read_new_lines(path: str | Path, offset: int) -> tuple[list[str], int]:
    path = Path(path)
    if not path.is_file():
        return [], 0
    size = path.stat().st_size
    if offset > size:
        offset = 0
    with path.open("rb") as file:
        file.seek(offset)
        data = file.read()
        new_offset = file.tell()
    if not data:
        return [], new_offset
    return data.decode("utf-8", errors="replace").splitlines(), new_offset


def read_text_records(directory: str | Path, *, limit: int) -> list[dict[str, Any]]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            record = parse_text_record(line)
            if record is not None and is_conversation_record(record):
                records.append(record)
    records.sort(key=lambda item: str(item.get("timestamp") or ""))
    return records[-limit:]


def parse_text_record(line: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    role = payload.get("role")
    text = payload.get("text")
    if not isinstance(role, str) or not isinstance(text, str):
        return None
    return payload


def is_conversation_record(record: dict[str, Any]) -> bool:
    module = str(record.get("module") or "")
    # TTS repeats the assistant text already recorded by llm.completed in normal turns.
    return module in {"asr", "stt", "llm"}


def debug_session_summary(path: Path, *, latest: Path | None) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    log_path = path / "debug.log"
    audio_dir = path / "audio_dumps"
    audio_count = len([item for item in audio_dir.iterdir() if item.is_file()]) if audio_dir.is_dir() else 0
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "latest": latest == path,
        "mtime": stat.st_mtime,
        "metadata": metadata_path.exists(),
        "debug_log": log_path.exists(),
        "debug_log_size": log_path.stat().st_size if log_path.exists() else 0,
        "audio_count": audio_count,
    }


def read_json_file(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _send_sse(handler: BaseHTTPRequestHandler, event: str, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    handler.wfile.write(f"event: {event}\n".encode())
    for line in encoded.splitlines() or [""]:
        handler.wfile.write(f"data: {line}\n".encode())
    handler.wfile.write(b"\n")
    handler.wfile.flush()


def _query_one(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def _query_int(
    query: dict[str, list[str]],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = _query_one(query, key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return min(maximum, max(minimum, parsed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoiceUI web console")
    parser.add_argument("--config", default=AUTO_CONFIG, help="VoiceUI config path or 'auto'")
    parser.add_argument("--host", default=_DEFAULT_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="Bind port")
    parser.add_argument(
        "--viewer-only",
        action="store_true",
        help="Do not create an assistant; only view existing debug_sessions logs",
    )
    parser.add_argument("--output-dir", help="Debug output dir in --viewer-only mode")
    args = parser.parse_args(argv)

    assistant: VoiceAssistant | None = None
    console: VoiceUiWebConsole | None = None
    try:
        load_dotenv()
        if args.viewer_only:
            console = start_web_console(
                None,
                host=args.host,
                port=args.port,
                debug_output_dir=args.output_dir or "debug_sessions",
            )
        else:
            config = load_config(args.config)
            config.input.mode = "text"
            config.debug.enabled = True
            if args.output_dir:
                config.debug.output_dir = args.output_dir
            configure_logging(config.logging)
            assistant = VoiceAssistant(config)
            console = start_web_console(assistant, host=args.host, port=args.port)
        print(f"VoiceUI web console: {console.url}")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log_event("web", "error", log_id="web.error", stage="main", error=exc)
        return 2
    finally:
        if console is not None:
            console.stop()
        if assistant is not None:
            assistant.close()
        reset_logging()


INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VoiceUI 控制台</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2e; --muted:#8ea0bd; --text:#edf2ff; --accent:#6ee7ff; --danger:#ff7a90; --ok:#9cffb1; --border:#26334f; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--text); }
    header { display:flex; gap:16px; align-items:center; padding:14px 18px; border-bottom:1px solid var(--border); background:#0e1628; position:sticky; top:0; z-index:1; }
    h1 { font-size:18px; margin:0; }
    .status { color:var(--muted); font-size:12px; }
    main { display:grid; grid-template-columns: minmax(320px, 0.9fr) minmax(420px, 1.1fr); gap:12px; padding:12px; height:calc(100vh - 58px); }
    section { background:var(--panel); border:1px solid var(--border); border-radius:12px; min-height:0; display:flex; flex-direction:column; overflow:hidden; }
    .tabs { display:flex; gap:6px; padding:8px; border-bottom:1px solid var(--border); }
    .tab, button { border:1px solid var(--border); background:#17223a; color:var(--text); border-radius:8px; padding:7px 10px; cursor:pointer; }
    .tab.active, button.primary { border-color:var(--accent); color:#001018; background:var(--accent); }
    .pane { display:none; min-height:0; flex:1; overflow:auto; padding:12px; }
    .pane.active { display:block; }
    #chat { display:flex; flex-direction:column; min-height:0; }
    #messages { flex:1; overflow:auto; padding:12px; display:flex; flex-direction:column; gap:10px; }
    .msg { max-width:88%; padding:9px 11px; border:1px solid var(--border); border-radius:12px; white-space:pre-wrap; }
    .user { align-self:flex-end; background:#19395b; }
    .assistant { align-self:flex-start; background:#18243c; }
    .meta { color:var(--muted); font-size:11px; margin-bottom:4px; }
    form { display:flex; gap:8px; padding:10px; border-top:1px solid var(--border); }
    textarea { flex:1; min-height:42px; max-height:150px; resize:vertical; background:#091225; color:var(--text); border:1px solid var(--border); border-radius:10px; padding:9px; }
    pre { margin:0; white-space:pre-wrap; word-break:break-word; font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }
    .log-line { padding:2px 0; border-bottom:1px solid rgba(255,255,255,.03); }
    .log-line.error { color:var(--danger); }
    .tools { display:flex; gap:8px; align-items:center; margin-bottom:10px; flex-wrap:wrap; }
    input, select { background:#091225; color:var(--text); border:1px solid var(--border); border-radius:8px; padding:7px; }
    table { border-collapse:collapse; width:100%; }
    th, td { text-align:left; border-bottom:1px solid var(--border); padding:7px; vertical-align:top; }
    a { color:var(--accent); }
    .pill { display:inline-block; padding:1px 6px; border-radius:999px; background:#223252; color:var(--muted); font-size:11px; }
    @media (max-width: 900px) { main { grid-template-columns:1fr; height:auto; } section { min-height:60vh; } }
  </style>
</head>
<body>
<header>
  <h1>VoiceUI 控制台</h1>
  <div class="status" id="status">连接中…</div>
</header>
<main>
  <section id="chat">
    <div class="tabs"><button class="tab active" data-pane="conversation">对话</button><button class="tab" data-pane="debug">Debug</button></div>
    <div id="conversation" class="pane active" style="display:flex; flex-direction:column; padding:0;">
      <div id="messages"></div>
      <form id="chat-form">
        <textarea id="chat-input" placeholder="输入文字，作为语音之外的第二输入源。Enter 发送，Shift+Enter 换行"></textarea>
        <button class="primary" type="submit">发送</button>
      </form>
    </div>
    <div id="debug" class="pane">
      <div class="tools"><button id="refresh-debug">刷新 Debug</button><select id="session-select"></select></div>
      <div id="debug-detail"></div>
    </div>
  </section>
  <section>
    <div class="tabs"><button class="tab active" data-pane="logs">日志</button></div>
    <div id="logs" class="pane active">
      <div class="tools"><button id="refresh-logs">刷新日志</button><label><input id="autoscroll" type="checkbox" checked /> 自动滚动</label><input id="log-filter" placeholder="过滤 module/event/text" /></div>
      <pre id="log-lines"></pre>
    </div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let logLines = [];
let records = [];
function esc(s){return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function roleLabel(r){ return r.role === 'user' ? '你' : '助手'; }
function renderConversation(){
  const box = $('messages');
  box.innerHTML = records.map(r => `<div class="msg ${r.role === 'user' ? 'user' : 'assistant'}"><div class="meta">${esc(roleLabel(r))} · ${esc(r.timestamp || '')}</div>${esc(r.text)}</div>`).join('');
  box.scrollTop = box.scrollHeight;
}
function renderLogs(){
  const filter = $('log-filter').value.trim().toLowerCase();
  const shown = filter ? logLines.filter(l => l.toLowerCase().includes(filter)) : logLines;
  $('log-lines').innerHTML = shown.map(l => `<div class="log-line ${/module=error|event=.*_error|event=error/.test(l) ? 'error' : ''}">${esc(l)}</div>`).join('');
  if ($('autoscroll').checked) $('logs').scrollTop = $('logs').scrollHeight;
}
async function loadStatus(){
  const s = await fetch('_rpc/status').then(r=>r.json());
  $('status').textContent = `session=${s.session || '-'} · input=${s.input_enabled ? 'enabled' : 'viewer-only'} · ${s.url}`;
}
async function loadConversation(){ records = (await fetch('_rpc/conversation?limit=300').then(r=>r.json())).records || []; renderConversation(); }
async function loadLogs(){ logLines = (await fetch('_rpc/logs?tail=800').then(r=>r.json())).lines || []; renderLogs(); }
async function loadDebug(){
  const data = await fetch('_rpc/debug/sessions').then(r=>r.json());
  const sel = $('session-select');
  sel.innerHTML = (data.sessions || []).map(s => `<option value="${esc(s.name)}">${esc(s.name)} ${s.latest ? '(latest)' : ''}</option>`).join('');
  if (sel.value) await loadDebugDetail(sel.value);
}
async function loadDebugDetail(name){
  const d = await fetch('_rpc/debug/session?name=' + encodeURIComponent(name)).then(r=>r.json());
  const turns = d.metadata?.turns || [];
  const audios = d.audio_files || [];
  $('debug-detail').innerHTML = `<p><span class="pill">${esc(d.path)}</span></p>` +
    `<h3>Turns</h3><table><tr><th>#</th><th>用户</th><th>助手</th><th>timings</th></tr>${turns.map(t=>`<tr><td>${esc(t.turn)}</td><td>${esc(t.transcript||'')}</td><td>${esc(t.reply||'')}</td><td><pre>${esc(JSON.stringify(t.timings_ms||{}, null, 2))}</pre></td></tr>`).join('')}</table>` +
    `<h3>Audio dumps</h3><ul>${audios.map(a=>`<li><a href="${esc(a.url)}" target="_blank">${esc(a.name)}</a> <span class="pill">${a.size} bytes</span></li>`).join('')}</ul>` +
    `<h3>metadata.json</h3><pre>${esc(JSON.stringify(d.metadata, null, 2))}</pre>`;
}
document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {
  const parent = btn.closest('section');
  parent.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  parent.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));
  $(btn.dataset.pane).classList.add('active');
}));
$('chat-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = $('chat-input').value.trim();
  if (!text) return;
  $('chat-input').value = '';
  records.push({role:'user', text, timestamp:new Date().toISOString()}); renderConversation();
  const res = await fetch('_rpc/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})}).then(r=>r.json());
  if (res.ok) records.push({role:'assistant', text:res.reply || '', timestamp:new Date().toISOString()});
  else records.push({role:'assistant', text:'发送失败：' + (res.error || 'unknown'), timestamp:new Date().toISOString()});
  renderConversation(); setTimeout(loadConversation, 500); setTimeout(loadLogs, 500);
});
$('chat-input').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('chat-form').requestSubmit(); }});
$('refresh-logs').onclick = loadLogs; $('refresh-debug').onclick = loadDebug; $('log-filter').oninput = renderLogs; $('session-select').onchange = e => loadDebugDetail(e.target.value);
function connectEvents(){
  const es = new EventSource('_rpc/events');
  es.addEventListener('logs', e => { logLines.push(...(JSON.parse(e.data).lines || [])); logLines = logLines.slice(-1200); renderLogs(); });
  es.addEventListener('conversation', e => { records.push(...(JSON.parse(e.data).records || [])); records = records.slice(-500); renderConversation(); });
  es.onerror = () => { es.close(); setTimeout(connectEvents, 3000); };
}
loadStatus(); loadConversation(); loadLogs(); loadDebug(); connectEvents();
setInterval(loadStatus, 10000);
</script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
