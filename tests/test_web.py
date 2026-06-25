from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from voiceui.audio_dump import AudioDumpManager
from voiceui.logs import configure_log_files, reset_logging
from voiceui.models import AssistantReply, DebugConfig
from voiceui.web import VoiceUiWebConsole, read_text_records, start_web_console


class _StubConfig:
    def __init__(self, output_dir: str, *, session_scope: str = "run"):
        self.debug = DebugConfig(
            enabled=True,
            output_dir=output_dir,
            session_scope=session_scope,
        )


class _StubAssistant:
    def __init__(self, output_dir: str, *, session_scope: str = "run"):
        self.config = _StubConfig(output_dir, session_scope=session_scope)
        self.audio_dump = AudioDumpManager(self.config.debug)
        self.received: list[str] = []

    def run_text_turn(self, text: str) -> AssistantReply:
        self.received.append(text)
        return AssistantReply(text=f"reply: {text}", routed_to="llm")


class WebConsoleTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_logging()

    def test_web_console_reads_logs_conversation_and_debug_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "20260624-120000"
            session.mkdir()
            (session / "debug.log").write_text("one\ntwo\n", encoding="utf-8")
            (session / "metadata.json").write_text(
                json.dumps({"turns": [{"turn": 1, "transcript": "hi", "reply": "ok"}]}),
                encoding="utf-8",
            )
            audio_dir = session / "audio_dumps"
            audio_dir.mkdir()
            (audio_dir / "utterance_01_00.00.00.000_00.00.00.010.wav").write_bytes(b"wav")
            text_dir = root / "text_records"
            text_dir.mkdir()
            (text_dir / "voice_text_2026-06-24.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-06-24T12:00:00.000",
                                "module": "stt",
                                "event": "completed",
                                "role": "user",
                                "text": "hello",
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-06-24T12:00:01.000",
                                "module": "llm",
                                "event": "completed",
                                "role": "assistant",
                                "text": "world",
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-06-24T12:00:02.000",
                                "module": "tts",
                                "event": "completed",
                                "role": "assistant",
                                "text": "world",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            console = VoiceUiWebConsole(None, debug_output_dir=root)

            self.assertEqual(console.status()["latest_session"], session.name)
            self.assertEqual(console.logs(tail=1)["lines"], ["two"])
            self.assertEqual(
                [(item["role"], item["text"]) for item in console.conversation()["records"]],
                [("user", "hello"), ("assistant", "world")],
            )
            sessions = console.debug_sessions()["sessions"]
            self.assertEqual(sessions[0]["audio_count"], 1)
            detail = console.debug_session(session.name)
            self.assertEqual(detail["metadata"]["turns"][0]["transcript"], "hi")
            self.assertEqual(len(detail["audio_files"]), 1)


    def test_turn_scoped_console_logs_use_root_debug_log_when_idle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_session = root / "20260624-120000"
            old_session.mkdir()
            (old_session / "debug.log").write_text("old session\n", encoding="utf-8")
            (root / "debug.log").write_text("service idle\n", encoding="utf-8")
            assistant = _StubAssistant(temp_dir, session_scope="turn")

            console = VoiceUiWebConsole(assistant, debug_output_dir=root)

            logs = console.logs()
            self.assertEqual(logs["path"], str(root / "debug.log"))
            self.assertEqual(logs["lines"], ["service idle"])
            self.assertEqual(console.logs(session="latest")["lines"], ["old session"])

    def test_chat_endpoint_submits_text_and_records_user_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_dir = Path(temp_dir) / "text_records"
            configure_log_files(text_record_dir=text_dir)
            assistant = _StubAssistant(temp_dir)
            console = start_web_console(
                assistant, host="127.0.0.1", port=0, debug_output_dir=temp_dir
            )
            try:
                payload = json.dumps({"text": "hello web"}).encode("utf-8")
                request = Request(
                    console.url + "_rpc/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:  # noqa: S310 - localhost test
                    data = json.loads(response.read().decode("utf-8"))

                self.assertTrue(data["ok"])
                self.assertEqual(data["reply"], "reply: hello web")
                self.assertEqual(assistant.received, ["hello web"])
                records = read_text_records(text_dir, limit=10)
                self.assertEqual(records[0]["text"], "hello web")
                self.assertEqual(records[0]["params"], {"source": "web"})
            finally:
                console.stop()


if __name__ == "__main__":
    unittest.main()
