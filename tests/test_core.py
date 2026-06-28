from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

from voiceui.audio import RawAudioRecording
from voiceui.core import (
    VoiceAssistant,
    _extract_weather_location,
    _extract_weather_target_day,
    _looks_like_time_query,
    _matches_termination_command,
    _parse_volume_request,
    _streaming_frame_policy,
    _StreamingSttHandle,
)
from voiceui.llm import ChatMessage, ToolCall, ToolChatResponse
from voiceui.models import (
    AssistantConfig,
    AssistantReply,
    ConversationConfig,
    DebugConfig,
    InputConfig,
    LlmConfig,
    ToolsConfig,
    Utterance,
    VadConfig,
    WakeConfig,
    WakeEvent,
)
from voiceui.tools import create_tool_runner
from voiceui.tts import ConsoleTextToSpeech
from voiceui.vad import SpeechStartTimeoutError


class FakeWake:
    def wait(self, _audio) -> WakeEvent:
        return WakeEvent(engine="test_wake", confidence=1.0, label="wake")


def _text_record_summary(records: list[dict[str, object]]) -> list[tuple[object, object, object]]:
    return [(record["module"], record["role"], record["text"]) for record in records]


class FakeVad:
    def __init__(
        self,
        items: list[Utterance | type[SpeechStartTimeoutError]],
        on_record=None,
    ):
        self.items = items
        self.start_timeouts: list[float] = []
        self.on_record = on_record

    def record(
        self,
        _audio,
        start_timeout_seconds: float = 0.0,
        stop_event: threading.Event | None = None,
        on_speech_start=None,
        on_speech_audio=None,
    ) -> Utterance:
        self.start_timeouts.append(start_timeout_seconds)
        if self.on_record is not None:
            self.on_record()
        if stop_event is not None and stop_event.is_set():
            raise SpeechStartTimeoutError("Stopped waiting for speech.")
        if not self.items:
            raise AssertionError("Unexpected VAD record call.")
        item = self.items.pop(0)
        if item is SpeechStartTimeoutError:
            if stop_event is not None:
                if start_timeout_seconds <= 0:
                    while not stop_event.is_set():
                        time.sleep(0.01)
                else:
                    deadline = time.monotonic() + start_timeout_seconds
                    while not stop_event.is_set() and time.monotonic() < deadline:
                        time.sleep(0.01)
            raise SpeechStartTimeoutError("Timed out waiting for speech.")
        if on_speech_start is not None:
            on_speech_start()
        if on_speech_audio is not None:
            on_speech_audio(item.pcm)
        return item


class InfiniteAudio:
    sample_rate = 16000
    block_ms = 20

    def __init__(self, chunk: bytes = b"\x00\x00" * 320):
        self.chunk = chunk
        self.raw_recordings: list[RawAudioRecording] = []

    def chunks(self) -> Iterator[bytes]:
        while True:
            raw_chunk = b"\x01\x00\x02\x00" * 320
            for recording in list(self.raw_recordings):
                recording.append(raw_chunk)
            yield self.chunk

    def start_raw_recording(self, max_seconds=None) -> RawAudioRecording:
        recording = RawAudioRecording(sample_rate=self.sample_rate, channels=2)
        self.raw_recordings.append(recording)
        return recording

    def stop_raw_recording(self, recording: RawAudioRecording) -> None:
        if recording in self.raw_recordings:
            self.raw_recordings.remove(recording)


class ConsumingTimeoutVad:
    def record(
        self,
        audio,
        start_timeout_seconds: float = 0.0,
        stop_event: threading.Event | None = None,
        on_speech_start=None,
    ) -> Utterance:
        for _chunk in audio.chunks():
            if stop_event is not None and stop_event.is_set():
                raise SpeechStartTimeoutError("Stopped waiting for speech.")
            time.sleep(0.005)
        raise SpeechStartTimeoutError("Stopped waiting for speech.")


class FakeStt:
    def transcribe(self, utterance: Utterance) -> str:
        return utterance.pcm.decode("utf-8")


class FakeStreamingSession:
    def __init__(self, transcript: str):
        self.transcript = transcript
        self.written: list[bytes] = []
        self.finished = False
        self.aborted = False

    def write(self, pcm: bytes) -> None:
        self.written.append(pcm)

    def finish(self) -> str:
        self.finished = True
        return self.transcript

    def abort(self) -> None:
        self.aborted = True


class FakeStreamingStt(FakeStt):
    def __init__(self, transcript: str):
        self.session = FakeStreamingSession(transcript)
        self.start_sample_rates: list[int] = []
        self.fallback_calls = 0

    def supports_streaming(self) -> bool:
        return True

    def start_streaming(self, sample_rate: int) -> FakeStreamingSession:
        self.start_sample_rates.append(sample_rate)
        return self.session

    def transcribe(self, utterance: Utterance) -> str:
        self.fallback_calls += 1
        return super().transcribe(utterance)



class BlockingStreamingSession(FakeStreamingSession):
    def __init__(self, transcript: str, block_event: threading.Event):
        super().__init__(transcript)
        self.block_event = block_event
        self.write_started = threading.Event()

    def write(self, pcm: bytes) -> None:
        self.write_started.set()
        self.block_event.wait(timeout=2.0)
        super().write(pcm)


class BlockingStreamingStt(FakeStreamingStt):
    def __init__(self, transcript: str, block_event: threading.Event):
        super().__init__(transcript)
        self.session = BlockingStreamingSession(transcript, block_event)


class RecordingChat:
    def __init__(self):
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return f"reply {len(self.calls)}"

    def stream_complete(self, messages: list[ChatMessage]):
        self.calls.append(list(messages))
        yield "reply"
        yield f" {len(self.calls)}"


class FixedReplyChat(RecordingChat):
    def __init__(self, response: str):
        super().__init__()
        self.response = response

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return self.response

    def stream_complete(self, messages: list[ChatMessage]):
        self.calls.append(list(messages))
        yield self.response


class FailingChat(RecordingChat):
    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        raise RuntimeError("llm unavailable")

    def stream_complete(self, messages: list[ChatMessage]):
        self.calls.append(list(messages))
        raise RuntimeError("stream unavailable")
        yield ""


class ToolCallingChat:
    def __init__(self):
        self.calls: list[list[ChatMessage]] = []
        self.tools: list[list[dict]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return "unused"

    def stream_complete(self, messages: list[ChatMessage]):
        self.calls.append(list(messages))
        yield "unused"

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        self.calls.append(list(messages))
        self.tools.append(tools)
        if len(self.calls) == 1:
            return ToolChatResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_current_time",
                        arguments={"timezone": "Asia/Shanghai"},
                        raw={
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_current_time",
                                "arguments": "{\"timezone\":\"Asia/Shanghai\"}",
                            },
                        },
                    )
                ]
            )
        return ToolChatResponse(content="final time reply")


class FakeTts:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        self.spoken.append(text)

    def speak_text_stream(self, text_chunks, stop_event: threading.Event | None = None) -> str:
        text = "".join(text_chunks)
        self.spoken.append(text)
        return text


class FailingTts(FakeTts):
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        self.spoken.append(text)
        raise RuntimeError("speaker unavailable")

    def speak_text_stream(self, text_chunks, stop_event: threading.Event | None = None) -> str:
        text = "".join(text_chunks)
        self.spoken.append(text)
        raise RuntimeError("speaker unavailable")


class FailingToolRunner:
    enabled = True

    def complete(self, messages: list[ChatMessage]) -> str:
        raise RuntimeError("tool runner unavailable")


class SlowToolRunner:
    enabled = True

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        time.sleep(self.delay)
        return "tool reply"


class StreamingToolRunner:
    enabled = True

    def __init__(self):
        self.complete_calls: list[list[ChatMessage]] = []
        self.stream_calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.complete_calls.append(list(messages))
        raise AssertionError("Expected streaming tool completion")

    def stream_complete(self, messages: list[ChatMessage]):
        self.stream_calls.append(list(messages))
        yield "streamed "
        yield "tool reply"


class MiotTextToolRunner:
    enabled = True

    def __init__(self):
        self.calls: list[list[ChatMessage]] = []
        self.can_handle_calls: list[str] = []
        self.followup_calls: list[str] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return "好的，书房空调已设置。"

    def can_handle_miot_text(self, text: str) -> bool:
        self.can_handle_calls.append(text)
        return "温度调成" in text

    def can_handle_miot_followup_text(self, text: str) -> bool:
        self.followup_calls.append(text)
        return text == "书房的空调。"


class MiotScheduleToolRunner:
    enabled = True

    def __init__(self, preview: dict[str, object], executed: dict[str, object] | None = None):
        self.preview = preview
        self.executed = executed or {}
        self.calls: list[tuple[dict[str, object], bool]] = []
        self.can_handle_calls: list[str] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        raise AssertionError(f"Unexpected normal tool completion: {messages!r}")

    def can_handle_miot_control_text(self, text: str) -> bool:
        self.can_handle_calls.append(text)
        return any(term in text for term in ("空调", "灯", "窗帘", "设备"))

    def run_miot_control(
        self,
        arguments: dict[str, object],
        *,
        remember: bool = True,
    ) -> dict[str, object]:
        self.calls.append((dict(arguments), remember))
        if arguments.get("dry_run"):
            return dict(self.preview)
        return dict(self.executed)

    def format_tool_response(self, payload: dict[str, object]) -> str:
        return str(payload.get("direct_response") or payload.get("message") or "")


class BargeFirstTts(FakeTts):
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        self.spoken.append(text)
        if len(self.spoken) != 1 or stop_event is None:
            return
        deadline = time.monotonic() + 1.0
        while not stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)


class BargeEveryTts(FakeTts):
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        self.spoken.append(text)
        if stop_event is None:
            return
        deadline = time.monotonic() + 1.0
        while not stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)


class FakeWakeAck:
    def __init__(self):
        self.calls = 0

    def play(self) -> None:
        self.calls += 1


class FakeMusicController:
    def __init__(self, active: bool = True):
        self.active = active
        self.ducked = False
        self.events: list[tuple[str, str]] = []
        self.stop_calls: list[bool] = []

    def duck(self, reason: str) -> None:
        self.ducked = True
        self.events.append(("duck", reason))

    def unduck(self, reason: str) -> None:
        self.ducked = False
        self.events.append(("unduck", reason))

    def is_active(self) -> bool:
        return self.active

    def stop(self, wait: bool = False):
        self.stop_calls.append(wait)
        self.active = False
        return {"status": "stopped"}


class SlowWakeAck(FakeWakeAck):
    def __init__(self, delay_seconds: float):
        super().__init__()
        self.delay_seconds = delay_seconds
        self.started = threading.Event()
        self.finished = threading.Event()

    def play(self) -> None:
        self.calls += 1
        self.started.set()
        time.sleep(self.delay_seconds)
        self.finished.set()


class CoreTests(unittest.TestCase):
    def test_run_once_starts_vad_while_wake_ack_is_playing(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=0),
            llm=LlmConfig(system_prompt="system"),
        )
        wake_ack = SlowWakeAck(delay_seconds=0.1)
        vad_saw_ack_finished: list[bool] = []
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        assistant.wake = FakeWake()
        assistant.wake_ack = wake_ack
        assistant.vad = FakeVad(
            [Utterance(pcm=b"first", sample_rate=16000, duration_ms=80)],
            on_record=lambda: vad_saw_ack_finished.append(wake_ack.finished.is_set()),
        )
        assistant.stt = FakeStt()
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        with contextlib.redirect_stdout(io.StringIO()):
            assistant.run_once()

        self.assertEqual(wake_ack.calls, 1)
        self.assertEqual(vad_saw_ack_finished, [False])
        self.assertTrue(wake_ack.finished.is_set())

    def test_run_conversation_keeps_context_for_follow_up_without_second_wake(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(
                follow_up_seconds=1,
                max_turns=4,
                follow_up_gate_enabled=False,
            ),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        fake_vad = FakeVad(
            [
                Utterance(pcm=b"first", sample_rate=16000, duration_ms=80),
                Utterance(pcm=b"second", sample_rate=16000, duration_ms=80),
                SpeechStartTimeoutError,
            ]
        )
        chat = RecordingChat()
        tts = FakeTts()
        wake_ack = FakeWakeAck()
        assistant.wake = FakeWake()
        assistant.wake_ack = wake_ack
        assistant.vad = fake_vad
        assistant.stt = FakeStt()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertIsInstance(reply, AssistantReply)
        self.assertEqual(wake_ack.calls, 1)
        self.assertEqual(fake_vad.start_timeouts, [8.0, 1, 1])
        self.assertEqual(tts.spoken, ["reply 1", "reply 2"])
        self.assertEqual([message.content for message in chat.calls[0]], ["system", "first"])
        self.assertEqual(
            [message.content for message in chat.calls[1]],
            ["system", "first", "reply 1", "second"],
        )

    def test_follow_up_exit_command_returns_to_wake_without_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        fake_vad = FakeVad(
            [
                Utterance(pcm=b"first", sample_rate=16000, duration_ms=80),
                Utterance(pcm="退出吧。".encode(), sample_rate=16000, duration_ms=80),
                SpeechStartTimeoutError,
            ]
        )
        chat = RecordingChat()
        tts = FakeTts()
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = fake_vad
        assistant.stt = FakeStt()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertEqual(reply.routed_to, "voice_termination")
        self.assertEqual(reply.text, "")
        self.assertEqual(tts.spoken, ["reply 1"])
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(fake_vad.items, [SpeechStartTimeoutError])

    def test_wake_speech_timeout_returns_to_wake_without_llm_or_tts(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(
                follow_up_seconds=1,
                wake_speech_start_timeout_seconds=2,
            ),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        fake_vad = FakeVad([SpeechStartTimeoutError])
        chat = RecordingChat()
        tts = FakeTts()
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = fake_vad
        assistant.stt = FakeStt()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()) as output:
            reply = assistant.run_conversation()

        self.assertEqual(reply.routed_to, "system")
        self.assertEqual(reply.text, "")
        self.assertEqual(chat.calls, [])
        self.assertEqual(tts.spoken, [])
        self.assertEqual(fake_vad.start_timeouts, [2])
        self.assertIn("event=wake_speech_timeout", output.getvalue())

    def test_wake_asr_termination_returns_to_wake_without_llm_or_tts(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        fake_vad = FakeVad(
            [
                Utterance(pcm="停止".encode(), sample_rate=16000, duration_ms=80),
                SpeechStartTimeoutError,
            ]
        )
        chat = RecordingChat()
        tts = FakeTts()
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = fake_vad
        assistant.stt = FakeStt()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertEqual(reply.routed_to, "voice_termination")
        self.assertEqual(reply.text, "")
        self.assertEqual(chat.calls, [])
        self.assertEqual(tts.spoken, [])
        self.assertEqual(fake_vad.items, [SpeechStartTimeoutError])

    def test_run_conversation_ducks_music_after_wake_until_exit(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=0),
            llm=LlmConfig(system_prompt="system"),
        )
        music = FakeMusicController()
        ducked_during_vad: list[bool] = []
        assistant = VoiceAssistant(config)
        assistant.music_controller = music
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = FakeVad(
            [Utterance(pcm=b"first", sample_rate=16000, duration_ms=80)],
            on_record=lambda: ducked_during_vad.append(music.ducked),
        )
        assistant.stt = FakeStt()
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        with contextlib.redirect_stdout(io.StringIO()):
            assistant.run_conversation()

        self.assertEqual(ducked_during_vad, [True])
        self.assertFalse(music.ducked)
        self.assertEqual(music.events[0], ("duck", "conversation"))
        self.assertEqual(music.events[-1], ("unduck", "conversation"))

    def test_text_turn_streams_llm_response_to_tts_and_session(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system", stream=True),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        chat = RecordingChat()
        tts = FakeTts()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("first")

        self.assertEqual(reply.text, "reply 1")
        self.assertEqual(tts.spoken, ["reply 1"])
        self.assertEqual([message.content for message in chat.calls[0]], ["system", "first"])
        self.assertEqual(
            [message.content for message in assistant.session.messages],
            ["system", "first", "reply 1"],
        )
        self.assertIsNone(assistant.audio_dump.current_turn_index)

    def test_console_tts_text_turn_uses_structured_log_output(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system", stream=False),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        assistant.chat = RecordingChat()
        assistant.tts = ConsoleTextToSpeech()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            reply = assistant.run_text_turn("first")

        self.assertEqual(reply.text, "reply 1")
        self.assertNotIn("assistant>", output.getvalue())
        self.assertIn("module=tts | event=completed", output.getvalue())
        self.assertIn(">>> TTS TEXT: reply 1", output.getvalue())

    def test_debug_session_writes_debug_log_and_daily_text_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AssistantConfig(
                input=InputConfig(mode="text"),
                llm=LlmConfig(system_prompt="system", stream=False),
                debug=DebugConfig(enabled=True, output_dir=temp_dir),
            )
            with patch("voiceui.core.warm_weather_cache"):
                assistant = VoiceAssistant(config)
            assistant.chat = RecordingChat()
            assistant.tts = FakeTts()

            with contextlib.redirect_stdout(io.StringIO()):
                reply = assistant.run_text_turn("first")

            self.assertEqual(reply.text, "reply 1")
            session_dirs = [
                path
                for path in Path(temp_dir).iterdir()
                if path.is_dir() and path.name != "text_records"
            ]
            self.assertEqual(len(session_dirs), 1)
            debug_log = session_dirs[0] / "debug.log"
            self.assertTrue(debug_log.exists())
            self.assertIn("module=tts | event=completed", debug_log.read_text(encoding="utf-8"))

            text_files = list((Path(temp_dir) / "text_records").glob("voice_text_*.jsonl"))
            self.assertEqual(len(text_files), 1)
            records = [
                json.loads(line)
                for line in text_files[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn(("llm", "assistant", "reply 1"), _text_record_summary(records))
            self.assertIn(("tts", "assistant", "reply 1"), _text_record_summary(records))

    def test_text_turn_reports_non_stream_llm_error_without_raising(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system", stream=False),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        chat = FailingChat()
        tts = FakeTts()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("讲个笑话")

        self.assertEqual(reply.routed_to, "error")
        self.assertEqual(reply.text, "刚才处理失败了，请再说一遍。")
        self.assertEqual(tts.spoken, [reply.text])
        self.assertEqual(
            [message.content for message in assistant.session.messages],
            ["system", "讲个笑话", reply.text],
        )

    def test_text_turn_reports_stream_llm_error_without_raising(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system", stream=True),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        chat = FailingChat()
        tts = FakeTts()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("讲个笑话")

        self.assertEqual(reply.routed_to, "error")
        self.assertEqual(reply.text, "刚才处理失败了，请再说一遍。")
        self.assertEqual(tts.spoken, [reply.text])
        self.assertEqual(
            [message.content for message in assistant.session.messages],
            ["system", "讲个笑话", reply.text],
        )

    def test_text_turn_runs_tool_calls_before_tts(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(
                enabled=True,
                allow_weather=False,
            ),
        )
        assistant = VoiceAssistant(config)
        chat = ToolCallingChat()
        tts = FakeTts()
        assistant.chat = chat
        assistant.tts = tts
        assistant.tool_runner = create_tool_runner(config, assistant.chat)

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("time")

        self.assertIn("现在是", reply.text)
        self.assertEqual(tts.spoken, [reply.text])
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(
            [message.content for message in assistant.session.messages],
            ["system", "time", reply.text],
        )

    def test_time_query_does_not_match_background_time_phrase(self) -> None:
        self.assertTrue(_looks_like_time_query("现在几点"))
        self.assertTrue(_looks_like_time_query("报一下时间"))
        self.assertFalse(_looks_like_time_query("一个月的时间里持续上涨超过60%"))

    def test_voice_termination_matching_uses_short_explicit_commands(self) -> None:
        phrases = ["停止", "结束", "闭嘴", "stop"]

        self.assertTrue(_matches_termination_command("停止。", phrases))
        self.assertTrue(_matches_termination_command("闭嘴", phrases))
        self.assertTrue(_matches_termination_command("stop", phrases))
        self.assertFalse(_matches_termination_command("停止播放", phrases))
        self.assertFalse(_matches_termination_command("结束后提醒我", phrases))

    def test_text_turn_does_not_route_background_time_phrase_to_time_tool(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(enabled=True, allow_time=True, allow_weather=False),
        )
        assistant = VoiceAssistant(config)
        assistant.chat = RecordingChat()
        assistant.tool_runner = create_tool_runner(config, assistant.chat)
        assistant.tts = FakeTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("一个月的时间里，持续上涨了超过60%。")

        self.assertEqual(reply.text, "reply 1")
        self.assertEqual(len(assistant.chat.calls), 1)

    def test_text_turn_schedules_alarm_without_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(enabled=True, allow_time=True, allow_weather=False),
        )
        assistant = VoiceAssistant(config)
        assistant.chat = RecordingChat()
        assistant.tool_runner = create_tool_runner(config, assistant.chat)
        assistant.tts = FakeTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("定一个一分钟后的闹钟。")

        self.assertEqual(reply.routed_to, "reminder")
        self.assertEqual(reply.text, "好的，1分钟后提醒你。")
        self.assertEqual(assistant.chat.calls, [])
        self.assertEqual(len(assistant.reminders.pending()), 1)
        self.assertNotIn("现在是", reply.text)
        with contextlib.redirect_stdout(io.StringIO()):
            assistant.reminders.cancel_all()

    def test_text_turn_clarifies_alarm_without_time(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("帮我设置一个闹钟。")

        self.assertEqual(reply.routed_to, "reminder")
        self.assertEqual(reply.text, "你想让我什么时候提醒？")
        self.assertEqual(assistant.chat.calls, [])

    def test_due_reminder_speaks_without_barge_in(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        tts = FakeTts()
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reminder = assistant.reminders.schedule_after(
                1,
                "闹钟时间到了。",
                kind="alarm",
            )
            fired = assistant.reminders.run_due(reminder.due_at)

        self.assertEqual(fired, [reminder.id])
        self.assertEqual(tts.spoken, ["闹钟时间到了。"])

    def test_text_turn_schedules_miot_control_after_dry_run(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        assistant.chat = RecordingChat()
        tts = FakeTts()
        assistant.tts = tts
        assistant.tool_runner = MiotScheduleToolRunner(
            preview={
                "status": "resolved",
                "decision": "resolved",
                "device": {
                    "name": "客厅空调",
                    "room_name": "客厅",
                    "device_class": "aircondition",
                },
                "action": "turn_off",
                "target_value": False,
            },
            executed={
                "status": "verified",
                "direct_response": "好的，客厅空调已关闭。",
            },
        )

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("十分钟后关闭空调")

        self.assertEqual(reply.routed_to, "miot_schedule")
        self.assertIn("帮你执行", reply.text)
        self.assertEqual(assistant.chat.calls, [])
        pending = assistant.reminders.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].kind, "miot")
        self.assertEqual(pending[0].payload["arguments"]["request"], "关闭空调")
        self.assertEqual(pending[0].payload["arguments"]["device"], "客厅空调")
        self.assertEqual(pending[0].payload["arguments"]["action"], "turn_off")
        self.assertEqual(
            assistant.tool_runner.calls[0],
            ({"request": "关闭空调", "dry_run": True}, False),
        )

        with contextlib.redirect_stdout(io.StringIO()):
            fired = assistant.reminders.run_due(pending[0].due_at)

        self.assertEqual(fired, [pending[0].id])
        self.assertEqual(tts.spoken, [reply.text, "好的，客厅空调已关闭。"])
        execute_arguments, remember = assistant.tool_runner.calls[1]
        self.assertTrue(remember)
        self.assertNotIn("dry_run", execute_arguments)
        self.assertEqual(execute_arguments["device"], "客厅空调")

    def test_text_turn_does_not_schedule_ambiguous_miot_control(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()
        assistant.tool_runner = MiotScheduleToolRunner(
            preview={
                "status": "ambiguous",
                "message": "找到多个匹配设备，请指定哪一个。",
                "candidates": [
                    {"name": "书房空调"},
                    {"name": "客厅空调"},
                ],
            }
        )

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("十分钟后关闭空调")

        self.assertEqual(reply.routed_to, "miot_schedule")
        self.assertEqual(reply.text, "找到多个匹配设备，请指定哪一个。")
        self.assertEqual(assistant.reminders.pending(), [])
        self.assertEqual(len(assistant.tool_runner.calls), 1)

    def test_text_turn_speaks_progress_prompt_when_tool_runner_is_slow(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(enabled=True, allow_weather=False),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        tts = FakeTts()
        tool_runner = SlowToolRunner()
        assistant.tts = tts
        assistant.tool_runner = tool_runner

        with patch("voiceui.core._TOOL_PROGRESS_PROMPT_DELAY_SECONDS", 0.01):
            with contextlib.redirect_stdout(io.StringIO()):
                reply = assistant.run_text_turn("帮我搜索一下新闻")

        self.assertEqual(reply.text, "tool reply")
        self.assertEqual(tts.spoken[0], "正在搜索，请稍等。")
        self.assertEqual(tts.spoken[-1], "tool reply")
        self.assertEqual(len(tool_runner.calls), 1)

    def test_text_turn_streams_tool_runner_when_llm_stream_enabled(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system", stream=True),
            tools=ToolsConfig(enabled=True, allow_weather=False),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        tts = FakeTts()
        tool_runner = StreamingToolRunner()
        assistant.tts = tts
        assistant.tool_runner = tool_runner

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("讲个笑话")

        self.assertEqual(reply.text, "streamed tool reply")
        self.assertEqual(tts.spoken, ["streamed tool reply"])
        self.assertEqual(len(tool_runner.stream_calls), 1)
        self.assertEqual(tool_runner.complete_calls, [])

    def test_text_turn_reports_tool_runner_error_without_raising(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(enabled=True, allow_weather=False),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        tts = FakeTts()
        assistant.tts = tts
        assistant.tool_runner = FailingToolRunner()

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("打开灯")

        self.assertEqual(reply.routed_to, "error")
        self.assertEqual(reply.text, "刚才处理失败了，请再说一遍。")
        self.assertEqual(tts.spoken, [reply.text])
        self.assertEqual(
            [message.content for message in assistant.session.messages],
            ["system", "打开灯", reply.text],
        )

    def test_text_turn_reports_local_tool_error_without_raising(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(enabled=True, allow_time=True, allow_weather=False),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        chat = RecordingChat()
        tts = FakeTts()
        assistant.chat = chat
        assistant.tts = tts

        with patch("voiceui.core.get_current_time", side_effect=RuntimeError("time failed")):
            with contextlib.redirect_stdout(io.StringIO()):
                reply = assistant.run_text_turn("现在几点")

        self.assertEqual(reply.routed_to, "error")
        self.assertEqual(reply.text, "刚才处理失败了，请再说一遍。")
        self.assertEqual(chat.calls, [])
        self.assertEqual(tts.spoken, [reply.text])

    def test_text_turn_tts_error_does_not_raise_after_successful_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system", stream=False),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        chat = RecordingChat()
        tts = FailingTts()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("讲个笑话")

        self.assertEqual(reply.text, "reply 1")
        self.assertEqual(tts.spoken, ["reply 1"])
        self.assertEqual(
            [message.content for message in assistant.session.messages],
            ["system", "讲个笑话", "reply 1"],
        )

    def test_text_turn_handles_default_weather_without_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(
                enabled=True,
                allow_weather=True,
                default_weather_location="上海",
            ),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        chat = RecordingChat()
        tts = FakeTts()
        assistant.chat = chat
        assistant.tts = tts
        weather_result = {
            "location": {"name": "上海"},
            "current": {
                "temperature_2m": 25.6,
                "apparent_temperature": 30.8,
                "weather_code": 61,
            },
            "units": {"temperature_2m": "°C"},
            "direct_response": "上海今天小雨，气温25.6°C，体感30.8°C。出门记得带伞。",
        }

        with patch("voiceui.core.get_current_weather", return_value=weather_result) as weather:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                reply = assistant.run_text_turn("你看一下今天的天气怎么样？")

        weather.assert_called_once_with({"location": "上海"})
        self.assertEqual(chat.calls, [])
        self.assertEqual(reply.text, weather_result["direct_response"])
        self.assertEqual(tts.spoken, [weather_result["direct_response"]])
        self.assertIn("mode=local_tools", output.getvalue())

    def test_text_turn_defers_temperature_adjustment_to_miot_runner(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(
                enabled=True,
                allow_weather=True,
                default_weather_location="北京昌平",
            ),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        runner = MiotTextToolRunner()
        assistant.tool_runner = runner
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        with patch("voiceui.core.get_current_weather") as weather:
            with contextlib.redirect_stdout(io.StringIO()):
                reply = assistant.run_text_turn("温度调成27度。")

        weather.assert_not_called()
        self.assertEqual(reply.text, "好的，书房空调已设置。")
        self.assertEqual(runner.can_handle_calls, ["温度调成27度。"])
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(assistant.tts.spoken, ["好的，书房空调已设置。"])

    def test_weather_location_extraction_ignores_tomorrow_prompt_without_city(self) -> None:
        self.assertEqual(_extract_weather_target_day("你知道明天的天气吗？"), "tomorrow")
        self.assertEqual(
            _extract_weather_location("你知道明天的天气吗？", default_location="北京昌平"),
            "北京昌平",
        )
        self.assertEqual(
            _extract_weather_location("北京昌平明天的天气怎么样？", default_location="上海"),
            "北京昌平",
        )

    def test_text_turn_handles_default_tomorrow_weather_without_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(
                enabled=True,
                allow_weather=True,
                default_weather_location="北京昌平",
            ),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        chat = RecordingChat()
        tts = FakeTts()
        assistant.chat = chat
        assistant.tts = tts
        weather_result = {
            "location": {"name": "北京昌平"},
            "target_day": "tomorrow",
            "daily": {
                "weather_code": 3,
                "temperature_2m_max": 29.0,
                "temperature_2m_min": 18.0,
            },
            "daily_units": {"temperature_2m_max": "°C"},
            "direct_response": "北京昌平明天阴天，最高29°C，最低18°C。",
        }

        with patch("voiceui.core.get_current_weather", return_value=weather_result) as weather:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                reply = assistant.run_text_turn("你知道明天的天气吗？")

        weather.assert_called_once_with({"location": "北京昌平", "target_day": "tomorrow"})
        self.assertEqual(chat.calls, [])
        self.assertEqual(reply.text, weather_result["direct_response"])
        self.assertEqual(tts.spoken, [weather_result["direct_response"]])
        self.assertIn("mode=local_tools", output.getvalue())

    def test_text_turn_weather_falls_back_to_default_location_on_lookup_error(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
            tools=ToolsConfig(
                enabled=True,
                allow_weather=True,
                default_weather_location="北京昌平",
            ),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        chat = RecordingChat()
        tts = FakeTts()
        assistant.chat = chat
        assistant.tts = tts
        weather_result = {
            "location": {"name": "北京昌平"},
            "target_day": "tomorrow",
            "daily": {"weather_code": 3},
            "direct_response": "北京昌平明天阴天。",
        }

        def fake_weather(arguments):
            if arguments["location"] == "你知道明天":
                raise RuntimeError("Could not find weather location: 你知道明天")
            return weather_result

        with patch("voiceui.core._extract_weather_location", return_value="你知道明天"):
            with patch("voiceui.core.get_current_weather", side_effect=fake_weather) as weather:
                with contextlib.redirect_stdout(io.StringIO()):
                    reply = assistant.run_text_turn("你知道明天的天气吗？")

        self.assertEqual(
            [call.args[0] for call in weather.call_args_list],
            [
                {"location": "你知道明天", "target_day": "tomorrow"},
                {"location": "北京昌平", "target_day": "tomorrow"},
            ],
        )
        self.assertEqual(chat.calls, [])
        self.assertEqual(reply.text, weather_result["direct_response"])
        self.assertEqual(tts.spoken, [weather_result["direct_response"]])

    def test_text_turn_stops_active_music_without_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        music = FakeMusicController(active=True)
        chat = RecordingChat()
        tts = FakeTts()
        assistant.music_controller = music
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_text_turn("停止播放")

        self.assertEqual(reply.text, "已停止播放。")
        self.assertEqual(music.stop_calls, [True])
        self.assertEqual(chat.calls, [])
        self.assertEqual(tts.spoken, ["已停止播放。"])

    def test_run_conversation_processes_barge_in_as_next_turn(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(
                follow_up_seconds=1,
                max_turns=4,
                barge_in_enabled=True,
                barge_in_gate_enabled=False,
            ),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        fake_vad = FakeVad(
            [
                Utterance(pcm=b"first", sample_rate=16000, duration_ms=80),
                Utterance(pcm=b"barge", sample_rate=16000, duration_ms=80),
                SpeechStartTimeoutError,
                SpeechStartTimeoutError,
            ]
        )
        chat = RecordingChat()
        tts = BargeFirstTts()
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = fake_vad
        assistant.stt = FakeStt()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertIsInstance(reply, AssistantReply)
        self.assertEqual(tts.spoken, ["reply 1", "reply 2"])
        self.assertEqual([message.content for message in chat.calls[0]], ["system", "first"])
        self.assertEqual(
            [message.content for message in chat.calls[1]],
            ["system", "first", "reply 1", "barge"],
        )
        self.assertIn(0.0, fake_vad.start_timeouts)

    def test_barge_in_termination_returns_to_wake_without_second_llm_turn(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(
                follow_up_seconds=1,
                max_turns=4,
                barge_in_enabled=True,
            ),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        fake_vad = FakeVad(
            [
                Utterance(pcm=b"first", sample_rate=16000, duration_ms=80),
                Utterance(pcm="闭嘴".encode(), sample_rate=16000, duration_ms=80),
                SpeechStartTimeoutError,
            ]
        )
        chat = RecordingChat()
        tts = BargeFirstTts()
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = fake_vad
        assistant.stt = FakeStt()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertEqual(reply.routed_to, "voice_termination")
        self.assertEqual(reply.text, "")
        self.assertEqual(tts.spoken, ["reply 1"])
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual([message.content for message in chat.calls[0]], ["system", "first"])
        self.assertIsNone(assistant._pending_barge_utterance)
        self.assertEqual(fake_vad.items, [SpeechStartTimeoutError])

    def test_follow_up_background_monologue_is_rejected_without_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        background = (
            "一个月的时间里，持续上涨了超过60%。一季度成我们欠的是一级适用率数据。"
        )
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = FakeVad(
            [
                Utterance(pcm=b"first", sample_rate=16000, duration_ms=80),
                Utterance(pcm=background.encode(), sample_rate=16000, duration_ms=1200),
            ]
        )
        assistant.stt = FakeStt()
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertEqual(reply.routed_to, "input_gate")
        self.assertEqual(reply.text, "我没听清。")
        self.assertEqual(len(assistant.chat.calls), 1)
        self.assertEqual(assistant.tts.spoken, ["reply 1"])

    def test_barge_in_ambiguous_short_text_rejects_silently_without_session_pollution(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(
                follow_up_seconds=0,
                barge_in_enabled=True,
            ),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = FakeVad(
            [
                Utterance(pcm=b"first", sample_rate=16000, duration_ms=80),
                Utterance(
                    pcm="来开提供临时民工。".encode(),
                    sample_rate=16000,
                    duration_ms=1200,
                ),
            ]
        )
        assistant.stt = FakeStt()
        assistant.chat = RecordingChat()
        assistant.tts = BargeFirstTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertEqual(reply.routed_to, "input_gate")
        self.assertEqual(reply.text, "我没听清。")
        self.assertEqual(len(assistant.chat.calls), 1)
        self.assertEqual(
            [message.content for message in assistant.session.messages],
            ["system", "first", "reply 1"],
        )
        self.assertEqual(assistant.tts.spoken, ["reply 1"])

    def test_follow_up_miot_pending_reference_bypasses_short_clarification(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        runner = MiotTextToolRunner()
        assistant.tool_runner = runner

        with contextlib.redirect_stdout(io.StringIO()):
            decision, reason, response = assistant._gate_voice_transcript(  # pylint: disable=protected-access
                "书房的空调。",
                "follow_up",
            )

        self.assertEqual((decision, reason, response), ("accept", "miot_followup_context", ""))
        self.assertEqual(runner.followup_calls, ["书房的空调。"])

    def test_wake_background_question_terms_clarify_before_direct_intent(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        background = (
            "所以一个五块钱都不给这些豪门，主要是联赛方式是不要联赛，"
            "这话放在他们身上，可能是那问题到底出在哪？往下看你就懂了。"
        )

        with contextlib.redirect_stdout(io.StringIO()):
            decision, reason, response = assistant._gate_voice_transcript(  # pylint: disable=protected-access
                background,
                "wake",
            )

        self.assertEqual(decision, "reject")
        self.assertEqual(reason, "wake_background_like")
        self.assertEqual(response, "")

    def test_wake_without_direct_intent_is_rejected_before_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)

        with contextlib.redirect_stdout(io.StringIO()):
            decision, reason, response = assistant._gate_voice_transcript(  # pylint: disable=protected-access
                "等我掉下去之后再说。",
                "wake",
            )

        self.assertEqual((decision, reason, response), ("reject", "wake_no_direct_intent", ""))

    def test_wake_false_wake_complaint_is_rejected_before_direct_question(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)

        with contextlib.redirect_stdout(io.StringIO()):
            decision, reason, response = assistant._gate_voice_transcript(  # pylint: disable=protected-access
                "谁叫你了？",
                "wake",
            )

        self.assertEqual((decision, reason, response), ("reject", "false_wake_complaint", ""))

    def test_follow_up_background_question_terms_reject_before_direct_intent(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        background = (
            "都淹死在平均之地。整个苏超12支队加起来才赚了3.57亿，"
            "为啥这么猛？因为提问了，哪怕小组就数据。"
        )

        with contextlib.redirect_stdout(io.StringIO()):
            decision, reason, response = assistant._gate_voice_transcript(  # pylint: disable=protected-access
                background,
                "follow_up",
            )

        self.assertEqual((decision, reason, response), ("reject", "background_monologue", ""))

    def test_follow_up_contextual_correction_bypasses_short_clarification(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)

        with contextlib.redirect_stdout(io.StringIO()):
            decision, reason, response = assistant._gate_voice_transcript(  # pylint: disable=protected-access
                "原神的火神不是虎视。",
                "follow_up",
            )

        self.assertEqual((decision, reason, response), ("accept", "contextual_correction", ""))

    def test_follow_up_iot_command_with_le_suffix_is_direct_intent(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=1),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)

        with contextlib.redirect_stdout(io.StringIO()):
            decision, reason, response = assistant._gate_voice_transcript(  # pylint: disable=protected-access
                "把卧室的灯关了。",
                "follow_up",
            )

        self.assertEqual((decision, reason, response), ("accept", "direct_intent", ""))

    def test_barge_in_ambiguous_text_does_not_clarify_or_trigger_echo_chain(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(
                follow_up_seconds=0,
                barge_in_enabled=True,
            ),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        self_echo = Utterance(pcm="我没太听清。".encode(), sample_rate=16000, duration_ms=1200)
        fake_vad = FakeVad(
            [
                Utterance(pcm=b"first", sample_rate=16000, duration_ms=80),
                Utterance(
                    pcm="来开提供临时民工。".encode(),
                    sample_rate=16000,
                    duration_ms=1200,
                ),
                self_echo,
            ]
        )
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = fake_vad
        assistant.stt = FakeStt()
        assistant.chat = RecordingChat()
        assistant.tts = BargeEveryTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertEqual(reply.routed_to, "input_gate")
        self.assertEqual(reply.text, "我没听清。")
        self.assertEqual(assistant.tts.spoken, ["reply 1"])
        self.assertEqual(fake_vad.items, [self_echo])
        self.assertIsNone(assistant._pending_barge_utterance)

    def test_barge_in_echo_of_normal_reply_is_rejected_without_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(
                follow_up_seconds=0,
                barge_in_enabled=True,
            ),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        assistant.wake = FakeWake()
        assistant.wake_ack = FakeWakeAck()
        assistant.vad = FakeVad(
            [
                Utterance(pcm=b"first", sample_rate=16000, duration_ms=80),
                Utterance(pcm="今天精神。".encode(), sample_rate=16000, duration_ms=1200),
            ]
        )
        assistant.stt = FakeStt()
        assistant.chat = FixedReplyChat("听起来你昨晚没休息好，今天精神确实容易亢奋。")
        assistant.tts = BargeEveryTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertEqual(reply.routed_to, "input_gate")
        self.assertEqual(reply.text, "我没听清。")
        self.assertEqual(len(assistant.chat.calls), 1)
        self.assertEqual(assistant.tts.spoken, [assistant.chat.response])

    def test_barge_in_no_speech_saves_monitor_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AssistantConfig(
                input=InputConfig(mode="audio"),
                wake=WakeConfig(engine="disabled"),
                conversation=ConversationConfig(barge_in_enabled=True),
                debug=DebugConfig(enabled=True, output_dir=temp_dir),
            )
            assistant = VoiceAssistant(config)
            assistant.command_audio = InfiniteAudio(chunk=b"\x01\x00" * 320)
            assistant.vad = ConsumingTimeoutVad()
            assistant.tts = BargeFirstTts()

            with contextlib.redirect_stdout(io.StringIO()):
                utterance = assistant._speak_with_barge_in("reply")

            self.assertIsNone(utterance)
            debug_dirs = [
                path
                for path in Path(temp_dir).iterdir()
                if path.is_dir() and path.name != "text_records"
            ]
            self.assertEqual(len(debug_dirs), 1)
            audio_dir = debug_dirs[0] / "audio_dumps"
            self.assertEqual(
                len(list(audio_dir.glob("barge_in_monitor_01_*.wav"))),
                1,
            )
            self.assertEqual(list(audio_dir.glob("barge_in_raw*.wav")), [])
            metadata = json.loads((debug_dirs[0] / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["barge_in"]), 1)
            self.assertEqual(metadata["barge_in"][0]["turn"], 1)


    def test_text_turn_handles_volume_locally_without_llm(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            tools=ToolsConfig(
                enabled=True,
                allow_time=False,
                allow_weather=False,
                allow_volume=True,
                allow_music=False,
            ),
        )
        with patch("voiceui.core.warm_weather_cache"):
            assistant = VoiceAssistant(config)
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        with patch("voiceui.core.set_system_output_volume") as set_volume:
            set_volume.return_value = {"after_percent": 60}
            reply = assistant.run_text_turn("音量调低20%")

        set_volume.assert_called_once_with(
            device=config.tts.playback_device,
            volume_percent=None,
            relative_percent=-20.0,
            muted=None,
        )
        self.assertEqual(reply.text, "好的，音量已调到60% 。")
        self.assertEqual(assistant.chat.calls, [])

    def test_parse_volume_request(self) -> None:
        self.assertEqual(
            _parse_volume_request("音量调低20%"),
            {"action": "set", "relative_percent": -20.0},
        )
        self.assertEqual(
            _parse_volume_request("音量调到60%"),
            {"action": "set", "volume_percent": 60.0},
        )
        self.assertEqual(_parse_volume_request("当前音量多少"), {"action": "get"})

    def test_audio_turn_streams_stt_during_vad(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=0),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        assistant.vad = FakeVad([Utterance(pcm=b"streamed", sample_rate=16000, duration_ms=80)])
        streaming_stt = FakeStreamingStt("streamed")
        assistant.stt = streaming_stt
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply, transcript = assistant._run_audio_turn(
                WakeEvent(engine="test", confidence=1.0, label="wake"),
                wake_ms=0,
            )

        self.assertEqual(transcript, "streamed")
        self.assertEqual(reply.text, "reply 1")
        self.assertEqual(streaming_stt.start_sample_rates, [16000])
        self.assertEqual(streaming_stt.session.written, [b"streamed"])
        self.assertTrue(streaming_stt.session.finished)
        self.assertEqual(streaming_stt.fallback_calls, 0)



    def test_streaming_frame_policy_matches_vad_timing(self) -> None:
        policy = _streaming_frame_policy(
            VadConfig(pre_roll_ms=240, min_speech_ms=250, frame_ms=20),
            audio_block_ms=80,
        )

        self.assertEqual(policy.frame_ms, 20)
        self.assertEqual(policy.buffer_frames, 25)
        self.assertAlmostEqual(policy.ready_timeout_seconds, 0.5)

    def test_streaming_stt_handle_writes_without_blocking_capture_thread(self) -> None:
        block_event = threading.Event()
        stt = BlockingStreamingStt("done", block_event)
        handle = _StreamingSttHandle(stt, 16000, max_buffered_chunks=1)
        handle.start()
        self.assertTrue(handle.wait_ready(timeout=1.0))

        started = time.monotonic()
        handle.write(b"frame")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.05)
        self.assertTrue(stt.session.write_started.wait(timeout=1.0))
        self.assertEqual(stt.session.written, [])
        block_event.set()

        self.assertEqual(handle.finish(), "done")
        self.assertEqual(stt.session.written, [b"frame"])
        self.assertEqual(handle.sent_chunks, 1)

    def test_audio_turn_starts_vad_before_streaming_stt_is_ready(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=0),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        assistant.vad = FakeVad([Utterance(pcm=b"streamed", sample_rate=16000, duration_ms=80)])
        streaming_stt = FakeStreamingStt("streamed")
        ready_event = threading.Event()
        vad_ready_state = {}

        def delayed_start(sample_rate: int):
            time.sleep(0.05)
            ready_event.set()
            return streaming_stt.session

        def on_record():
            vad_ready_state["ready"] = ready_event.is_set()

        streaming_stt.start_streaming = delayed_start
        assistant.stt = streaming_stt
        assistant.vad = FakeVad(
            [Utterance(pcm=b"streamed", sample_rate=16000, duration_ms=80)],
            on_record=on_record,
        )
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        with contextlib.redirect_stdout(io.StringIO()):
            reply, transcript = assistant._run_audio_turn(
                WakeEvent(engine="test", confidence=1.0, label="wake"),
                wake_ms=0,
            )

        self.assertEqual(transcript, "streamed")
        self.assertEqual(reply.text, "reply 1")
        self.assertFalse(vad_ready_state["ready"])
        self.assertEqual(streaming_stt.session.written, [b"streamed"])

    def test_audio_turn_uses_preopened_streaming_stt_handle(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            wake=WakeConfig(engine="disabled"),
            conversation=ConversationConfig(follow_up_seconds=0),
            llm=LlmConfig(system_prompt="system"),
        )
        assistant = VoiceAssistant(config)
        assistant.vad = FakeVad([Utterance(pcm=b"streamed", sample_rate=16000, duration_ms=80)])
        streaming_stt = FakeStreamingStt("streamed")
        assistant.stt = streaming_stt
        assistant.chat = RecordingChat()
        assistant.tts = FakeTts()

        assistant._start_standby_streaming_stt(phase="test")
        standby = assistant._standby_stt_handle
        self.assertIsNotNone(standby)
        self.assertTrue(standby.wait_ready(timeout=1.0))

        with contextlib.redirect_stdout(io.StringIO()):
            reply, transcript = assistant._run_audio_turn(
                WakeEvent(engine="test", confidence=1.0, label="wake"),
                wake_ms=0,
            )

        self.assertEqual(transcript, "streamed")
        self.assertEqual(reply.text, "reply 1")
        self.assertIsNone(assistant._standby_stt_handle)
        self.assertEqual(streaming_stt.start_sample_rates, [16000])
        self.assertEqual(streaming_stt.session.written, [b"streamed"])
        self.assertTrue(streaming_stt.session.finished)

    def test_barge_in_streams_stt_and_stashes_transcript(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="audio"),
            conversation=ConversationConfig(barge_in_enabled=True),
        )
        assistant = VoiceAssistant(config)
        assistant.vad = FakeVad([Utterance(pcm=b"barge", sample_rate=16000, duration_ms=80)])
        streaming_stt = FakeStreamingStt("barge transcript")
        assistant.stt = streaming_stt
        assistant.tts = BargeFirstTts()

        with contextlib.redirect_stdout(io.StringIO()):
            utterance = assistant._speak_with_barge_in("reply")

        self.assertIsInstance(utterance, Utterance)
        self.assertEqual(assistant._pending_barge_transcript, "barge transcript")
        self.assertEqual(streaming_stt.session.written, [b"barge"])
        self.assertTrue(streaming_stt.session.finished)
        self.assertEqual(streaming_stt.fallback_calls, 0)


if __name__ == "__main__":
    unittest.main()
