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
        self.assertEqual(fake_vad.start_timeouts, [0.0, 1, 1])
        self.assertEqual(tts.spoken, ["reply 1", "reply 2"])
        self.assertEqual([message.content for message in chat.calls[0]], ["system", "first"])
        self.assertEqual(
            [message.content for message in chat.calls[1]],
            ["system", "first", "reply 1", "second"],
        )

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

    def test_barge_in_ambiguous_short_text_clarifies_without_session_pollution(self) -> None:
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
        self.assertEqual(reply.text, "我没太听清，你是想找临时用工渠道吗？")
        self.assertEqual(len(assistant.chat.calls), 1)
        self.assertEqual(
            [message.content for message in assistant.session.messages],
            ["system", "first", "reply 1"],
        )
        self.assertEqual(
            assistant.tts.spoken,
            ["reply 1", "我没太听清，你是想找临时用工渠道吗？"],
        )

    def test_input_gate_clarification_does_not_start_another_barge_in(self) -> None:
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
        self.assertEqual(reply.text, "我没太听清，你是想找临时用工渠道吗？")
        self.assertEqual(assistant.tts.spoken, ["reply 1", reply.text])
        self.assertEqual(fake_vad.items, [self_echo])
        self.assertIsNone(assistant._pending_barge_utterance)

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
