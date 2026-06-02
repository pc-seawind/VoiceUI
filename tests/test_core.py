from __future__ import annotations

import contextlib
import io
import threading
import time
import unittest

from voiceui.core import VoiceAssistant
from voiceui.llm import ChatMessage
from voiceui.models import (
    AssistantConfig,
    AssistantReply,
    ConversationConfig,
    InputConfig,
    LlmConfig,
    Utterance,
    WakeConfig,
    WakeEvent,
)
from voiceui.vad import SpeechStartTimeoutError


class FakeWake:
    def wait(self, _audio) -> WakeEvent:
        return WakeEvent(engine="test_wake", confidence=1.0, label="wake")


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
        return item


class FakeStt:
    def transcribe(self, utterance: Utterance) -> str:
        return utterance.pcm.decode("utf-8")


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


class FakeTts:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        self.spoken.append(text)

    def speak_text_stream(self, text_chunks, stop_event: threading.Event | None = None) -> str:
        text = "".join(text_chunks)
        self.spoken.append(text)
        return text


class BargeFirstTts(FakeTts):
    def speak(self, text: str, stop_event: threading.Event | None = None) -> None:
        self.spoken.append(text)
        if len(self.spoken) != 1 or stop_event is None:
            return
        deadline = time.monotonic() + 1.0
        while not stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)


class FakeWakeAck:
    def __init__(self):
        self.calls = 0

    def play(self) -> None:
        self.calls += 1


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
            conversation=ConversationConfig(follow_up_seconds=1, max_turns=4),
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

    def test_text_turn_streams_llm_response_to_tts_and_session(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            llm=LlmConfig(system_prompt="system", stream=True),
        )
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

    def test_run_conversation_processes_barge_in_as_next_turn(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
