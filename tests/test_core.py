from __future__ import annotations

import contextlib
import io
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from pathlib import Path

from voiceui.core import VoiceAssistant
from voiceui.llm import ChatMessage
from voiceui.models import (
    AssistantConfig,
    AssistantReply,
    ConversationConfig,
    DebugConfig,
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

    def chunks(self) -> Iterator[bytes]:
        while True:
            yield self.chunk


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
            debug_dirs = list(Path(temp_dir).glob("*-barge-in-*"))
            self.assertEqual(len(debug_dirs), 1)
            self.assertTrue((debug_dirs[0] / "barge_in_monitor.wav").exists())
            self.assertTrue((debug_dirs[0] / "metadata.json").exists())

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
