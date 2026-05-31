from __future__ import annotations

import contextlib
import io
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
    def __init__(self, items: list[Utterance | type[SpeechStartTimeoutError]]):
        self.items = items
        self.start_timeouts: list[float] = []

    def record(self, _audio, start_timeout_seconds: float = 0.0) -> Utterance:
        self.start_timeouts.append(start_timeout_seconds)
        item = self.items.pop(0)
        if item is SpeechStartTimeoutError:
            raise SpeechStartTimeoutError("Timed out waiting for speech.")
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


class FakeTts:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class CoreTests(unittest.TestCase):
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
        assistant.wake = FakeWake()
        assistant.vad = fake_vad
        assistant.stt = FakeStt()
        assistant.chat = chat
        assistant.tts = tts

        with contextlib.redirect_stdout(io.StringIO()):
            reply = assistant.run_conversation()

        self.assertIsInstance(reply, AssistantReply)
        self.assertEqual(fake_vad.start_timeouts, [0.0, 1, 1])
        self.assertEqual(tts.spoken, ["reply 1", "reply 2"])
        self.assertEqual([message.content for message in chat.calls[0]], ["system", "first"])
        self.assertEqual(
            [message.content for message in chat.calls[1]],
            ["system", "first", "reply 1", "second"],
        )


if __name__ == "__main__":
    unittest.main()
