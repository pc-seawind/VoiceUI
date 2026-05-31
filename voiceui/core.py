from __future__ import annotations

import time

from voiceui.audio import create_audio_input
from voiceui.home_assistant import HomeAssistantClient
from voiceui.llm import create_chat_client
from voiceui.models import AssistantConfig, AssistantReply
from voiceui.session import ConversationSession
from voiceui.stt import create_stt
from voiceui.tts import create_tts
from voiceui.vad import create_vad_recorder
from voiceui.wake import create_wake_detector


class VoiceAssistant:
    def __init__(self, config: AssistantConfig):
        self.config = config
        audio_enabled = config.input.mode == "audio"
        self.wake_audio = create_audio_input(
            config.audio,
            enabled=audio_enabled,
            selected_channel=config.audio.wake_stream_channel,
        )
        self.command_audio = create_audio_input(
            config.audio,
            enabled=audio_enabled,
            selected_channel=config.audio.command_stream_channel,
        )
        self.wake = create_wake_detector(config.wake)
        self.vad = create_vad_recorder(config.vad)
        self.stt = create_stt(config.stt)
        self.chat = create_chat_client(config.llm)
        self.tts = create_tts(config.tts)
        self.home = HomeAssistantClient(config.home_assistant)
        self.session = ConversationSession(config.llm, config.conversation)

    def run_text_turn(self, text: str) -> AssistantReply:
        transcript = text.strip()
        if not transcript:
            return AssistantReply(text="I did not hear anything.", routed_to="system")

        self.session.add_user(transcript)
        response = self.chat.complete(self.session.messages)
        if not response:
            response = "I could not produce a response."
        self.session.add_assistant(response)
        self.tts.speak(response)
        return AssistantReply(text=response)

    def run_once(self) -> AssistantReply:
        if self.config.input.mode == "text":
            text = input("you> ")
            return self.run_text_turn(text)

        wake_started = time.monotonic()
        wake = self.wake.wait(self.wake_audio)
        wake_ms = int((time.monotonic() - wake_started) * 1000)
        print(
            f"wake> engine={wake.engine} label={wake.label} "
            f"confidence={wake.confidence:.3f} latency_ms={wake_ms}"
        )

        vad_started = time.monotonic()
        utterance = self.vad.record(self.command_audio)
        vad_ms = int((time.monotonic() - vad_started) * 1000)
        print(f"vad> duration_ms={utterance.duration_ms} latency_ms={vad_ms}")

        stt_started = time.monotonic()
        transcript = self.stt.transcribe(utterance)
        stt_ms = int((time.monotonic() - stt_started) * 1000)
        print(f"stt> latency_ms={stt_ms} text={transcript}")

        return self.run_text_turn(transcript)

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"error> {exc}")
                time.sleep(1)
