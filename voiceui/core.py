from __future__ import annotations

import time

from voiceui.audio import create_audio_input
from voiceui.debug import DebugRecorder, TurnDebugData
from voiceui.home_assistant import HomeAssistantClient
from voiceui.llm import create_chat_client
from voiceui.models import AssistantConfig, AssistantReply, WakeEvent
from voiceui.session import ConversationSession
from voiceui.stt import create_stt
from voiceui.tts import create_tts
from voiceui.vad import SpeechStartTimeoutError, create_vad_recorder
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
        self.debug = DebugRecorder(config.debug)

    def run_text_turn(self, text: str) -> AssistantReply:
        transcript = text.strip()
        if not transcript:
            return AssistantReply(text="I did not hear anything.", routed_to="system")

        reply, _timings = self._complete_transcript(transcript)
        return reply

    def _complete_transcript(self, transcript: str) -> tuple[AssistantReply, dict[str, int]]:
        self.session.add_user(transcript)
        timings: dict[str, int] = {}

        llm_started = time.monotonic()
        response = self.chat.complete(self.session.messages)
        timings["llm"] = int((time.monotonic() - llm_started) * 1000)
        print(f"llm> latency_ms={timings['llm']}")
        if not response:
            response = "I could not produce a response."
        self.session.add_assistant(response)

        tts_started = time.monotonic()
        self.tts.speak(response)
        timings["tts"] = int((time.monotonic() - tts_started) * 1000)
        print(f"tts> latency_ms={timings['tts']}")
        return AssistantReply(text=response), timings

    def _wait_for_wake(self) -> tuple[WakeEvent, int]:
        wake_started = time.monotonic()
        wake = self.wake.wait(self.wake_audio)
        wake_ms = int((time.monotonic() - wake_started) * 1000)
        print(
            f"wake> engine={wake.engine} label={wake.label} "
            f"confidence={wake.confidence:.3f} latency_ms={wake_ms}"
        )
        return wake, wake_ms

    def _run_audio_turn(
        self,
        wake: WakeEvent,
        wake_ms: int,
        speech_start_timeout_seconds: float = 0.0,
    ) -> tuple[AssistantReply, str]:
        vad_started = time.monotonic()
        utterance = self.vad.record(
            self.command_audio,
            start_timeout_seconds=speech_start_timeout_seconds,
        )
        vad_ms = int((time.monotonic() - vad_started) * 1000)
        print(f"vad> duration_ms={utterance.duration_ms} latency_ms={vad_ms}")

        stt_started = time.monotonic()
        transcript = self.stt.transcribe(utterance)
        stt_ms = int((time.monotonic() - stt_started) * 1000)
        print(f"stt> latency_ms={stt_ms} text={transcript}")

        transcript = transcript.strip()
        if not transcript:
            reply = AssistantReply(text="I did not hear anything.", routed_to="system")
            response_timings = {"llm": 0, "tts": 0}
            print("assistant> I did not hear anything.")
        else:
            reply, response_timings = self._complete_transcript(transcript)
        timings = {
            "wake": wake_ms,
            "vad": vad_ms,
            "stt": stt_ms,
            **response_timings,
        }
        debug_data = TurnDebugData(
            node_id=self.config.node.id,
            room=self.config.node.room,
            wake={
                "engine": wake.engine,
                "label": wake.label,
                "confidence": wake.confidence,
            },
            timings_ms=timings,
            utterance={
                "duration_ms": utterance.duration_ms,
                "sample_rate": utterance.sample_rate,
                "bytes": len(utterance.pcm),
            },
            transcript=transcript,
            reply=reply.text,
            routed_to=reply.routed_to,
        )
        debug_dir = self.debug.save_turn(debug_data, utterance)
        if debug_dir:
            print(f"debug> saved={debug_dir}")
        return reply, transcript

    def run_once(self) -> AssistantReply:
        if self.config.input.mode == "text":
            text = input("you> ")
            return self.run_text_turn(text)

        wake, wake_ms = self._wait_for_wake()
        reply, _transcript = self._run_audio_turn(wake, wake_ms)
        return reply

    def run_conversation(self) -> AssistantReply:
        if self.config.input.mode == "text":
            return self.run_once()

        wake, wake_ms = self._wait_for_wake()
        self.session.reset()
        reply, transcript = self._run_audio_turn(wake, wake_ms)
        follow_up_seconds = self.config.conversation.follow_up_seconds
        if follow_up_seconds <= 0 or not transcript:
            return reply

        while True:
            print(f"session> listening_for_follow_up seconds={follow_up_seconds}")
            follow_up_wake = WakeEvent(
                engine="follow_up",
                confidence=1.0,
                label="no_wake",
            )
            try:
                reply, transcript = self._run_audio_turn(
                    follow_up_wake,
                    wake_ms=0,
                    speech_start_timeout_seconds=follow_up_seconds,
                )
            except SpeechStartTimeoutError:
                print("session> follow_up_timeout returning_to_wake")
                return reply
            if not transcript:
                print("session> empty_follow_up returning_to_wake")
                return reply

    def run_forever(self) -> None:
        while True:
            try:
                self.run_conversation()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"error> {exc}")
                time.sleep(1)
