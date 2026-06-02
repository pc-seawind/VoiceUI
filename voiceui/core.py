from __future__ import annotations

import threading
import time

from voiceui.audio import create_audio_input
from voiceui.debug import DebugRecorder, TurnDebugData
from voiceui.home_assistant import HomeAssistantClient
from voiceui.llm import create_chat_client
from voiceui.models import AssistantConfig, AssistantReply, Utterance, WakeEvent
from voiceui.session import ConversationSession
from voiceui.stt import create_stt
from voiceui.tts import create_tts
from voiceui.vad import SpeechStartTimeoutError, create_vad_recorder
from voiceui.wake import create_wake_detector
from voiceui.wake_ack import create_wake_ack_player


class _WakeAckHandle:
    def __init__(
        self,
        thread: threading.Thread,
        result: dict[str, int],
    ):
        self.thread = thread
        self.result = result

    def join(self) -> int:
        self.thread.join()
        return self.result.get("latency_ms", 0)


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
        self.wake_ack = create_wake_ack_player(
            config.wake_ack,
            fallback_device=config.tts.playback_device,
        )
        self.home = HomeAssistantClient(config.home_assistant)
        self.session = ConversationSession(config.llm, config.conversation)
        self.debug = DebugRecorder(config.debug)
        self._pending_barge_utterance: Utterance | None = None

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
        barge_utterance = None
        if self._should_listen_for_barge_in():
            barge_utterance = self._speak_with_barge_in(response)
        else:
            self.tts.speak(response)
        timings["tts"] = int((time.monotonic() - tts_started) * 1000)
        if barge_utterance is not None:
            self._pending_barge_utterance = barge_utterance
            timings["barge_in"] = barge_utterance.duration_ms
        print(f"tts> latency_ms={timings['tts']}")
        return AssistantReply(text=response), timings

    def _should_listen_for_barge_in(self) -> bool:
        return (
            self.config.input.mode == "audio"
            and self.config.conversation.barge_in_enabled
        )

    def _speak_with_barge_in(self, text: str) -> Utterance | None:
        playback_stop_event = threading.Event()
        monitor_stop_event = threading.Event()
        state: dict[str, object] = {}
        check_seconds = max(0.05, self.config.conversation.barge_in_check_seconds)

        def on_speech_start() -> None:
            if not playback_stop_event.is_set():
                print("barge_in> speech_start")
            playback_stop_event.set()

        def monitor() -> None:
            while not monitor_stop_event.is_set():
                try:
                    utterance = self.vad.record(
                        self.command_audio,
                        start_timeout_seconds=check_seconds,
                        stop_event=monitor_stop_event,
                        on_speech_start=on_speech_start,
                    )
                except SpeechStartTimeoutError:
                    continue
                except Exception as exc:
                    state["error"] = exc
                    return
                state["utterance"] = utterance
                return

        monitor_thread = threading.Thread(
            target=monitor,
            name="voiceui-barge-in",
            daemon=True,
        )
        monitor_thread.start()

        try:
            self.tts.speak(text, stop_event=playback_stop_event)
        finally:
            if playback_stop_event.is_set():
                max_wait_seconds = max(
                    1.0,
                    self.config.vad.max_speech_ms / 1000
                    + self.config.vad.silence_ms / 1000
                    + 1.0,
                )
                monitor_thread.join(timeout=max_wait_seconds)
                if monitor_thread.is_alive():
                    monitor_stop_event.set()
                    monitor_thread.join(timeout=1.0)
                    print("barge_in> timeout waiting_for_utterance")
            else:
                monitor_stop_event.set()
                monitor_thread.join(timeout=1.0)

        error = state.get("error")
        if isinstance(error, Exception):
            print(f"barge_in> error={error}")
            return None

        utterance = state.get("utterance")
        if isinstance(utterance, Utterance):
            print(f"barge_in> captured duration_ms={utterance.duration_ms}")
            return utterance
        return None

    def _wait_for_wake(self) -> tuple[WakeEvent, int]:
        wake_started = time.monotonic()
        wake = self.wake.wait(self.wake_audio)
        wake_ms = int((time.monotonic() - wake_started) * 1000)
        print(
            f"wake> engine={wake.engine} label={wake.label} "
            f"confidence={wake.confidence:.3f} latency_ms={wake_ms}"
        )
        return wake, wake_ms

    def _start_wake_ack(self) -> _WakeAckHandle:
        result: dict[str, int] = {}

        def play() -> None:
            ack_started = time.monotonic()
            try:
                self.wake_ack.play()
            except Exception as exc:
                print(f"wake_ack> error={exc}")
                result["latency_ms"] = 0
                return
            ack_ms = int((time.monotonic() - ack_started) * 1000)
            result["latency_ms"] = ack_ms
            if ack_ms:
                print(f"wake_ack> latency_ms={ack_ms} mode=background")

        thread = threading.Thread(target=play, name="voiceui-wake-ack", daemon=True)
        thread.start()
        return _WakeAckHandle(thread=thread, result=result)

    def _run_audio_turn(
        self,
        wake: WakeEvent,
        wake_ms: int,
        wake_ack_handle: _WakeAckHandle | None = None,
        speech_start_timeout_seconds: float = 0.0,
    ) -> tuple[AssistantReply, str]:
        if self._pending_barge_utterance is not None:
            utterance = self._pending_barge_utterance
            self._pending_barge_utterance = None
            vad_ms = 0
            print(f"vad> source=barge_in duration_ms={utterance.duration_ms} latency_ms=0")
        else:
            vad_started = time.monotonic()
            utterance = self.vad.record(
                self.command_audio,
                start_timeout_seconds=speech_start_timeout_seconds,
            )
            vad_ms = int((time.monotonic() - vad_started) * 1000)
            print(f"vad> duration_ms={utterance.duration_ms} latency_ms={vad_ms}")

        wake_ack_ms = wake_ack_handle.join() if wake_ack_handle is not None else 0

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
            "wake_ack": wake_ack_ms,
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
        wake_ack_handle = self._start_wake_ack()
        reply, _transcript = self._run_audio_turn(
            wake,
            wake_ms,
            wake_ack_handle=wake_ack_handle,
        )
        return reply

    def run_conversation(self) -> AssistantReply:
        if self.config.input.mode == "text":
            return self.run_once()

        wake, wake_ms = self._wait_for_wake()
        wake_ack_handle = self._start_wake_ack()
        self.session.reset()
        reply, transcript = self._run_audio_turn(
            wake,
            wake_ms,
            wake_ack_handle=wake_ack_handle,
        )
        follow_up_seconds = self.config.conversation.follow_up_seconds
        if (follow_up_seconds <= 0 or not transcript) and self._pending_barge_utterance is None:
            return reply

        while True:
            if self._pending_barge_utterance is None:
                if follow_up_seconds <= 0:
                    return reply
                print(f"session> listening_for_follow_up seconds={follow_up_seconds}")
                speech_start_timeout_seconds = follow_up_seconds
            else:
                print("session> processing_barge_in")
                speech_start_timeout_seconds = 0.0
            follow_up_wake = WakeEvent(
                engine="follow_up",
                confidence=1.0,
                label="no_wake",
            )
            try:
                reply, transcript = self._run_audio_turn(
                    follow_up_wake,
                    wake_ms=0,
                    speech_start_timeout_seconds=speech_start_timeout_seconds,
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
