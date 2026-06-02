from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator

from voiceui.audio import RecordingAudioInput, create_audio_input
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


class _StreamingSttHandle:
    def __init__(self, stt, sample_rate: int):
        self.stt = stt
        self.sample_rate = sample_rate
        self.requested_at: float | None = None
        self.ready_at: float | None = None
        self.completed_at: float | None = None
        self.sent_chunks = 0
        self.result = ""
        self.error: Exception | None = None
        self._done = object()
        self._items: queue.Queue[bytes | object] = queue.Queue()
        self._session = None
        self._thread = threading.Thread(
            target=self._run,
            name="voiceui-stt-stream",
            daemon=True,
        )

    def start(self) -> None:
        self.requested_at = time.monotonic()
        self._thread.start()

    def write(self, pcm: bytes) -> None:
        if pcm:
            self.sent_chunks += 1
            self._items.put(pcm)

    def finish(self) -> str:
        self._items.put(self._done)
        self._thread.join()
        if self.error is not None:
            raise self.error
        return self.result

    def abort(self) -> None:
        self._items.put(self._done)
        if self._session is not None:
            self._session.abort()
        self._thread.join(timeout=1.0)

    def ready_latency_ms(self) -> int | None:
        if self.requested_at is None or self.ready_at is None:
            return None
        return int((self.ready_at - self.requested_at) * 1000)

    def total_latency_ms(self) -> int:
        if self.requested_at is None:
            return 0
        completed_at = self.completed_at or time.monotonic()
        return int((completed_at - self.requested_at) * 1000)

    def _run(self) -> None:
        try:
            self._session = self.stt.start_streaming(self.sample_rate)
            self.ready_at = time.monotonic()
            while True:
                item = self._items.get()
                if item is self._done:
                    break
                self._session.write(item)
            self.result = self._session.finish()
            self.completed_at = time.monotonic()
        except Exception as exc:
            self.error = exc
            if self._session is not None:
                self._session.abort()
            self.completed_at = time.monotonic()


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
        self._pending_barge_transcript: str | None = None
        self._pending_barge_stt_ms = 0
        self._pending_barge_stt_extra_timings: dict[str, int] = {}
        if audio_enabled:
            self._warm_up_audio_path()
        self._print_barge_in_config()

    def _warm_up_audio_path(self) -> None:
        warm_up_started = time.monotonic()
        try:
            warmed = self.vad.warm_up()
        except Exception as exc:
            print(f"vad> warm_up_error={exc}")
            return
        if warmed:
            latency_ms = int((time.monotonic() - warm_up_started) * 1000)
            print(f"vad> warmed_up latency_ms={latency_ms}")

    def run_text_turn(self, text: str) -> AssistantReply:
        transcript = text.strip()
        if not transcript:
            return AssistantReply(text="I did not hear anything.", routed_to="system")

        reply, _timings = self._complete_transcript(transcript)
        return reply

    def _complete_transcript(self, transcript: str) -> tuple[AssistantReply, dict[str, int]]:
        self.session.add_user(transcript)
        timings: dict[str, int] = {}
        barge_utterance = None

        if self.config.llm.stream:
            response, barge_utterance = self._stream_and_speak_response(timings)
        else:
            llm_started = time.monotonic()
            response = self.chat.complete(self.session.messages)
            timings["llm"] = int((time.monotonic() - llm_started) * 1000)
            print(f"llm> latency_ms={timings['llm']}")
            if not response:
                response = "I could not produce a response."
            self.session.add_assistant(response)

            tts_started = time.monotonic()
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

    def _stream_and_speak_response(self, timings: dict[str, int]) -> tuple[str, Utterance | None]:
        messages = list(self.session.messages)
        llm_stop_event = threading.Event()
        llm_stream_stats: dict[str, int] = {}
        text_chunks = self._start_tracked_llm_stream(
            messages,
            timings,
            stop_event=llm_stop_event,
            stream_stats=llm_stream_stats,
        )
        barge_utterance = None
        tts_started = time.monotonic()
        try:
            if self._should_listen_for_barge_in():
                response, barge_utterance = self._speak_text_stream_with_barge_in(
                    text_chunks,
                    llm_stop_event=llm_stop_event,
                )
            else:
                response = self.tts.speak_text_stream(text_chunks)
        finally:
            llm_stop_event.set()
        timings["tts"] = int((time.monotonic() - tts_started) * 1000)
        self._print_streaming_llm_stats(timings, llm_stream_stats)

        if not response:
            response = "I could not produce a response."
            fallback_tts_started = time.monotonic()
            if self._should_listen_for_barge_in():
                barge_utterance = self._speak_with_barge_in(response)
            else:
                self.tts.speak(response)
            timings["tts"] += int((time.monotonic() - fallback_tts_started) * 1000)
        self.session.add_assistant(response)
        return response, barge_utterance

    def _start_tracked_llm_stream(
        self,
        messages: list,
        timings: dict[str, int],
        stop_event: threading.Event | None = None,
        stream_stats: dict[str, int] | None = None,
    ) -> Iterator[str]:
        items: queue.Queue[object] = queue.Queue()
        done = object()

        def producer() -> None:
            llm_started = time.monotonic()
            first_token_ms: int | None = None
            chunks = 0
            try:
                for chunk in self.chat.stream_complete(messages):
                    if stop_event is not None and stop_event.is_set():
                        break
                    if not chunk:
                        continue
                    if first_token_ms is None:
                        first_token_ms = int((time.monotonic() - llm_started) * 1000)
                        timings["llm_first_token"] = first_token_ms
                    chunks += 1
                    items.put(chunk)
            except Exception as exc:
                items.put(exc)
            finally:
                timings["llm"] = int((time.monotonic() - llm_started) * 1000)
                if first_token_ms is None:
                    timings["llm_first_token"] = timings["llm"]
                if stream_stats is not None:
                    stream_stats["chunks"] = chunks
                items.put(done)

        thread = threading.Thread(target=producer, name="voiceui-llm-stream", daemon=True)
        thread.start()

        while True:
            if stop_event is not None and stop_event.is_set() and items.empty():
                break
            try:
                item = items.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            if (
                stream_stats is not None
                and not stream_stats.get("first_token_logged")
                and "llm_first_token" in timings
            ):
                stream_stats["first_token_logged"] = 1
                print(f"llm> first_token_ms={timings['llm_first_token']}")
            yield item  # type: ignore[misc]

    def _print_streaming_llm_stats(
        self,
        timings: dict[str, int],
        stream_stats: dict[str, int],
    ) -> None:
        if "llm" not in timings:
            return
        print(
            "llm> "
            f"latency_ms={timings['llm']} "
            f"first_token_ms={timings.get('llm_first_token', timings['llm'])} "
            f"stream_chunks={stream_stats.get('chunks', 0)}"
        )

    def _should_listen_for_barge_in(self) -> bool:
        return (
            self.config.input.mode == "audio"
            and self.config.conversation.barge_in_enabled
        )

    def _print_barge_in_config(self) -> None:
        enabled = self._should_listen_for_barge_in()
        reason = "enabled"
        if self.config.input.mode != "audio":
            reason = f"input_mode_{self.config.input.mode}"
        elif not self.config.conversation.barge_in_enabled:
            reason = "conversation_disabled"
        print(
            "barge_in> "
            f"enabled={str(enabled).lower()} reason={reason} "
            f"input_mode={self.config.input.mode} "
            f"conversation_enabled={str(self.config.conversation.barge_in_enabled).lower()} "
            f"vad_engine={self.config.vad.engine} threshold={self.config.vad.threshold} "
            f"command_channel={self.config.audio.command_stream_channel}"
        )

    def _print_barge_in_monitor_started(self, mode: str) -> None:
        print(
            "barge_in> monitor_started "
            f"mode={mode} start_timeout_seconds=0.0 "
            f"vad_engine={self.config.vad.engine} threshold={self.config.vad.threshold} "
            f"command_channel={self.config.audio.command_stream_channel}"
        )

    def _save_barge_in_monitor(
        self,
        *,
        mode: str,
        state: dict[str, object],
        monitor_audio: RecordingAudioInput,
    ) -> None:
        pcm = monitor_audio.pcm()
        result = "no_speech"
        utterance = state.get("utterance")
        if isinstance(utterance, Utterance):
            result = "captured"
        elif "error" in state:
            result = "error"

        metadata: dict[str, object] = {
            "vad_engine": self.config.vad.engine,
            "vad_threshold": self.config.vad.threshold,
            "command_channel": self.config.audio.command_stream_channel,
        }
        if isinstance(utterance, Utterance):
            metadata["utterance_duration_ms"] = utterance.duration_ms
        if "timeout" in state:
            metadata["timeout"] = str(state["timeout"])
        if "error" in state:
            metadata["error"] = str(state["error"])

        debug_dir = self.debug.save_barge_in_monitor(
            mode=mode,
            result=result,
            pcm=pcm,
            sample_rate=monitor_audio.sample_rate,
            duration_ms=monitor_audio.duration_ms(),
            metadata=metadata,
        )
        if debug_dir:
            print(
                "barge_in> monitor_saved="
                f"{debug_dir} duration_ms={monitor_audio.duration_ms()} result={result}"
            )

    def _record_barge_in_utterance(
        self,
        *,
        monitor_audio: RecordingAudioInput,
        monitor_stop_event: threading.Event,
        on_speech_start,
        mode: str,
    ) -> tuple[Utterance, str | None, int, dict[str, int]]:
        if not self._should_stream_stt():
            utterance = self.vad.record(
                monitor_audio,
                start_timeout_seconds=0.0,
                stop_event=monitor_stop_event,
                on_speech_start=on_speech_start,
            )
            return utterance, None, 0, {}

        stream_handle: _StreamingSttHandle | None = None
        stt_start_reference = time.monotonic()

        def start_streaming_stt() -> None:
            nonlocal stream_handle
            if stream_handle is not None:
                return
            stream_handle = _StreamingSttHandle(self.stt, monitor_audio.sample_rate)
            stream_handle.start()
            start_ms = int((time.monotonic() - stt_start_reference) * 1000)
            print(f"stt> streaming_started source=barge_in mode={mode} elapsed_ms={start_ms}")

        def combined_speech_start() -> None:
            on_speech_start()
            start_streaming_stt()

        def on_speech_audio(pcm: bytes) -> None:
            if stream_handle is None:
                start_streaming_stt()
            assert stream_handle is not None
            stream_handle.write(pcm)

        try:
            utterance = self.vad.record(
                monitor_audio,
                start_timeout_seconds=0.0,
                stop_event=monitor_stop_event,
                on_speech_start=combined_speech_start,
                on_speech_audio=on_speech_audio,
            )
        except Exception:
            if stream_handle is not None:
                stream_handle.abort()
            raise

        if stream_handle is None:
            return utterance, None, 0, {}

        finalize_started = time.monotonic()
        transcript = stream_handle.finish()
        stt_ms = int((time.monotonic() - finalize_started) * 1000)
        stt_total_ms = stream_handle.total_latency_ms()
        ready_ms = stream_handle.ready_latency_ms()
        ready_fragment = f" ready_ms={ready_ms}" if ready_ms is not None else ""
        print(
            "stt> "
            f"latency_ms={stt_ms} mode=streaming source=barge_in "
            f"total_latency_ms={stt_total_ms} sent_chunks={stream_handle.sent_chunks}"
            f"{ready_fragment} text={transcript}"
        )
        return utterance, transcript, stt_ms, {"stt_total": stt_total_ms}

    def _speak_with_barge_in(self, text: str) -> Utterance | None:
        playback_stop_event = threading.Event()
        monitor_stop_event = threading.Event()
        state: dict[str, object] = {}
        monitor_audio = RecordingAudioInput(self.command_audio)

        def on_speech_start() -> None:
            if not playback_stop_event.is_set():
                print("barge_in> speech_start")
            playback_stop_event.set()

        def monitor() -> None:
            try:
                utterance, transcript, stt_ms, stt_extra = self._record_barge_in_utterance(
                    monitor_audio=monitor_audio,
                    monitor_stop_event=monitor_stop_event,
                    on_speech_start=on_speech_start,
                    mode="full",
                )
            except SpeechStartTimeoutError as exc:
                state["timeout"] = str(exc)
                return
            except Exception as exc:
                state["error"] = exc
                return
            state["utterance"] = utterance
            if transcript is not None:
                state["transcript"] = transcript
                state["stt_ms"] = stt_ms
                state["stt_extra_timings"] = stt_extra

        monitor_thread = threading.Thread(
            target=monitor,
            name="voiceui-barge-in",
            daemon=True,
        )
        self._print_barge_in_monitor_started("full")
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

        self._save_barge_in_monitor(
            mode="full",
            state=state,
            monitor_audio=monitor_audio,
        )

        error = state.get("error")
        if isinstance(error, Exception):
            print(f"barge_in> error={error}")
            return None

        utterance = state.get("utterance")
        if isinstance(utterance, Utterance):
            transcript = state.get("transcript")
            if isinstance(transcript, str):
                self._pending_barge_transcript = transcript
                self._pending_barge_stt_ms = int(state.get("stt_ms") or 0)
                extra_timings = state.get("stt_extra_timings")
                self._pending_barge_stt_extra_timings = (
                    dict(extra_timings) if isinstance(extra_timings, dict) else {}
                )
            print(f"barge_in> captured duration_ms={utterance.duration_ms}")
            return utterance
        print("barge_in> no_speech")
        return None

    def _speak_text_stream_with_barge_in(
        self,
        text_chunks: Iterator[str],
        llm_stop_event: threading.Event | None = None,
    ) -> tuple[str, Utterance | None]:
        playback_stop_event = threading.Event()
        monitor_stop_event = threading.Event()
        state: dict[str, object] = {}
        monitor_audio = RecordingAudioInput(self.command_audio)

        def on_speech_start() -> None:
            if not playback_stop_event.is_set():
                print("barge_in> speech_start")
            playback_stop_event.set()
            if llm_stop_event is not None:
                llm_stop_event.set()

        def monitor() -> None:
            try:
                utterance, transcript, stt_ms, stt_extra = self._record_barge_in_utterance(
                    monitor_audio=monitor_audio,
                    monitor_stop_event=monitor_stop_event,
                    on_speech_start=on_speech_start,
                    mode="stream",
                )
            except SpeechStartTimeoutError as exc:
                state["timeout"] = str(exc)
                return
            except Exception as exc:
                state["error"] = exc
                return
            state["utterance"] = utterance
            if transcript is not None:
                state["transcript"] = transcript
                state["stt_ms"] = stt_ms
                state["stt_extra_timings"] = stt_extra

        monitor_thread = threading.Thread(
            target=monitor,
            name="voiceui-barge-in",
            daemon=True,
        )
        self._print_barge_in_monitor_started("stream")
        monitor_thread.start()

        try:
            text = self.tts.speak_text_stream(text_chunks, stop_event=playback_stop_event)
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

        self._save_barge_in_monitor(
            mode="stream",
            state=state,
            monitor_audio=monitor_audio,
        )

        error = state.get("error")
        if isinstance(error, Exception):
            print(f"barge_in> error={error}")
            return text, None

        utterance = state.get("utterance")
        if isinstance(utterance, Utterance):
            transcript = state.get("transcript")
            if isinstance(transcript, str):
                self._pending_barge_transcript = transcript
                self._pending_barge_stt_ms = int(state.get("stt_ms") or 0)
                extra_timings = state.get("stt_extra_timings")
                self._pending_barge_stt_extra_timings = (
                    dict(extra_timings) if isinstance(extra_timings, dict) else {}
                )
            print(f"barge_in> captured duration_ms={utterance.duration_ms}")
            return text, utterance
        print("barge_in> no_speech")
        return text, None

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
        stt_extra_timings: dict[str, int] = {}
        if self._pending_barge_utterance is not None:
            utterance = self._pending_barge_utterance
            pending_transcript = self._pending_barge_transcript
            self._pending_barge_utterance = None
            self._pending_barge_transcript = None
            vad_ms = 0
            print(f"vad> source=barge_in duration_ms={utterance.duration_ms} latency_ms=0")
            wake_ack_ms = wake_ack_handle.join() if wake_ack_handle is not None else 0
            if pending_transcript is not None:
                transcript = pending_transcript
                stt_ms = self._pending_barge_stt_ms
                stt_extra_timings = dict(self._pending_barge_stt_extra_timings)
                self._pending_barge_stt_ms = 0
                self._pending_barge_stt_extra_timings = {}
                print(f"stt> source=barge_in_stream latency_ms={stt_ms} text={transcript}")
            else:
                stt_started = time.monotonic()
                transcript = self.stt.transcribe(utterance)
                stt_ms = int((time.monotonic() - stt_started) * 1000)
                print(f"stt> latency_ms={stt_ms} text={transcript}")
        elif self._should_stream_stt():
            (
                utterance,
                transcript,
                vad_ms,
                stt_ms,
                stt_extra_timings,
            ) = self._record_and_stream_transcribe(
                speech_start_timeout_seconds=speech_start_timeout_seconds,
            )
            wake_ack_ms = wake_ack_handle.join() if wake_ack_handle is not None else 0
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
            **stt_extra_timings,
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
        debug_dir = self.debug.save_turn(debug_data, utterance, wake_audio=wake)
        if debug_dir:
            print(f"debug> saved={debug_dir}")
        return reply, transcript

    def _should_stream_stt(self) -> bool:
        supports_streaming = getattr(self.stt, "supports_streaming", None)
        return bool(callable(supports_streaming) and supports_streaming())

    def _record_and_stream_transcribe(
        self,
        *,
        speech_start_timeout_seconds: float,
    ) -> tuple[Utterance, str, int, int, dict[str, int]]:
        vad_started = time.monotonic()
        stream_handle: _StreamingSttHandle | None = None

        def on_speech_start() -> None:
            nonlocal stream_handle
            if stream_handle is not None:
                return
            stream_handle = _StreamingSttHandle(self.stt, self.command_audio.sample_rate)
            stream_handle.start()
            start_ms = int((time.monotonic() - vad_started) * 1000)
            print(f"stt> streaming_started vad_elapsed_ms={start_ms}")

        def on_speech_audio(pcm: bytes) -> None:
            if stream_handle is None:
                on_speech_start()
            assert stream_handle is not None
            stream_handle.write(pcm)

        try:
            utterance = self.vad.record(
                self.command_audio,
                start_timeout_seconds=speech_start_timeout_seconds,
                on_speech_start=on_speech_start,
                on_speech_audio=on_speech_audio,
            )
        except Exception:
            if stream_handle is not None:
                stream_handle.abort()
            raise

        vad_ms = int((time.monotonic() - vad_started) * 1000)
        print(f"vad> duration_ms={utterance.duration_ms} latency_ms={vad_ms}")

        if stream_handle is None:
            stt_started = time.monotonic()
            transcript = self.stt.transcribe(utterance)
            stt_ms = int((time.monotonic() - stt_started) * 1000)
            print(f"stt> latency_ms={stt_ms} mode=fallback text={transcript}")
            return utterance, transcript, vad_ms, stt_ms, {}

        finalize_started = time.monotonic()
        transcript = stream_handle.finish()
        stt_ms = int((time.monotonic() - finalize_started) * 1000)
        stt_total_ms = stream_handle.total_latency_ms()
        ready_ms = stream_handle.ready_latency_ms()
        ready_fragment = f" ready_ms={ready_ms}" if ready_ms is not None else ""
        print(
            "stt> "
            f"latency_ms={stt_ms} mode=streaming total_latency_ms={stt_total_ms} "
            f"sent_chunks={stream_handle.sent_chunks}{ready_fragment} text={transcript}"
        )
        return utterance, transcript, vad_ms, stt_ms, {"stt_total": stt_total_ms}

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
