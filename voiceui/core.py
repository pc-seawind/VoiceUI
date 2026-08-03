from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher

from voiceui.audio import RecordingAudioInput, create_audio_input
from voiceui.audio_dump import AudioDumpManager, configure_audio_dump
from voiceui.cron import CronScheduler
from voiceui.debug import DebugRecorder, TurnDebugData
from voiceui.home_assistant import HomeAssistantClient
from voiceui.llm import create_chat_client
from voiceui.logs import configure_log_files, log_event, record_text_event
from voiceui.models import (
    DEFAULT_VOICE_TERMINATION_PHRASES,
    AssistantConfig,
    AssistantReply,
    Utterance,
    WakeEvent,
)
from voiceui.reminders import (
    Reminder,
    ReminderScheduler,
    format_pending_reminders,
    format_reminder_confirmation,
    looks_like_reminder_cancel_text,
    looks_like_reminder_create_text,
    looks_like_reminder_status_text,
    looks_like_reminder_text,
    parse_reminder_request,
    parse_scheduled_command,
)
from voiceui.session import ConversationSession
from voiceui.streaming import BoundedBackpressureQueue, StreamFramePolicy
from voiceui.stt import create_stt
from voiceui.system_volume import get_system_output_volume, set_system_output_volume
from voiceui.tools import (
    create_tool_runner,
    format_current_time_response,
    format_weather_response,
    get_current_time,
    get_current_weather,
    warm_weather_cache,
)
from voiceui.tts import create_tts
from voiceui.vad import SpeechStartTimeoutError, create_vad_recorder
from voiceui.wake import create_wake_detector
from voiceui.wake_ack import create_wake_ack_player

_TOOL_PROGRESS_PROMPT_DELAY_SECONDS = 1.2
_EMPTY_INPUT_RESPONSE = "我没听清。"
_INPUT_CLARIFICATION_RESPONSE = "我没太听清，请再说一遍。"


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


@dataclass(frozen=True, slots=True)
class _SpokenResponse:
    text: str
    normalized: str
    spoken_at: float


@dataclass(frozen=True, slots=True)
class _SelfEchoMatch:
    matched_chars: int
    age_ms: int


class _StreamingSttHandle:
    _STOP = object()

    def __init__(
        self,
        stt,
        sample_rate: int,
        *,
        policy: StreamFramePolicy | None = None,
        max_buffered_chunks: int | None = None,
    ):
        self.stt = stt
        self.sample_rate = sample_rate
        if policy is None:
            policy = StreamFramePolicy(
                frame_ms=20,
                buffer_frames=max(1, max_buffered_chunks or 1),
                ready_timeout_seconds=0.5,
            )
        self.policy = policy
        self.max_buffered_chunks = max(1, policy.buffer_frames)
        self.requested_at: float | None = None
        self.ready_at: float | None = None
        self.completed_at: float | None = None
        self.sent_chunks = 0
        self.dropped_chunks = 0
        self.result = ""
        self.error: Exception | None = None
        self._session = None
        self._write_queue: queue.Queue[bytes | object] = queue.Queue()
        self._ready_event = threading.Event()
        self._writer_done = threading.Event()
        self._started = False
        self._closed = False
        self._abort_requested = False
        self._io_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._started = True
            self.requested_at = time.monotonic()
            self._thread = threading.Thread(
                target=self._open_session,
                name="voiceui-stt-stream-open",
                daemon=True,
            )
            self._thread.start()
            self._writer_thread = threading.Thread(
                target=self._write_loop,
                name="voiceui-stt-stream-writer",
                daemon=True,
            )
            self._writer_thread.start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        if not self._started:
            return False
        return self._ready_event.wait(timeout=timeout)

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        if not self._started:
            self.start()
        if self.error is not None and self._ready_event.is_set():
            raise self.error
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Streaming STT session is already closed.")
            # Do not call the cloud session from the VAD/audio capture thread.
            # Some providers pace writes internally; blocking here can stop
            # sounddevice reads long enough to create audible gaps in the
            # captured utterance.
            self._write_queue.put(pcm)

    def finish(self) -> str:
        if not self._started:
            self.start()
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._write_queue.put(self._STOP)
        if self._writer_thread is not None:
            self._writer_thread.join()
        session = self._session_or_raise()
        with self._io_lock:
            try:
                self.result = session.finish()
            except Exception as exc:
                self.error = exc
                raise
            finally:
                self.completed_at = time.monotonic()
        if self.error is not None:
            raise self.error
        return self.result

    def abort(self) -> None:
        if not self._started:
            self.completed_at = time.monotonic()
            return
        with self._state_lock:
            self._abort_requested = True
            if not self._closed:
                self._closed = True
                self._write_queue.put(self._STOP)
        if self.wait_ready(timeout=1.0) and self._session is not None:
            with self._io_lock:
                if self.completed_at is None:
                    self._session.abort()
                    self.completed_at = time.monotonic()
        else:
            self.completed_at = time.monotonic()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=1.0)

    def ready_latency_ms(self) -> int | None:
        if self.requested_at is None or self.ready_at is None:
            return None
        return int((self.ready_at - self.requested_at) * 1000)

    def total_latency_ms(self) -> int:
        if self.requested_at is None:
            return 0
        completed_at = self.completed_at or time.monotonic()
        return int((completed_at - self.requested_at) * 1000)

    def _session_or_raise(self):
        if not self._started:
            self.start()
        self.wait_ready(timeout=None)
        if self.error is not None:
            raise self.error
        if self._session is None:
            raise RuntimeError("Streaming STT session did not start.")
        return self._session

    def _open_session(self) -> None:
        try:
            self._session = self.stt.start_streaming(self.sample_rate)
            self.ready_at = time.monotonic()
        except Exception as exc:
            self.error = exc
            self.completed_at = time.monotonic()
        finally:
            self._ready_event.set()

    def _write_loop(self) -> None:
        try:
            self.wait_ready(timeout=None)
            if self.error is not None or self._session is None:
                return
            while True:
                item = self._write_queue.get()
                if item is self._STOP:
                    return
                if self._abort_requested:
                    return
                try:
                    with self._io_lock:
                        if self._abort_requested:
                            return
                        self._session.write(item)
                    self.sent_chunks += 1
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    self.error = exc
                    return
        finally:
            self._writer_done.set()


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
        self.tool_runner = create_tool_runner(config, self.chat)
        self.music_controller = (
            self.tool_runner.music_controller if self.tool_runner is not None else None
        )
        self.session = ConversationSession(config.llm, config.conversation)
        self.audio_dump = AudioDumpManager(config.debug)
        configure_audio_dump(self.audio_dump)
        configure_log_files(
            debug_log_path=self.audio_dump.debug_log_path(),
            text_record_dir=self.audio_dump.text_record_dir(),
        )
        self.debug = DebugRecorder(config.debug, audio_dump=self.audio_dump)
        self.reminders = ReminderScheduler(self._handle_reminder_due)
        self._pending_barge_utterance: Utterance | None = None
        self._pending_barge_transcript: str | None = None
        self._pending_barge_stt_ms = 0
        self._pending_barge_stt_extra_timings: dict[str, int] = {}
        self._standby_stt_handle: _StreamingSttHandle | None = None
        self._recent_spoken_responses: list[_SpokenResponse] = []
        self._turn_lock = threading.RLock()
        if audio_enabled:
            self._warm_up_runtime_modules()
        self._print_barge_in_config()
        self._start_weather_cache_warmup()

    def _create_cron_scheduler(self) -> CronScheduler | None:
        if not self.config.cron.enabled:
            return None

        def run_job(job) -> None:
            self.run_text_turn(job.text)

        return CronScheduler(self.config.cron, run_job)

    def close(self) -> None:
        if self._standby_stt_handle is not None:
            self._standby_stt_handle.abort()
            self._standby_stt_handle = None
        self.reminders.stop()
        self.audio_dump.stop_system_input_dump()
        configure_audio_dump(None)
        configure_log_files()

    def _start_system_input_dump(self) -> None:
        if self.config.input.mode != "audio":
            return
        self.audio_dump.start_system_input_dump(self.config.audio)

    def _warm_up_runtime_modules(self) -> None:
        components = (
            ("wake", self.wake),
            ("vad", self.vad),
            ("stt", self.stt),
            ("llm", self.chat),
            ("tts", self.tts),
        )
        with ThreadPoolExecutor(
            max_workers=len(components),
            thread_name_prefix="voiceui-warmup",
        ) as executor:
            futures = [
                executor.submit(self._warm_up_component, module, component)
                for module, component in components
            ]
            for future in futures:
                future.result()

    def _warm_up_component(self, module: str, component: object) -> None:
        warm_up = getattr(component, "warm_up", None)
        if not callable(warm_up):
            return
        started = time.monotonic()
        try:
            warmed = bool(warm_up())
        except Exception as exc:
            log_event(module, "warm_up_error", log_id=f"{module}.warm_up_error", error=exc)
            return
        if warmed:
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(module, "warmed_up", log_id=f"{module}.warmed_up", latency_ms=latency_ms)

    def _start_weather_cache_warmup(self) -> None:
        if not (
            self.config.tools.enabled
            and self.config.tools.allow_weather
            and self.config.tools.default_weather_location
        ):
            return

        def warm_up() -> None:
            started = time.monotonic()
            try:
                warm_weather_cache(self.config.tools.default_weather_location)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                log_event(
                    "weather",
                    "warmup_error",
                    log_id="weather.warmup_error",
                    error=exc,
                )
                return
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(
                "weather",
                "warmed_up",
                log_id="weather.warmed_up",
                latency_ms=latency_ms,
            )

        thread = threading.Thread(target=warm_up, name="voiceui-weather-warmup", daemon=True)
        thread.start()

    def run_text_turn(self, text: str) -> AssistantReply:
        with self._turn_lock:
            transcript = text.strip()
            if not transcript:
                return AssistantReply(text=_EMPTY_INPUT_RESPONSE, routed_to="system")

            turn_index = self.audio_dump.begin_turn()
            try:
                reply, _timings = self._complete_transcript(transcript)
                return reply
            finally:
                self.audio_dump.end_turn(turn_index)

    def _complete_transcript(self, transcript: str) -> tuple[AssistantReply, dict[str, int]]:
        self.session.add_user(transcript)
        timings: dict[str, int] = {}

        local_response = self._try_handle_local_conversation_command(transcript)
        if local_response is not None:
            return self._finish_local_system_response(
                local_response,
                timings,
                mode="local_conversation",
            )

        try:
            local_response = self._try_handle_local_miot_schedule_command(transcript)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return self._finish_processing_error(exc, timings, mode="local_miot_schedule")
        if local_response is not None:
            timings["llm"] = 0
            log_event(
                "llm",
                "completed",
                log_id="llm.completed",
                latency_ms=0,
                mode="local_miot_schedule",
            )
            return self._finish_generated_response(
                local_response,
                timings,
                routed_to="miot_schedule",
            )

        try:
            local_response = self._try_handle_local_reminder_command(transcript)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return self._finish_processing_error(exc, timings, mode="local_reminder")
        if local_response is not None:
            timings["llm"] = 0
            log_event(
                "llm",
                "completed",
                log_id="llm.completed",
                latency_ms=0,
                mode="local_reminder",
            )
            return self._finish_generated_response(local_response, timings, routed_to="reminder")

        try:
            local_response = self._try_handle_local_music_command(transcript)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return self._finish_processing_error(exc, timings, mode="local_music")
        if local_response is not None:
            timings["llm"] = 0
            return self._finish_generated_response(local_response, timings)

        try:
            local_response = self._try_handle_local_volume_command(transcript)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return self._finish_processing_error(exc, timings, mode="local_volume")
        if local_response is not None:
            timings["llm"] = 0
            log_event(
                "llm",
                "completed",
                log_id="llm.completed",
                latency_ms=0,
                mode="local_volume",
            )
            return self._finish_generated_response(local_response, timings, routed_to="volume")

        try:
            local_response = self._try_handle_local_info_command(transcript)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return self._finish_processing_error(exc, timings, mode="local_tools")
        if local_response is not None:
            timings["llm"] = 0
            log_event(
                "llm",
                "completed",
                log_id="llm.completed",
                latency_ms=0,
                mode="local_tools",
            )
            return self._finish_generated_response(local_response, timings)

        if self.tool_runner is not None and self.tool_runner.enabled:
            if self.config.llm.stream and callable(
                getattr(self.tool_runner, "stream_complete", None)
            ):
                try:
                    response, barge_utterance = self._stream_and_speak_tool_response(
                        transcript,
                        timings,
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    return self._finish_processing_error(exc, timings, mode="tools_stream")
                if barge_utterance is not None:
                    self._pending_barge_utterance = barge_utterance
                    timings["barge_in"] = barge_utterance.duration_ms
                record_text_event("llm", "completed", response, mode="tools_stream")
                log_event(
                    "tts",
                    "completed",
                    log_id="tts.completed",
                    latency_ms=timings["tts"],
                    ok=True,
                    text=response,
                )
                return AssistantReply(text=response), timings

            llm_started = time.monotonic()
            try:
                response = self._complete_tools_with_progress_prompt(transcript)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                timings["llm"] = int((time.monotonic() - llm_started) * 1000)
                return self._finish_processing_error(exc, timings, mode="tools")
            timings["llm"] = int((time.monotonic() - llm_started) * 1000)
            log_event(
                "llm",
                "completed",
                log_id="llm.completed",
                latency_ms=timings["llm"],
                mode="tools",
            )
            if not response:
                response = "I could not produce a response."
            return self._finish_generated_response(response, timings)
        elif self.config.llm.stream:
            try:
                response, barge_utterance = self._stream_and_speak_response(timings)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return self._finish_processing_error(exc, timings, mode="llm_stream")
        else:
            llm_started = time.monotonic()
            try:
                response = self.chat.complete(self.session.messages)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                timings["llm"] = int((time.monotonic() - llm_started) * 1000)
                return self._finish_processing_error(exc, timings, mode="llm")
            timings["llm"] = int((time.monotonic() - llm_started) * 1000)
            log_event("llm", "completed", log_id="llm.completed", latency_ms=timings["llm"])
            if not response:
                response = "I could not produce a response."
            return self._finish_generated_response(response, timings)

        if barge_utterance is not None:
            self._pending_barge_utterance = barge_utterance
            timings["barge_in"] = barge_utterance.duration_ms
        record_text_event("llm", "completed", response, mode="stream")
        log_event(
            "tts",
            "completed",
            log_id="tts.completed",
            latency_ms=timings["tts"],
            ok=True,
            text=response,
        )
        return AssistantReply(text=response), timings

    def _complete_tools_with_progress_prompt(self, transcript: str) -> str:
        assert self.tool_runner is not None
        messages = list(self.session.messages)
        return self._run_with_progress_prompt(
            operation=lambda: self.tool_runner.complete(messages),
            prompt=_tool_progress_prompt_for_text(transcript),
            label=_tool_progress_label_for_text(transcript),
        )

    def _run_with_progress_prompt(self, operation, prompt: str, label: str) -> str:
        completed = threading.Event()
        result: dict[str, object] = {}

        def worker() -> None:
            try:
                result["value"] = operation()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                result["error"] = exc
            finally:
                completed.set()

        worker_thread = threading.Thread(
            target=worker,
            name=f"voiceui-tool-{label}",
            daemon=True,
        )
        worker_thread.start()

        prompt_stop_event = threading.Event()
        prompt_thread: threading.Thread | None = None
        try:
            if not completed.wait(_TOOL_PROGRESS_PROMPT_DELAY_SECONDS):
                log_event(
                    "tools",
                    "progress_prompt",
                    log_id="tools.progress_prompt",
                    label=label,
                    prompt=prompt,
                )
                prompt_thread = threading.Thread(
                    target=self._speak_progress_prompt,
                    args=(prompt, prompt_stop_event, label),
                    name=f"voiceui-tool-progress-{label}",
                    daemon=True,
                )
                prompt_thread.start()
                completed.wait()
        finally:
            prompt_stop_event.set()
            if prompt_thread is not None:
                prompt_thread.join(timeout=1.5)
            worker_thread.join(timeout=0.1)

        error = result.get("error")
        if isinstance(error, Exception):
            raise error
        return str(result.get("value") or "")

    def _stream_and_speak_tool_response(
        self,
        transcript: str,
        timings: dict[str, int],
    ) -> tuple[str, Utterance | None]:
        del transcript
        assert self.tool_runner is not None
        messages = list(self.session.messages)
        stream_complete = self.tool_runner.stream_complete
        return self._stream_and_speak_response(
            timings,
            text_stream_factory=lambda: stream_complete(messages),
            log_mode="tools_stream",
        )

    def _try_handle_local_conversation_command(self, transcript: str) -> str | None:
        if _looks_like_end_conversation_command(transcript):
            return "好的。"
        if _looks_like_assistant_name_greeting(transcript):
            return "我在。"
        return None

    def _is_voice_termination_command(self, transcript: str) -> bool:
        if not bool(getattr(self.config.conversation, "voice_termination_enabled", True)):
            return False
        phrases = getattr(
            self.config.conversation,
            "voice_termination_phrases",
            DEFAULT_VOICE_TERMINATION_PHRASES,
        )
        return _matches_termination_command(transcript, phrases)

    def _clear_pending_barge_in(self) -> None:
        self._pending_barge_utterance = None
        self._pending_barge_transcript = None
        self._pending_barge_stt_ms = 0
        self._pending_barge_stt_extra_timings = {}

    def _finish_voice_termination_response(
        self,
        transcript: str,
        timings: dict[str, int],
        *,
        source: str,
    ) -> tuple[AssistantReply, dict[str, int]]:
        self._clear_pending_barge_in()
        self.session.reset()
        timings["llm"] = 0
        reply_text = str(
            getattr(self.config.conversation, "voice_termination_reply", "") or ""
        ).strip()
        log_event(
            "assistant",
            "voice_terminated",
            log_id="assistant.voice_terminated",
            source=source,
            text_len=len(transcript),
            reply=bool(reply_text),
        )
        if not reply_text:
            timings["tts"] = 0
            return AssistantReply(text="", routed_to="voice_termination"), timings

        tts_started = time.monotonic()
        try:
            self._speak_plain(reply_text)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            timings["tts"] = int((time.monotonic() - tts_started) * 1000)
            log_event(
                "tts",
                "completed",
                log_id="tts.completed",
                latency_ms=timings["tts"],
                ok=False,
                error=str(exc),
                text=reply_text,
            )
            log_event(
                "error",
                "runtime",
                log_id="error.runtime",
                stage="voice_termination_tts",
                error=str(exc),
            )
            return AssistantReply(text=reply_text, routed_to="voice_termination"), timings

        timings["tts"] = int((time.monotonic() - tts_started) * 1000)
        self._remember_spoken_response(reply_text)
        log_event(
            "tts",
            "completed",
            log_id="tts.completed",
            latency_ms=timings["tts"],
            ok=True,
            text=reply_text,
        )
        return AssistantReply(text=reply_text, routed_to="voice_termination"), timings

    def _finish_local_system_response(
        self,
        response: str,
        timings: dict[str, int],
        *,
        mode: str,
    ) -> tuple[AssistantReply, dict[str, int]]:
        timings["llm"] = 0
        log_event(
            "llm",
            "completed",
            log_id="llm.completed",
            latency_ms=0,
            mode=mode,
        )
        return self._finish_generated_response(response, timings, routed_to="system")

    def _gate_voice_transcript(self, transcript: str, source: str) -> tuple[str, str, str]:
        echo_match = self._match_self_echo(transcript, source)
        if echo_match is not None:
            log_event(
                "assistant",
                "input_gate",
                log_id="assistant.input_gate",
                source=source,
                decision="reject",
                reason="self_echo",
                text_len=len(transcript),
                matched_chars=echo_match.matched_chars,
                age_ms=echo_match.age_ms,
            )
            return "reject", "self_echo", ""
        if not _voice_input_gate_enabled(self.config.conversation, source):
            return "accept", "disabled", ""
        miot_reason = self._miot_voice_gate_accept_reason(transcript, source)
        if miot_reason:
            log_event(
                "assistant",
                "input_gate",
                log_id="assistant.input_gate",
                source=source,
                decision="accept",
                reason=miot_reason,
                text_len=len(transcript),
            )
            return "accept", miot_reason, ""
        decision, reason = _classify_voice_input(transcript, source)
        log_event(
            "assistant",
            "input_gate",
            log_id="assistant.input_gate",
            source=source,
            decision=decision,
            reason=reason,
            text_len=len(transcript),
        )
        if decision == "clarify":
            return decision, reason, _clarification_response_for_text(transcript)
        return decision, reason, ""

    def _miot_voice_gate_accept_reason(self, transcript: str, source: str) -> str:
        if source not in {"barge_in", "follow_up"}:
            return ""
        runner = self.tool_runner
        if runner is None or not getattr(runner, "enabled", False):
            return ""
        can_handle_followup = getattr(runner, "can_handle_miot_followup_text", None)
        if callable(can_handle_followup) and can_handle_followup(transcript):
            return "miot_followup_context"
        return ""

    def _match_self_echo(self, transcript: str, source: str) -> _SelfEchoMatch | None:
        if source not in {"barge_in", "follow_up"}:
            return None
        if not bool(getattr(self.config.conversation, "self_echo_filter_enabled", True)):
            return None
        normalized = _normalize_self_echo_text(transcript)
        if len(normalized) < 2:
            return None
        now = time.monotonic()
        window_seconds = max(
            0.0,
            float(getattr(self.config.conversation, "self_echo_window_seconds", 8.0)),
        )
        cutoff = now - window_seconds
        self._recent_spoken_responses = [
            spoken for spoken in self._recent_spoken_responses if spoken.spoken_at >= cutoff
        ]
        best_match: _SelfEchoMatch | None = None
        for spoken in self._recent_spoken_responses:
            matched_chars = _self_echo_matched_chars(normalized, spoken.normalized)
            if not _is_self_echo_match(normalized, spoken.normalized, matched_chars):
                continue
            age_ms = int((now - spoken.spoken_at) * 1000)
            candidate = _SelfEchoMatch(matched_chars=matched_chars, age_ms=age_ms)
            if best_match is None or candidate.matched_chars > best_match.matched_chars:
                best_match = candidate
        return best_match

    def _finish_input_gate_clarification(
        self,
        response: str,
        timings: dict[str, int],
    ) -> tuple[AssistantReply, dict[str, int]]:
        timings["llm"] = 0
        barge_utterance = self._speak_response_safely(response, timings)
        if barge_utterance is not None:
            self._pending_barge_utterance = barge_utterance
            timings["barge_in"] = barge_utterance.duration_ms
        return AssistantReply(text=response, routed_to="input_gate"), timings

    def _speak_progress_prompt(
        self,
        prompt: str,
        stop_event: threading.Event,
        label: str,
    ) -> None:
        try:
            self._speak_plain(prompt, stop_event=stop_event)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_event(
                "tools",
                "progress_prompt",
                log_id="tools.progress_prompt",
                label=label,
                ok=False,
                error=exc,
            )

    def _finish_generated_response(
        self,
        response: str,
        timings: dict[str, int],
        *,
        routed_to: str = "llm",
    ) -> tuple[AssistantReply, dict[str, int]]:
        spoken_response = self._prepare_spoken_response(response, routed_to=routed_to)
        self.session.add_assistant(spoken_response)
        record_text_event("llm", "completed", spoken_response, routed_to=routed_to)
        barge_utterance = self._speak_response_safely(spoken_response, timings)
        if barge_utterance is not None:
            self._pending_barge_utterance = barge_utterance
            timings["barge_in"] = barge_utterance.duration_ms
        return AssistantReply(text=spoken_response, routed_to=routed_to), timings

    def _prepare_spoken_response(self, response: str, *, routed_to: str) -> str:
        max_chars = self.config.conversation.max_spoken_reply_chars
        if (
            self.config.input.mode != "audio"
            or max_chars <= 0
            or routed_to == "error"
            or len(response) <= max_chars
        ):
            return response
        compacted = _compact_spoken_response(response, max_chars)
        if compacted == response:
            return response
        log_event(
            "assistant",
            "reply_compacted",
            log_id="assistant.reply_compacted",
            original_chars=len(response),
            compacted_chars=len(compacted),
            routed_to=routed_to,
        )
        return compacted

    def _finish_processing_error(
        self,
        exc: Exception,
        timings: dict[str, int],
        *,
        mode: str,
    ) -> tuple[AssistantReply, dict[str, int]]:
        timings.setdefault("llm", 0)
        response = _format_processing_error_response(exc)
        log_event(
            "llm",
            "completed",
            log_id="llm.completed",
            latency_ms=timings["llm"],
            mode=mode,
            ok=False,
            error=str(exc),
        )
        log_event(
            "error",
            "runtime",
            log_id="error.runtime",
            stage=mode,
            error=str(exc),
        )
        return self._finish_generated_response(response, timings, routed_to="error")

    def _speak_response_safely(
        self,
        response: str,
        timings: dict[str, int],
    ) -> Utterance | None:
        tts_started = time.monotonic()
        barge_utterance = None
        try:
            if self._should_listen_for_barge_in():
                barge_utterance = self._speak_with_barge_in(response)
            else:
                self._speak_plain(response)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            timings["tts"] = int((time.monotonic() - tts_started) * 1000)
            log_event(
                "tts",
                "completed",
                log_id="tts.completed",
                latency_ms=timings["tts"],
                ok=False,
                error=str(exc),
                text=response,
            )
            log_event(
                "error",
                "runtime",
                log_id="error.runtime",
                stage="tts",
                error=str(exc),
            )
            return None
        timings["tts"] = int((time.monotonic() - tts_started) * 1000)
        self._remember_spoken_response(response)
        log_event(
            "tts",
            "completed",
            log_id="tts.completed",
            latency_ms=timings["tts"],
            ok=True,
            text=response,
        )
        return barge_utterance

    def _remember_spoken_response(self, response: str) -> None:
        normalized = _normalize_self_echo_text(response)
        if not normalized:
            return
        now = time.monotonic()
        window_seconds = max(
            0.0,
            float(getattr(self.config.conversation, "self_echo_window_seconds", 8.0)),
        )
        cutoff = now - window_seconds
        self._recent_spoken_responses = [
            spoken for spoken in self._recent_spoken_responses if spoken.spoken_at >= cutoff
        ]
        self._recent_spoken_responses.append(
            _SpokenResponse(text=response, normalized=normalized, spoken_at=now)
        )
        self._recent_spoken_responses = self._recent_spoken_responses[-8:]

    def _stream_and_speak_response(
        self,
        timings: dict[str, int],
        *,
        text_stream_factory=None,
        log_mode: str = "stream",
    ) -> tuple[str, Utterance | None]:
        messages = list(self.session.messages)
        llm_stop_event = threading.Event()
        llm_stream_stats: dict[str, int] = {}
        text_chunks = self._start_tracked_llm_stream(
            messages,
            timings,
            stop_event=llm_stop_event,
            stream_stats=llm_stream_stats,
            text_stream_factory=text_stream_factory,
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
                response = self._speak_stream_plain(text_chunks)
        finally:
            llm_stop_event.set()
        timings["tts"] = int((time.monotonic() - tts_started) * 1000)
        self._print_streaming_llm_stats(timings, llm_stream_stats, mode=log_mode)

        if not response:
            response = "I could not produce a response."
            fallback_tts_started = time.monotonic()
            if self._should_listen_for_barge_in():
                barge_utterance = self._speak_with_barge_in(response)
            else:
                self._speak_plain(response)
            timings["tts"] += int((time.monotonic() - fallback_tts_started) * 1000)
        self._remember_spoken_response(response)
        self.session.add_assistant(response)
        return response, barge_utterance

    def _speak_plain(self, text: str, stop_event: threading.Event | None = None) -> None:
        self._duck_music("tts")
        try:
            self.tts.speak(text, stop_event=stop_event)
        finally:
            self._unduck_music("tts")

    def _speak_stream_plain(
        self,
        text_chunks: Iterator[str],
        stop_event: threading.Event | None = None,
    ) -> str:
        self._duck_music("tts")
        try:
            return self.tts.speak_text_stream(text_chunks, stop_event=stop_event)
        finally:
            self._unduck_music("tts")

    def _duck_music(self, reason: str) -> None:
        duck = getattr(self.music_controller, "duck", None)
        if callable(duck):
            duck(reason)

    def _unduck_music(self, reason: str) -> None:
        unduck = getattr(self.music_controller, "unduck", None)
        if callable(unduck):
            unduck(reason)

    def _handle_reminder_due(self, reminder: Reminder) -> None:
        with self._turn_lock:
            text = self._due_reminder_text(reminder)
            tts_started = time.monotonic()
            try:
                self._speak_plain(text)
            except Exception as exc:
                latency_ms = int((time.monotonic() - tts_started) * 1000)
                log_event(
                    "tts",
                    "completed",
                    log_id="tts.completed",
                    latency_ms=latency_ms,
                    ok=False,
                    error=str(exc),
                    text=text,
                )
                log_event(
                    "error",
                    "runtime",
                    log_id="error.runtime",
                    stage="reminder_tts",
                    error=str(exc),
                )
                raise
            latency_ms = int((time.monotonic() - tts_started) * 1000)
            log_event(
                "tts",
                "completed",
                log_id="tts.completed",
                latency_ms=latency_ms,
                ok=True,
                text=text,
            )

    def _due_reminder_text(self, reminder: Reminder) -> str:
        if reminder.kind != "miot":
            return reminder.text
        runner = self.tool_runner
        run_miot = getattr(runner, "run_miot_control", None)
        if runner is None or not getattr(runner, "enabled", False) or not callable(run_miot):
            return (
                "\u5230\u70b9\u4e86\uff0c\u4f46\u6211\u73b0\u5728\u4e0d\u80fd"
                "\u6267\u884c\u7c73\u5bb6\u8bbe\u5907\u63a7\u5236\u3002"
            )

        payload = reminder.payload if isinstance(reminder.payload, dict) else {}
        arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        if not arguments:
            arguments = {"request": reminder.text}
        result = run_miot(dict(arguments), remember=True)
        formatter = getattr(runner, "format_tool_response", None)
        if callable(formatter):
            response = str(formatter(result) or "").strip()
        else:
            response = str(result.get("direct_response") or result.get("message") or "").strip()
        return response or (
            "\u5230\u70b9\u4e86\uff0c\u4f46\u7c73\u5bb6\u8bbe\u5907"
            "\u63a7\u5236\u6ca1\u6709\u8fd4\u56de\u7ed3\u679c\u3002"
        )

    def _try_handle_local_miot_schedule_command(self, transcript: str) -> str | None:
        runner = self.tool_runner
        if runner is None or not getattr(runner, "enabled", False):
            return None
        parsed = parse_scheduled_command(
            transcript,
            max_delay_seconds=self.config.conversation.max_reminder_delay_seconds,
        )
        if parsed is None:
            return None

        can_handle = getattr(runner, "can_handle_miot_control_text", None)
        if not callable(can_handle) or not can_handle(parsed.command_text):
            return None
        run_miot = getattr(runner, "run_miot_control", None)
        if not callable(run_miot):
            return None

        preview = run_miot({"request": parsed.command_text, "dry_run": True}, remember=False)
        status = str(preview.get("status") or "")
        if preview.get("ok") is False or status not in {"resolved", "group_resolved"}:
            formatter = getattr(runner, "format_tool_response", None)
            if callable(formatter):
                response = str(formatter(preview) or "").strip()
            else:
                response = str(preview.get("direct_response") or preview.get("message") or "")
            return response or (
                "\u8fd9\u4e2a\u7c73\u5bb6\u5b9a\u65f6\u6307\u4ee4"
                "\u8fd8\u4e0d\u80fd\u786e\u5b9a\u6267\u884c\u76ee\u6807\u3002"
            )

        arguments = _scheduled_miot_arguments_from_preview(parsed.command_text, preview)
        reminder = self.reminders.schedule_at(
            parsed.due_at,
            parsed.command_text,
            kind="miot",
            payload={
                "tool": "xiaomi_miot_control_device",
                "arguments": arguments,
            },
        )
        log_event(
            "miot",
            "scheduled",
            log_id="miot.scheduled",
            id=reminder.id,
            due_at=reminder.due_at.isoformat(timespec="seconds"),
            command_len=len(parsed.command_text),
        )
        return f"\u597d\u7684\uff0c{parsed.label}\u5e2e\u4f60\u6267\u884c\u3002"

    def _try_handle_local_reminder_command(self, transcript: str) -> str | None:
        if not looks_like_reminder_text(transcript):
            return None
        if not self.config.conversation.reminders_enabled:
            return "我现在不能设置闹钟或提醒。"

        normalized = transcript.lower().replace(" ", "")
        if looks_like_reminder_cancel_text(transcript):
            count = self.reminders.cancel_all()
            if count:
                return "已取消待提醒的闹钟。"
            return "当前没有待提醒的闹钟。"

        if any(term in normalized for term in ("什么反应", "怎么响", "响了以后")):
            return "到点后我会直接播报提醒。"

        if any(term in normalized for term in ("为什么", "没响", "没提醒", "没有提醒", "还没有")):
            return format_pending_reminders(self.reminders.pending())

        parsed = parse_reminder_request(
            transcript,
            max_delay_seconds=self.config.conversation.max_reminder_delay_seconds,
        )
        if parsed is not None:
            self.reminders.schedule_at(parsed.due_at, parsed.text, kind=parsed.kind)
            return format_reminder_confirmation(parsed)

        if looks_like_reminder_status_text(transcript):
            return format_pending_reminders(self.reminders.pending())

        if looks_like_reminder_create_text(transcript):
            return "你想让我什么时候提醒？"

        return None

    def _try_handle_local_music_command(self, transcript: str) -> str | None:
        music = self.music_controller
        if music is None or not self._music_is_active():
            return None

        normalized = transcript.lower().replace(" ", "")
        stop_terms = (
            "停止播放",
            "停止音乐",
            "暂停播放",
            "暂停音乐",
            "别放了",
            "不要放了",
            "关掉音乐",
            "关闭音乐",
            "把音乐关了",
            "stopmusic",
            "pausemusic",
            "stopthemusic",
        )
        if not any(term in normalized for term in stop_terms):
            return None

        result = music.stop(wait=True)
        status = result.get("status") if isinstance(result, dict) else ""
        if status == "idle":
            return "当前没有音乐在播放。"
        return "已停止播放。"

    def _music_is_active(self) -> bool:
        is_active = getattr(self.music_controller, "is_active", None)
        if callable(is_active):
            return bool(is_active())
        return False

    def _try_handle_local_volume_command(self, transcript: str) -> str | None:
        if not (self.config.tools.enabled and self.config.tools.allow_volume):
            return None
        normalized = transcript.lower().replace(" ", "")
        if not _looks_like_volume_query(normalized):
            return None
        started = time.monotonic()
        try:
            request = _parse_volume_request(normalized)
            if request["action"] == "get":
                result = get_system_output_volume(device=self.config.tts.playback_device)
            else:
                result = set_system_output_volume(
                    device=self.config.tts.playback_device,
                    volume_percent=request.get("volume_percent"),
                    relative_percent=request.get("relative_percent"),
                    muted=request.get("muted"),
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(
                "tool",
                "executed",
                log_id="tool.executed",
                name="local_system_volume",
                latency_ms=latency_ms,
                ok=False,
                mode="local",
                error=str(exc),
            )
            raise
        latency_ms = int((time.monotonic() - started) * 1000)
        log_event(
            "tool",
            "executed",
            log_id="tool.executed",
            name="local_system_volume",
            latency_ms=latency_ms,
            ok=True,
            mode="local",
        )
        if request["action"] == "get":
            return f"现在音量是{int(round(float(result.get('after_percent', 0))))}% 。"
        if request.get("muted") is True:
            return "已静音。"
        if request.get("muted") is False:
            return "已取消静音。"
        return f"好的，音量已调到{int(round(float(result.get('after_percent', 0))))}% 。"

    def _try_handle_local_info_command(self, transcript: str) -> str | None:
        if not self.config.tools.enabled:
            return None
        normalized = transcript.lower().replace(" ", "")
        if self._should_defer_local_info_to_miot(transcript):
            return None
        if self.config.tools.allow_weather and _looks_like_weather_query(normalized):
            location = _extract_weather_location(
                transcript,
                default_location=self.config.tools.default_weather_location,
            )
            if not location:
                return None
            arguments = {"location": location}
            target_day = _extract_weather_target_day(transcript)
            if target_day != "today":
                arguments["target_day"] = target_day
            started = time.monotonic()
            try:
                result = get_current_weather(arguments)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                latency_ms = int((time.monotonic() - started) * 1000)
                log_event(
                    "tool",
                    "executed",
                    log_id="tool.executed",
                    name="get_current_weather",
                    latency_ms=latency_ms,
                    ok=False,
                    mode="local",
                    error=str(exc),
                )
                fallback_location = self.config.tools.default_weather_location.strip()
                if fallback_location and fallback_location != location:
                    fallback_arguments = {"location": fallback_location}
                    if target_day != "today":
                        fallback_arguments["target_day"] = target_day
                    fallback_started = time.monotonic()
                    try:
                        result = get_current_weather(fallback_arguments)
                    except Exception as fallback_exc:  # pylint: disable=broad-exception-caught
                        fallback_latency_ms = int((time.monotonic() - fallback_started) * 1000)
                        log_event(
                            "tool",
                            "executed",
                            log_id="tool.executed",
                            name="get_current_weather",
                            latency_ms=fallback_latency_ms,
                            ok=False,
                            mode="local_fallback",
                            error=str(fallback_exc),
                        )
                        return _format_weather_error_response(fallback_exc)
                    fallback_latency_ms = int((time.monotonic() - fallback_started) * 1000)
                    log_event(
                        "tool",
                        "executed",
                        log_id="tool.executed",
                        name="get_current_weather",
                        latency_ms=fallback_latency_ms,
                        ok=True,
                        mode="local_fallback",
                    )
                    return str(result.get("direct_response") or format_weather_response(result))
                return _format_weather_error_response(exc)
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(
                "tool",
                "executed",
                log_id="tool.executed",
                name="get_current_weather",
                latency_ms=latency_ms,
                ok=True,
                mode="local",
            )
            return str(result.get("direct_response") or format_weather_response(result))

        if self.config.tools.allow_time and _looks_like_time_query(normalized):
            started = time.monotonic()
            result = get_current_time({"timezone": "Asia/Shanghai"})
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(
                "tool",
                "executed",
                log_id="tool.executed",
                name="get_current_time",
                latency_ms=latency_ms,
                ok=True,
                mode="local",
            )
            return str(result.get("direct_response") or format_current_time_response(result))
        return None

    def _should_defer_local_info_to_miot(self, transcript: str) -> bool:
        runner = self.tool_runner
        if runner is None or not getattr(runner, "enabled", False):
            return False
        can_handle = getattr(runner, "can_handle_miot_text", None)
        return callable(can_handle) and bool(can_handle(transcript))

    def _start_tracked_llm_stream(
        self,
        messages: list,
        timings: dict[str, int],
        stop_event: threading.Event | None = None,
        stream_stats: dict[str, int] | None = None,
        text_stream_factory=None,
    ) -> Iterator[str]:
        items: BoundedBackpressureQueue[object] = BoundedBackpressureQueue(
            StreamFramePolicy.text_audio_default().buffer_frames
        )
        done = object()

        def producer() -> None:
            llm_started = time.monotonic()
            first_token_ms: int | None = None
            chunks = 0
            try:
                text_stream = (
                    text_stream_factory()
                    if text_stream_factory is not None
                    else self.chat.stream_complete(messages)
                )
                for chunk in text_stream:
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
            except TimeoutError:
                if not thread.is_alive():
                    break
                continue
            if item is done:
                if stream_stats is not None:
                    stream_stats.update(items.stats())
                break
            if isinstance(item, Exception):
                if stream_stats is not None:
                    stream_stats.update(items.stats())
                raise item
            if (
                stream_stats is not None
                and not stream_stats.get("first_token_logged")
                and "llm_first_token" in timings
            ):
                stream_stats["first_token_logged"] = 1
                log_event(
                    "llm",
                    "first_token",
                    log_id="llm.first_token",
                    latency_ms=timings["llm_first_token"],
                )
            yield item  # type: ignore[misc]

    def _print_streaming_llm_stats(
        self,
        timings: dict[str, int],
        stream_stats: dict[str, int],
        *,
        mode: str = "stream",
    ) -> None:
        if "llm" not in timings:
            return
        log_event(
            "llm",
            "stream_completed",
            log_id="llm.stream_completed",
            latency_ms=timings["llm"],
            first_token_ms=timings.get("llm_first_token", timings["llm"]),
            stream_chunks=stream_stats.get("chunks", 0),
            blocked_puts=stream_stats.get("blocked_puts", 0),
            blocked_put_ms=stream_stats.get("blocked_put_ms", 0),
            mode=mode,
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
        log_event(
            "barge_in",
            "config",
            log_id="barge_in.config",
            enabled=enabled,
            reason=reason,
            input_mode=self.config.input.mode,
            conversation_enabled=self.config.conversation.barge_in_enabled,
            vad_engine=self.config.vad.engine,
            threshold=self.config.vad.threshold,
            command_channel=self.config.audio.command_stream_channel,
        )

    def _print_barge_in_monitor_started(self, mode: str) -> None:
        log_event(
            "barge_in",
            "monitor_started",
            log_id="barge_in.monitor_started",
            mode=mode,
            start_timeout_seconds=0.0,
            vad_engine=self.config.vad.engine,
            threshold=self.config.vad.threshold,
            command_channel=self.config.audio.command_stream_channel,
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
            log_event(
                "barge_in",
                "monitor_saved",
                log_id="barge_in.monitor_saved",
                path=debug_dir,
                duration_ms=monitor_audio.duration_ms(),
                result=result,
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
            stream_handle = _StreamingSttHandle(
                self.stt,
                monitor_audio.sample_rate,
                policy=_streaming_frame_policy(
                    self.config.vad,
                    monitor_audio.block_ms,
                ),
            )
            stream_handle.start()
            ready = stream_handle.wait_ready(timeout=0)
            start_ms = int((time.monotonic() - stt_start_reference) * 1000)
            log_event(
                "stt",
                "streaming_started",
                log_id="stt.streaming_started",
                source="barge_in",
                mode=mode,
                elapsed_ms=start_ms,
                ready=ready,
                ready_ms=stream_handle.ready_latency_ms() or 0,
            )

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
        if not transcript.strip():
            fallback_started = time.monotonic()
            transcript = self._fallback_transcribe_empty_stream(
                utterance,
                source="barge_in",
                stream_total_latency_ms=stream_handle.total_latency_ms(),
                sent_chunks=stream_handle.sent_chunks,
                dropped_chunks=stream_handle.dropped_chunks,
            )
            stt_ms += int((time.monotonic() - fallback_started) * 1000)
        stt_total_ms = stream_handle.total_latency_ms()
        ready_ms = stream_handle.ready_latency_ms()
        params = {
            "latency_ms": stt_ms,
            "mode": "streaming",
            "source": "barge_in",
            "total_latency_ms": stt_total_ms,
            "sent_chunks": stream_handle.sent_chunks,
            "dropped_chunks": stream_handle.dropped_chunks,
            "text": transcript,
        }
        if ready_ms is not None:
            params["ready_ms"] = ready_ms
        log_event(
            "stt",
            "completed",
            log_id="stt.completed",
            **params,
        )
        return utterance, transcript, stt_ms, {"stt_total": stt_total_ms}

    def _speak_with_barge_in(self, text: str) -> Utterance | None:
        playback_stop_event = threading.Event()
        monitor_stop_event = threading.Event()
        state: dict[str, object] = {}
        monitor_audio = RecordingAudioInput(self.command_audio)

        def on_speech_start() -> None:
            if not playback_stop_event.is_set():
                log_event("barge_in", "speech_start", log_id="barge_in.speech_start")
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
            self._speak_plain(text, stop_event=playback_stop_event)
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
                    log_event(
                        "barge_in",
                        "timeout",
                        log_id="barge_in.timeout",
                        reason="waiting_for_utterance",
                    )
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
            log_event("barge_in", "error", log_id="barge_in.error", error=error)
            return None

        utterance = state.get("utterance")
        if isinstance(utterance, Utterance):
            transcript = state.get("transcript")
            if isinstance(transcript, str):
                if not transcript.strip():
                    self._clear_pending_barge_in()
                    log_event(
                        "barge_in",
                        "no_speech",
                        log_id="barge_in.no_speech",
                        reason="empty_transcript",
                    )
                    return None
                self._pending_barge_transcript = transcript
                self._pending_barge_stt_ms = int(state.get("stt_ms") or 0)
                extra_timings = state.get("stt_extra_timings")
                self._pending_barge_stt_extra_timings = (
                    dict(extra_timings) if isinstance(extra_timings, dict) else {}
                )
            log_event(
                "barge_in",
                "captured",
                log_id="barge_in.captured",
                duration_ms=utterance.duration_ms,
            )
            return utterance
        log_event("barge_in", "no_speech", log_id="barge_in.no_speech")
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
                log_event("barge_in", "speech_start", log_id="barge_in.speech_start")
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
            text = self._speak_stream_plain(text_chunks, stop_event=playback_stop_event)
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
                    log_event(
                        "barge_in",
                        "timeout",
                        log_id="barge_in.timeout",
                        reason="waiting_for_utterance",
                    )
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
            log_event("barge_in", "error", log_id="barge_in.error", error=error)
            return text, None

        utterance = state.get("utterance")
        if isinstance(utterance, Utterance):
            transcript = state.get("transcript")
            if isinstance(transcript, str):
                if not transcript.strip():
                    self._clear_pending_barge_in()
                    log_event(
                        "barge_in",
                        "no_speech",
                        log_id="barge_in.no_speech",
                        reason="empty_transcript",
                    )
                    return text, None
                self._pending_barge_transcript = transcript
                self._pending_barge_stt_ms = int(state.get("stt_ms") or 0)
                extra_timings = state.get("stt_extra_timings")
                self._pending_barge_stt_extra_timings = (
                    dict(extra_timings) if isinstance(extra_timings, dict) else {}
                )
            log_event(
                "barge_in",
                "captured",
                log_id="barge_in.captured",
                duration_ms=utterance.duration_ms,
            )
            return text, utterance
        log_event("barge_in", "no_speech", log_id="barge_in.no_speech")
        return text, None

    def _wait_for_wake(self) -> tuple[WakeEvent, int]:
        wake_started = time.monotonic()
        wake = self.wake.wait(self.wake_audio)
        wake_ms = int((time.monotonic() - wake_started) * 1000)
        log_event(
            "wake",
            "detected",
            log_id="wake.detected",
            engine=wake.engine,
            label=wake.label,
            confidence=f"{wake.confidence:.3f}",
            latency_ms=wake_ms,
        )
        dump = self.audio_dump.write_voice_path_dump(
            None,
            "wake",
            wake.pcm,
            sample_rate=wake.sample_rate,
            duration_ms=wake.duration_ms,
        )
        if dump is not None:
            wake.dump_path = str(dump.path)
            wake.dump_start_ms = dump.start_ms
            wake.dump_end_ms = dump.end_ms
            log_event(
                "wake",
                "audio_saved",
                log_id="wake.audio_saved",
                path=dump.path,
            )
        return wake, wake_ms

    def _start_wake_ack(self) -> _WakeAckHandle:
        result: dict[str, int] = {}

        def play() -> None:
            ack_started = time.monotonic()
            try:
                self.wake_ack.play()
            except Exception as exc:
                log_event("wake_ack", "error", log_id="wake_ack.error", error=exc)
                result["latency_ms"] = 0
                return
            ack_ms = int((time.monotonic() - ack_started) * 1000)
            result["latency_ms"] = ack_ms
            if ack_ms:
                log_event(
                    "wake_ack",
                    "played",
                    log_id="wake_ack.played",
                    latency_ms=ack_ms,
                    mode="background",
                )

        thread = threading.Thread(target=play, name="voiceui-wake-ack", daemon=True)
        thread.start()
        return _WakeAckHandle(thread=thread, result=result)

    def _run_audio_turn(
        self,
        wake: WakeEvent,
        wake_ms: int,
        wake_ack_handle: _WakeAckHandle | None = None,
        speech_start_timeout_seconds: float = 0.0,
        max_speech_ms: int | None = None,
        turn_index: int | None = None,
    ) -> tuple[AssistantReply, str]:
        if turn_index is None:
            turn_index = self.audio_dump.begin_turn()
        stt_extra_timings: dict[str, int] = {}
        gate_source = "wake"
        if self._pending_barge_utterance is not None:
            gate_source = "barge_in"
            utterance = self._pending_barge_utterance
            pending_transcript = self._pending_barge_transcript
            self._pending_barge_utterance = None
            self._pending_barge_transcript = None
            vad_ms = 0
            log_event(
                "vad",
                "completed",
                log_id="vad.completed",
                source="barge_in",
                duration_ms=utterance.duration_ms,
                latency_ms=0,
            )
            wake_ack_ms = wake_ack_handle.join() if wake_ack_handle is not None else 0
            if pending_transcript is not None:
                transcript = pending_transcript
                stt_ms = self._pending_barge_stt_ms
                stt_extra_timings = dict(self._pending_barge_stt_extra_timings)
                self._pending_barge_stt_ms = 0
                self._pending_barge_stt_extra_timings = {}
                log_event(
                    "stt",
                    "completed",
                    log_id="stt.completed",
                    source="barge_in_stream",
                    latency_ms=stt_ms,
                    text=transcript,
                )
            else:
                stt_started = time.monotonic()
                transcript = self.stt.transcribe(utterance)
                stt_ms = int((time.monotonic() - stt_started) * 1000)
                log_event(
                    "stt",
                    "completed",
                    log_id="stt.completed",
                    latency_ms=stt_ms,
                    text=transcript,
                )
        elif self._should_stream_stt():
            if wake.engine == "follow_up":
                gate_source = "follow_up"
            (
                utterance,
                transcript,
                vad_ms,
                stt_ms,
                stt_extra_timings,
            ) = self._record_and_stream_transcribe(
                speech_start_timeout_seconds=speech_start_timeout_seconds,
                max_speech_ms=max_speech_ms,
            )
            wake_ack_ms = wake_ack_handle.join() if wake_ack_handle is not None else 0
        else:
            if wake.engine == "follow_up":
                gate_source = "follow_up"
            vad_started = time.monotonic()
            utterance = self.vad.record(
                self.command_audio,
                start_timeout_seconds=speech_start_timeout_seconds,
                max_speech_ms=max_speech_ms,
            )
            vad_ms = int((time.monotonic() - vad_started) * 1000)
            log_event(
                "vad",
                "completed",
                log_id="vad.completed",
                duration_ms=utterance.duration_ms,
                latency_ms=vad_ms,
            )

            wake_ack_ms = wake_ack_handle.join() if wake_ack_handle is not None else 0
            stt_started = time.monotonic()
            transcript = self.stt.transcribe(utterance)
            stt_ms = int((time.monotonic() - stt_started) * 1000)
            log_event(
                "stt",
                "completed",
                log_id="stt.completed",
                latency_ms=stt_ms,
                text=transcript,
            )

        transcript = transcript.strip()
        returned_transcript = transcript
        if not transcript:
            reply = AssistantReply(text=_EMPTY_INPUT_RESPONSE, routed_to="system")
            response_timings = {"llm": 0, "tts": 0}
            log_event("assistant", "empty_input", log_id="assistant.empty_input")
        elif self._is_voice_termination_command(transcript):
            reply, response_timings = self._finish_voice_termination_response(
                transcript,
                {},
                source=gate_source,
            )
            returned_transcript = ""
        else:
            gate_decision, _gate_reason, gate_response = self._gate_voice_transcript(
                transcript,
                gate_source,
            )
            if gate_decision == "reject":
                reply = AssistantReply(text=_EMPTY_INPUT_RESPONSE, routed_to="input_gate")
                response_timings = {"llm": 0, "tts": 0}
                returned_transcript = ""
            elif gate_decision == "clarify":
                reply, response_timings = self._finish_input_gate_clarification(
                    gate_response,
                    {},
                )
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
            log_event("debug", "saved", log_id="debug.saved", path=debug_dir)
        return reply, returned_transcript

    def _streaming_stt_policy(self) -> StreamFramePolicy:
        return _streaming_frame_policy(
            self.config.vad,
            self.command_audio.block_ms,
        )

    def _start_standby_streaming_stt(self, *, phase: str) -> None:
        if not self._should_stream_stt():
            return
        if self._standby_stt_handle is not None:
            return
        handle = _StreamingSttHandle(
            self.stt,
            self.command_audio.sample_rate,
            policy=self._streaming_stt_policy(),
        )
        handle.start()
        self._standby_stt_handle = handle
        log_event(
            "stt",
            "streaming_preopened",
            log_id="stt.streaming_preopened",
            phase=phase,
        )

    def _take_standby_streaming_stt(self) -> _StreamingSttHandle:
        handle = self._standby_stt_handle
        self._standby_stt_handle = None
        if handle is None:
            handle = _StreamingSttHandle(
                self.stt,
                self.command_audio.sample_rate,
                policy=self._streaming_stt_policy(),
            )
            handle.start()
        return handle

    def _should_stream_stt(self) -> bool:
        supports_streaming = getattr(self.stt, "supports_streaming", None)
        return bool(callable(supports_streaming) and supports_streaming())

    def _record_and_stream_transcribe(
        self,
        *,
        speech_start_timeout_seconds: float,
        max_speech_ms: int | None = None,
    ) -> tuple[Utterance, str, int, int, dict[str, int]]:
        vad_started = time.monotonic()
        stream_handle: _StreamingSttHandle | None = self._take_standby_streaming_stt()
        ready = stream_handle.wait_ready(timeout=0)
        log_event(
            "stt",
            "streaming_started",
            log_id="stt.streaming_started",
            vad_elapsed_ms=0,
            phase="pre_speech",
            ready=ready,
            ready_ms=stream_handle.ready_latency_ms() or 0,
        )

        def on_speech_start() -> None:
            start_ms = int((time.monotonic() - vad_started) * 1000)
            log_event(
                "stt",
                "streaming_speech_started",
                log_id="stt.streaming_speech_started",
                vad_elapsed_ms=start_ms,
            )

        def on_speech_audio(pcm: bytes) -> None:
            assert stream_handle is not None
            stream_handle.write(pcm)

        try:
            utterance = self.vad.record(
                self.command_audio,
                start_timeout_seconds=speech_start_timeout_seconds,
                on_speech_start=on_speech_start,
                on_speech_audio=on_speech_audio,
                max_speech_ms=max_speech_ms,
            )
        except Exception:
            if stream_handle is not None:
                stream_handle.abort()
            raise

        vad_ms = int((time.monotonic() - vad_started) * 1000)
        log_event(
            "vad",
            "completed",
            log_id="vad.completed",
            duration_ms=utterance.duration_ms,
            latency_ms=vad_ms,
        )

        if stream_handle is None:
            stt_started = time.monotonic()
            transcript = self.stt.transcribe(utterance)
            stt_ms = int((time.monotonic() - stt_started) * 1000)
            log_event(
                "stt",
                "completed",
                log_id="stt.completed",
                latency_ms=stt_ms,
                mode="fallback",
                text=transcript,
            )
            return utterance, transcript, vad_ms, stt_ms, {}

        finalize_started = time.monotonic()
        transcript = stream_handle.finish()
        stt_ms = int((time.monotonic() - finalize_started) * 1000)
        if not transcript.strip():
            fallback_started = time.monotonic()
            transcript = self._fallback_transcribe_empty_stream(
                utterance,
                source="wake",
                stream_total_latency_ms=stream_handle.total_latency_ms(),
                sent_chunks=stream_handle.sent_chunks,
                dropped_chunks=stream_handle.dropped_chunks,
            )
            stt_ms += int((time.monotonic() - fallback_started) * 1000)
        stt_total_ms = stream_handle.total_latency_ms()
        ready_ms = stream_handle.ready_latency_ms()
        params = {
            "latency_ms": stt_ms,
            "mode": "streaming",
            "total_latency_ms": stt_total_ms,
            "sent_chunks": stream_handle.sent_chunks,
            "dropped_chunks": stream_handle.dropped_chunks,
            "text": transcript,
        }
        if ready_ms is not None:
            params["ready_ms"] = ready_ms
        log_event(
            "stt",
            "completed",
            log_id="stt.completed",
            **params,
        )
        return utterance, transcript, vad_ms, stt_ms, {"stt_total": stt_total_ms}

    def _fallback_transcribe_empty_stream(
        self,
        utterance: Utterance,
        *,
        source: str,
        stream_total_latency_ms: int,
        sent_chunks: int,
        dropped_chunks: int,
    ) -> str:
        log_event(
            "stt",
            "streaming_empty_fallback",
            log_id="stt.streaming_empty_fallback",
            source=source,
            utterance_duration_ms=utterance.duration_ms,
            utterance_bytes=len(utterance.pcm),
            stream_total_latency_ms=stream_total_latency_ms,
            sent_chunks=sent_chunks,
            dropped_chunks=dropped_chunks,
            ok="start",
        )
        fallback_started = time.monotonic()
        try:
            refresh_token = getattr(self.stt, "refresh_token", None)
            if callable(refresh_token):
                refresh_token(reason="streaming_empty_fallback")
            transcript = self.stt.transcribe(utterance)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_event(
                "stt",
                "streaming_empty_fallback",
                log_id="stt.streaming_empty_fallback",
                source=source,
                latency_ms=int((time.monotonic() - fallback_started) * 1000),
                ok=False,
                error=exc,
                text="",
            )
            return ""
        log_event(
            "stt",
            "streaming_empty_fallback",
            log_id="stt.streaming_empty_fallback",
            source=source,
            latency_ms=int((time.monotonic() - fallback_started) * 1000),
            ok=True,
            text=transcript,
        )
        return transcript

    def _wake_speech_start_timeout_seconds(self) -> float:
        return max(
            0.0,
            float(
                getattr(
                    self.config.conversation,
                    "wake_speech_start_timeout_seconds",
                    8.0,
                )
            ),
        )

    def _follow_up_max_speech_ms(self) -> int | None:
        value = int(getattr(self.config.conversation, "follow_up_max_speech_ms", 10000))
        if value <= 0:
            return None
        return value

    def _finish_wake_speech_timeout(
        self,
        wake_ack_handle: _WakeAckHandle | None,
    ) -> AssistantReply:
        if wake_ack_handle is not None:
            wake_ack_handle.join()
        timeout_seconds = self._wake_speech_start_timeout_seconds()
        log_event(
            "session",
            "wake_speech_timeout",
            log_id="session.wake_speech_timeout",
            seconds=timeout_seconds,
            next_state="returning_to_wake",
        )
        self.audio_dump.end_turn()
        return AssistantReply(text="", routed_to="system")

    def run_once(self) -> AssistantReply:
        self._start_system_input_dump()
        if self.config.input.mode == "text":
            try:
                text = input("you> ")
                return self.run_text_turn(text)
            finally:
                self.close()

        try:
            wake, wake_ms = self._wait_for_wake()
            self._start_standby_streaming_stt(phase="post_wake")
            with self._turn_lock:
                self._duck_music("conversation")
                try:
                    turn_index = self.audio_dump.ensure_turn()
                    wake_ack_handle = self._start_wake_ack()
                    try:
                        reply, _transcript = self._run_audio_turn(
                            wake,
                            wake_ms,
                            wake_ack_handle=wake_ack_handle,
                            speech_start_timeout_seconds=(
                                self._wake_speech_start_timeout_seconds()
                            ),
                            turn_index=turn_index,
                        )
                    except SpeechStartTimeoutError:
                        return self._finish_wake_speech_timeout(wake_ack_handle)
                    return reply
                finally:
                    self._unduck_music("conversation")
        finally:
            self.close()

    def run_conversation(self, keep_audio_dump_running: bool = False) -> AssistantReply:
        self._start_system_input_dump()
        if self.config.input.mode == "text":
            try:
                return self.run_once()
            finally:
                if not keep_audio_dump_running:
                    self.close()

        try:
            wake, wake_ms = self._wait_for_wake()
            self._start_standby_streaming_stt(phase="post_wake")
            with self._turn_lock:
                self._duck_music("conversation")
                try:
                    turn_index = self.audio_dump.ensure_turn()
                    wake_ack_handle = self._start_wake_ack()
                    self.session.reset()
                    try:
                        reply, transcript = self._run_audio_turn(
                            wake,
                            wake_ms,
                            wake_ack_handle=wake_ack_handle,
                            speech_start_timeout_seconds=(
                                self._wake_speech_start_timeout_seconds()
                            ),
                            turn_index=turn_index,
                        )
                    except SpeechStartTimeoutError:
                        return self._finish_wake_speech_timeout(wake_ack_handle)
                    follow_up_seconds = self.config.conversation.follow_up_seconds
                    if (
                        follow_up_seconds <= 0 or not transcript
                    ) and self._pending_barge_utterance is None:
                        return reply

                    while True:
                        if self._pending_barge_utterance is None:
                            if follow_up_seconds <= 0:
                                return reply
                            self._start_standby_streaming_stt(phase="follow_up")
                            log_event(
                                "session",
                                "listening_for_follow_up",
                                log_id="session.listening_for_follow_up",
                                seconds=follow_up_seconds,
                            )
                            speech_start_timeout_seconds = follow_up_seconds
                        else:
                            log_event(
                                "session",
                                "processing_barge_in",
                                log_id="session.processing_barge_in",
                            )
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
                                max_speech_ms=self._follow_up_max_speech_ms(),
                            )
                        except SpeechStartTimeoutError:
                            log_event(
                                "session",
                                "follow_up_timeout",
                                log_id="session.follow_up_timeout",
                                next_state="returning_to_wake",
                            )
                            return reply
                        if not transcript:
                            log_event(
                                "session",
                                "empty_follow_up",
                                log_id="session.empty_follow_up",
                                next_state="returning_to_wake",
                            )
                            return reply
                finally:
                    self._unduck_music("conversation")
        finally:
            if not keep_audio_dump_running:
                self.close()

    def run_forever(self) -> None:
        self._start_system_input_dump()
        cron_scheduler = self._create_cron_scheduler()
        if cron_scheduler is not None:
            cron_scheduler.start()
        last_error_key = ""
        repeated_errors = 0
        try:
            while True:
                try:
                    self.run_conversation(keep_audio_dump_running=True)
                    last_error_key = ""
                    repeated_errors = 0
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    error_key = f"{type(exc).__name__}:{exc}"
                    if error_key == last_error_key:
                        repeated_errors += 1
                    else:
                        last_error_key = error_key
                        repeated_errors = 1
                    sleep_seconds = _runtime_error_backoff_seconds(repeated_errors)
                    log_event(
                        "error",
                        "runtime",
                        log_id="error.runtime",
                        error=exc,
                        repeat_count=repeated_errors,
                        retry_sleep_seconds=f"{sleep_seconds:.1f}",
                    )
                    time.sleep(sleep_seconds)
        finally:
            if cron_scheduler is not None:
                cron_scheduler.stop()
            self.close()


def _looks_like_volume_query(normalized: str) -> bool:
    return any(term in normalized for term in ("音量", "声音", "静音", "volume", "mute"))


def _parse_volume_request(normalized: str) -> dict[str, object]:
    if any(term in normalized for term in ("静音", "mute")):
        if any(term in normalized for term in ("取消静音", "解除静音", "unmute")):
            return {"action": "set", "muted": False}
        return {"action": "set", "muted": True}
    if any(term in normalized for term in ("多少", "几", "当前", "现在", "查询", "查看")):
        return {"action": "get"}
    percent = _extract_percent_number(normalized)
    if any(term in normalized for term in ("调低", "降低", "小声", "小一点", "降", "低一点")):
        relative = -(percent if percent is not None else 10.0)
        return {"action": "set", "relative_percent": relative}
    louder_terms = ("调高", "提高", "大声", "大一点", "增大", "升高", "加大")
    if any(term in normalized for term in louder_terms):
        return {"action": "set", "relative_percent": percent if percent is not None else 10.0}
    if percent is not None:
        return {"action": "set", "volume_percent": percent}
    return {"action": "get"}


def _extract_percent_number(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)%?", text)
    if not match:
        return None
    return max(0.0, min(100.0, float(match.group(1))))


def _looks_like_weather_query(normalized: str) -> bool:
    if _looks_like_iot_temperature_text(normalized):
        return False
    return any(term in normalized for term in ("天气", "气温", "温度", "下雨", "降雨", "有雨"))


def _looks_like_iot_temperature_text(normalized: str) -> bool:
    if "温度" not in normalized and "度" not in normalized:
        return False
    device_terms = (
        "空调",
        "冷气",
        "空调机",
        "加湿器",
        "净化器",
        "空气净化器",
        "传感器",
        "设备",
    )
    if any(term in normalized for term in device_terms):
        return True
    return any(
        term in normalized
        for term in ("调成", "调到", "调高", "调低", "设置", "设为")
    )


def _scheduled_miot_arguments_from_preview(
    command_text: str,
    preview: dict[str, object],
) -> dict[str, object]:
    arguments: dict[str, object] = {"request": command_text}
    query = preview.get("query") if isinstance(preview.get("query"), dict) else {}
    device = preview.get("device") if isinstance(preview.get("device"), dict) else {}

    for key in ("area", "device", "device_class", "action"):
        value = str(query.get(key) or "").strip()
        if value:
            arguments[key] = value

    if device:
        if device.get("name"):
            arguments["device"] = str(device["name"])
        if device.get("room_name"):
            arguments["area"] = str(device["room_name"])
        if device.get("device_class"):
            arguments["device_class"] = str(device["device_class"])

    action = str(preview.get("action") or "").strip()
    if action:
        arguments["action"] = action

    if "target_value" in preview and preview.get("target_value") is not None:
        arguments["value"] = preview["target_value"]
    elif isinstance(query, dict) and "value" in query and query.get("value") is not None:
        arguments["value"] = query["value"]

    return arguments


def _looks_like_time_query(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "几点",
            "几点了",
            "现在时间",
            "当前时间",
            "报一下时间",
            "告诉我时间",
        )
    )


def _voice_input_gate_enabled(conversation_config: object, source: str) -> bool:
    if not bool(getattr(conversation_config, "input_gate_enabled", True)):
        return False
    if source == "barge_in":
        return bool(getattr(conversation_config, "barge_in_gate_enabled", True))
    if source == "follow_up":
        return bool(getattr(conversation_config, "follow_up_gate_enabled", True))
    return True


def _classify_voice_input(transcript: str, source: str) -> tuple[str, str]:
    text = transcript.strip()
    if _is_unusable_transcript(text):
        return "reject", "unusable_text"
    if source == "wake" and _looks_like_false_wake_complaint(text):
        return "reject", "false_wake_complaint"
    strong_intent = _looks_like_strong_voice_intent(text)
    if source == "wake" and _looks_like_background_wake_text(text, strong_intent):
        return "reject", "wake_background_like"
    if source in {"barge_in", "follow_up"}:
        if _looks_like_contextual_correction(text):
            return "accept", "contextual_correction"
        if len(text) >= 32 and _looks_like_background_monologue(text) and not strong_intent:
            return "reject", "background_monologue"
    if _looks_like_direct_voice_intent(text):
        return "accept", "direct_intent"
    if source == "wake":
        return "reject", "wake_no_direct_intent"
    if source == "barge_in":
        return "reject", "ambiguous_barge_in"
    if len(text) <= 24:
        return "clarify", "ambiguous_short"
    return "clarify", "ambiguous_follow_up"


def _is_unusable_transcript(text: str) -> bool:
    if not text:
        return True
    meaningful = [
        char
        for char in text
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    ]
    if not meaningful:
        return True
    return len(set(meaningful)) <= 1 and len(meaningful) >= 4


def _normalize_self_echo_text(text: str) -> str:
    return "".join(
        char
        for char in text.lower()
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _self_echo_matched_chars(transcript: str, spoken: str) -> int:
    if not transcript or not spoken:
        return 0
    if transcript in spoken:
        return len(transcript)
    match = SequenceMatcher(None, transcript, spoken).find_longest_match(
        0,
        len(transcript),
        0,
        len(spoken),
    )
    return match.size


def _is_self_echo_match(transcript: str, spoken: str, matched_chars: int) -> bool:
    text_len = len(transcript)
    if text_len < 2 or not spoken:
        return False
    if transcript in spoken:
        if text_len >= 4:
            return True
        return spoken.startswith(transcript)
    if text_len <= 3:
        return _common_prefix_chars(transcript, spoken) >= 2 and "我没太听清" in spoken
    if text_len <= 8 and spoken.startswith(transcript[:4]):
        return matched_chars >= 4
    return False


def _looks_like_false_wake_complaint(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    return any(
        term in normalized
        for term in (
            "谁叫你了",
            "谁喊你了",
            "没叫你",
            "没有叫你",
            "没喊你",
            "没有喊你",
            "不是叫你",
            "不是喊你",
        )
    )


def _runtime_error_backoff_seconds(repeat_count: int) -> float:
    if repeat_count <= 1:
        return 1.0
    return min(60.0, float(2 ** min(repeat_count - 1, 6)))


def _common_prefix_chars(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        count += 1
    return count


def _normalize_termination_text(text: str) -> str:
    return "".join(
        char
        for char in text.lower()
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _matches_termination_command(text: str, phrases: object) -> bool:
    normalized = _normalize_termination_text(text)
    if not normalized:
        return False
    if not isinstance(phrases, list | tuple | set):
        phrases = DEFAULT_VOICE_TERMINATION_PHRASES
    normalized_phrases = {
        phrase_normalized
        for phrase in phrases
        if (phrase_normalized := _normalize_termination_text(str(phrase)))
    }
    return normalized in normalized_phrases


def _looks_like_end_conversation_command(text: str) -> bool:
    if _matches_termination_command(text, DEFAULT_VOICE_TERMINATION_PHRASES):
        return True
    normalized = text.lower().strip(" \t\r\n，。！？,.!?;；：:")
    normalized = normalized.replace(" ", "")
    if normalized in {
        "退出",
        "退出吧",
        "结束",
        "结束吧",
        "结束对话",
        "不聊了",
        "先这样",
        "先这样吧",
        "就这样",
        "就这样吧",
        "没事了",
        "不用了",
        "算了",
        "休眠",
        "睡眠",
        "停",
        "停止",
        "stop",
        "exit",
        "quit",
    }:
        return True
    return normalized.startswith("退出") and len(normalized) <= 5


def _looks_like_direct_voice_intent(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    if _looks_like_assistant_name_greeting(text):
        return True
    if (
        normalized.isascii()
        and any(char.isalpha() for char in normalized)
        and len(normalized) <= 32
    ):
        return True
    if _looks_like_iot_voice_intent(normalized):
        return True
    if _looks_like_weather_query(normalized) or _looks_like_time_query(normalized):
        return True
    if any(mark in text for mark in ("?", "？")):
        return True
    question_terms = (
        "吗",
        "呢",
        "什么",
        "怎么",
        "为什么",
        "多少",
        "哪个",
        "哪一个",
        "谁",
        "能不能",
        "可不可以",
        "是不是",
    )
    if any(term in normalized for term in question_terms):
        return True
    intent_terms = (
        "帮我",
        "请",
        "麻烦",
        "你能",
        "你可以",
        "告诉我",
        "说一下",
        "讲一下",
        "讲个",
        "介绍一下",
        "查一下",
        "查查",
        "查询",
        "搜索",
        "搜一下",
        "百度",
        "打开",
        "关闭",
        "开灯",
        "关灯",
        "开空调",
        "关空调",
        "播放",
        "停止播放",
        "暂停",
        "继续",
        "取消",
        "不用了",
        "不要了",
        "算了",
        "闹钟",
        "提醒",
        "定时",
        "音量",
        "静音",
        "加油",
        "你好",
    )
    if any(term in normalized for term in intent_terms):
        return True
    if normalized in {"好的", "好", "可以", "不要", "不用", "停", "停止"}:
        return True
    return False


def _looks_like_assistant_name_greeting(text: str) -> bool:
    normalized = _normalize_spoken_text(text)
    if not normalized:
        return False
    assistant_names = ("丽娜", "琳娜", "莉娜", "leela", "lina", "lena")
    if normalized in assistant_names:
        return True
    if not any(name in normalized for name in assistant_names):
        return False
    greetings = ("hello", "hi", "hey", "哈喽", "嗨", "嘿", "你好")
    return any(normalized.startswith(greeting) for greeting in greetings)


def _normalize_spoken_text(text: str) -> str:
    return "".join(
        char
        for char in text.lower()
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _looks_like_strong_voice_intent(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    if _looks_like_iot_voice_intent(normalized):
        return True
    if _looks_like_weather_query(normalized) or _looks_like_time_query(normalized):
        return True
    strong_terms = (
        "帮我",
        "请",
        "麻烦",
        "你能",
        "你可以",
        "告诉我",
        "说一下",
        "讲一下",
        "查一下",
        "查查",
        "查询",
        "搜索",
        "搜一下",
        "百度",
        "播放",
        "停止播放",
        "暂停",
        "继续",
        "取消",
        "闹钟",
        "提醒",
        "定时",
        "音量",
        "静音",
        "你好",
    )
    return any(term in normalized for term in strong_terms)


def _looks_like_iot_voice_intent(normalized: str) -> bool:
    device_terms = (
        "米家",
        "灯",
        "开关",
        "窗帘",
        "帘",
        "空调",
        "冷气",
        "空调机",
        "净化器",
        "加湿器",
        "插座",
        "插排",
        "排插",
        "风扇",
        "电扇",
        "吊扇",
        "设备",
    )
    if not any(term in normalized for term in device_terms):
        return False
    if any(
        term in normalized
        for term in (
            "打开",
            "开启",
            "关闭",
            "关掉",
            "关上",
            "关了",
            "开了",
            "开灯",
            "关灯",
            "拉上",
            "拉开",
            "调成",
            "调到",
            "设置",
            "设为",
            "亮度",
            "温度",
            "模式",
            "制冷",
            "制热",
            "除湿",
            "睡眠",
        )
    ):
        return True
    return normalized.startswith(("开", "关"))


def _looks_like_contextual_correction(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    if len(normalized) > 32:
        return False
    if any(term in normalized for term in ("我说的是", "刚才说的是", "应该是", "是说")):
        return True
    return "不是" in normalized and "是不是" not in normalized and (
        normalized.startswith("不是")
        or normalized.startswith("是")
        or "，不是" in normalized
        or ",不是" in normalized
        or len(normalized) <= 18
    )


def _looks_like_background_wake_text(text: str, strong_intent: bool) -> bool:
    if strong_intent:
        return False
    if len(text) >= 45 and _looks_like_background_monologue(text):
        return True
    normalized = text.replace(" ", "")
    if len(normalized) >= 18 and _looks_like_background_continuation(normalized):
        return True
    return False


def _looks_like_background_continuation(normalized: str) -> bool:
    if any(term in normalized for term in ("往下看你就懂了", "你就懂了", "这话放在")):
        return True
    if "嘛，就" in normalized or "嘛,就" in normalized:
        return True
    stripped = normalized.rstrip("。.!！?？,，;；")
    return stripped.endswith(("因为", "所以", "然后", "但是", "不过", "主要是因为"))


def _looks_like_background_monologue(text: str) -> bool:
    punctuation_count = sum(text.count(mark) for mark in ("，", "。", ",", ".", "；", ";"))
    digit_count = sum(1 for char in text if char.isdigit())
    filler_terms = ("然后", "所以", "我们", "这个", "那个", "其实", "因为", "如果")
    filler_count = sum(text.count(term) for term in filler_terms)
    return punctuation_count >= 1 or digit_count >= 2 or filler_count >= 2


def _clarification_response_for_text(transcript: str) -> str:
    normalized = transcript.replace(" ", "")
    if "临时" in normalized and any(term in normalized for term in ("工", "用工", "民工")):
        return "我没太听清，你是想找临时用工渠道吗？"
    if "闹钟" in normalized or "提醒" in normalized:
        return "你想让我什么时候提醒？"
    return _INPUT_CLARIFICATION_RESPONSE


def _compact_spoken_response(response: str, max_chars: int) -> str:
    cleaned = _clean_spoken_response(response)
    if len(cleaned) <= max_chars:
        return cleaned
    if any(term in cleaned for term in ("输入似乎", "表述似乎", "有些混乱", "让人困惑")):
        return _INPUT_CLARIFICATION_RESPONSE
    first_sentence = _first_spoken_sentence(cleaned)
    if first_sentence and len(first_sentence) <= max_chars:
        return first_sentence
    if max_chars <= 1:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 1].rstrip("，,；;：:、 ") + "。"


def _clean_spoken_response(response: str) -> str:
    lines: list[str] = []
    for raw_line in response.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip().replace("**", "")
        if not line:
            continue
        line = line.lstrip("-* ")
        while line and line[0].isdigit():
            line = line[1:].lstrip(".、)） ")
        if line:
            lines.append(line)
    return " ".join(lines).strip()


def _first_spoken_sentence(text: str) -> str:
    end_positions = [
        text.find(mark)
        for mark in ("。", "！", "？", "!", "?")
        if text.find(mark) >= 0
    ]
    if not end_positions:
        return ""
    end = min(end_positions) + 1
    return text[:end].strip()


def _extract_weather_location(transcript: str, default_location: str = "") -> str:
    text = transcript.strip()
    marker_index = -1
    for term in (
        "天气",
        "气温",
        "温度",
        "下雨",
        "降雨",
        "有雨",
    ):
        index = text.find(term)
        if index >= 0 and (marker_index < 0 or index < marker_index):
            marker_index = index
    candidates = []
    if marker_index >= 0:
        candidates.append(text[:marker_index])
    candidates.append(text)
    for candidate in candidates:
        cleaned = _clean_weather_location_candidate(candidate)
        if _is_weather_location_candidate(cleaned):
            return cleaned
    return default_location.strip()


def _extract_weather_target_day(transcript: str) -> str:
    if "明天" in transcript or "明日" in transcript:
        return "tomorrow"
    return "today"




def _streaming_frame_policy(vad_config, audio_block_ms: int) -> StreamFramePolicy:
    return StreamFramePolicy.from_vad(vad_config, audio_block_ms=audio_block_ms)

def _format_weather_error_response(exc: Exception) -> str:
    message = str(exc)
    if "Could not find weather location" in message:
        return "我没找到这个地点的天气，你可以说具体城市或区域。"
    return "天气服务暂时不可用，稍后再试。"


def _format_processing_error_response(exc: Exception) -> str:
    del exc
    return "刚才处理失败了，请再说一遍。"


def _tool_progress_label_for_text(text: str) -> str:
    normalized = text.lower().replace(" ", "")
    if any(
        term in normalized
        for term in (
            "\u641c\u7d22",
            "\u767e\u5ea6",
            "tavily",
            "websearch",
            "\u65b0\u95fb",
            "\u6700\u65b0",
        )
    ):
        return "search"
    if any(
        term in normalized
        for term in (
            "\u7c73\u5bb6",
            "\u5bb6\u91cc",
            "\u51c0\u5316\u5668",
            "\u7a7a\u6c14\u51c0\u5316\u5668",
            "\u8bbe\u5907",
            "\u663e\u793a",
        )
    ):
        return "device"
    return "tool"


def _tool_progress_prompt_for_text(text: str) -> str:
    label = _tool_progress_label_for_text(text)
    if label == "search":
        return "\u6b63\u5728\u641c\u7d22\uff0c\u8bf7\u7a0d\u7b49\u3002"
    if label == "device":
        return "\u6b63\u5728\u67e5\u8be2\u8bbe\u5907\u72b6\u6001\uff0c\u8bf7\u7a0d\u7b49\u3002"
    return "\u6b63\u5728\u5904\u7406\uff0c\u8bf7\u7a0d\u7b49\u3002"


def _clean_weather_location_candidate(text: str) -> str:
    cleaned = text.strip()
    variants = [cleaned]
    for separator in ("，", ",", "。", "；", ";"):
        if separator in cleaned:
            variants.insert(0, cleaned.rsplit(separator, 1)[-1])

    for variant in variants:
        candidate = variant
        for term in (
            "你知道",
            "我想知道",
            "想知道",
            "帮我查一下",
            "帮我看一下",
            "帮我查查",
            "帮我",
            "你看一下",
            "查一下",
            "看一下",
            "告诉我",
            "请问",
            "麻烦",
            "我在",
            "今天",
            "明天",
            "明日",
            "现在",
            "当前",
            "当地",
            "这边",
            "会不会",
            "有没有",
            "天气",
            "气温",
            "温度",
            "下雨",
            "降雨",
            "有雨",
            "怎么样",
            "如何",
            "怎样",
            "的",
            "吗",
            "呢",
            "？",
            "?",
            "。",
            "，",
            ",",
            "、",
            "！",
            "!",
        ):
            candidate = candidate.replace(term, "")
        candidate = candidate.strip()
        if candidate:
            return candidate
    return ""


def _is_weather_location_candidate(candidate: str) -> bool:
    if not 1 < len(candidate) <= 12:
        return False
    return candidate not in {"你知道", "知道", "帮我", "请问", "麻烦", "今天", "明天", "明日"}
