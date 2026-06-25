from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from voiceui.audio import resolve_sounddevice_device, write_pcm16_wav
from voiceui.logs import configure_log_files, log_event
from voiceui.models import AudioConfig, DebugConfig


@dataclass(frozen=True, slots=True)
class AudioDumpFile:
    path: Path
    start_ms: int
    end_ms: int


class AudioDumpManager:
    def __init__(self, config: DebugConfig):
        self.config = config
        self.origin_mono = time.monotonic()
        self._session_dir: Path | None = None
        self._system_thread: threading.Thread | None = None
        self._system_stop_event: threading.Event | None = None
        self._system_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._turn_index = 0
        self._current_turn_index: int | None = None

    @property
    def audio_enabled(self) -> bool:
        return bool(self.config.enabled and self.config.save_audio)

    @property
    def voice_path_enabled(self) -> bool:
        return bool(self.audio_enabled and self.config.voice_path_dump_enabled)

    def elapsed_ms(self) -> int:
        return max(0, int((time.monotonic() - self.origin_mono) * 1000))

    @property
    def turn_scoped_sessions(self) -> bool:
        return getattr(self.config, "session_scope", "run") == "turn"

    def begin_turn(self) -> int:
        with self._session_lock:
            return self._begin_turn_locked()

    def ensure_turn(self) -> int:
        with self._session_lock:
            if self._current_turn_index is None:
                return self._begin_turn_locked()
            return self._current_turn_index

    def end_turn(self, turn_index: int | None = None) -> None:
        reset_log_path = False
        with self._session_lock:
            if turn_index is None or self._current_turn_index == turn_index:
                self._current_turn_index = None
                reset_log_path = self.turn_scoped_sessions
        if reset_log_path:
            configure_log_files(
                debug_log_path=self._root_debug_log_path(),
                text_record_dir=self.text_record_dir(),
            )

    def _begin_turn_locked(self) -> int:
        self._turn_index += 1
        self._current_turn_index = self._turn_index
        if self.turn_scoped_sessions:
            self._session_dir = _create_debug_session_dir(Path(self.config.output_dir))
            configure_log_files(
                debug_log_path=self._session_dir / "debug.log",
                text_record_dir=self.text_record_dir(),
            )
        return self._current_turn_index

    @property
    def current_turn_index(self) -> int | None:
        with self._session_lock:
            return self._current_turn_index

    def debug_session_dir(self) -> Path | None:
        if not self.config.enabled:
            return None
        with self._session_lock:
            if self.turn_scoped_sessions and self._current_turn_index is None:
                # In turn-scoped mode, opening the web page or starting the
                # service must not create an empty timestamped turn directory.
                return self._session_dir
            if self._session_dir is None:
                self._session_dir = _create_debug_session_dir(Path(self.config.output_dir))
            return self._session_dir

    def debug_log_path(self) -> Path | None:
        if not self.config.enabled:
            return None
        if self.turn_scoped_sessions:
            with self._session_lock:
                session_dir = (
                    self._session_dir if self._current_turn_index is not None else None
                )
            if session_dir is None:
                return self._root_debug_log_path()
            return session_dir / "debug.log"
        session_dir = self.debug_session_dir()
        if session_dir is None:
            return None
        return session_dir / "debug.log"

    def _root_debug_log_path(self) -> Path:
        return Path(self.config.output_dir) / "debug.log"

    def text_record_dir(self) -> Path | None:
        if not self.config.enabled:
            return None
        return Path(self.config.output_dir) / "text_records"

    def audio_dump_output_dir(self) -> Path:
        session_dir = self.debug_session_dir()
        if session_dir is None:
            return Path(self.config.output_dir) / "audio_dumps"
        return session_dir / "audio_dumps"

    def write_voice_path_dump(
        self,
        directory: str | Path | None,
        kind: str,
        pcm: bytes,
        *,
        sample_rate: int,
        channels: int = 1,
        start_ms: int | None = None,
        end_ms: int | None = None,
        duration_ms: int | None = None,
        turn_index: int | None = None,
    ) -> AudioDumpFile | None:
        if not self.voice_path_enabled or not pcm:
            return None
        channels = max(1, int(channels))
        if end_ms is None:
            end_ms = self.elapsed_ms()
        if start_ms is None:
            if duration_ms is None:
                duration_ms = _pcm_duration_ms(pcm, sample_rate, channels)
            start_ms = max(0, end_ms - max(0, int(duration_ms)))
        if turn_index is None:
            turn_index = self.ensure_turn()
        output_dir = Path(directory) if directory is not None else self.voice_path_output_dir()
        dump = self._write_dump(
            output_dir,
            kind,
            pcm,
            sample_rate=sample_rate,
            channels=channels,
            start_ms=start_ms,
            end_ms=end_ms,
            turn_index=turn_index,
        )
        log_event(
            "audio_dump",
            "voice_path_saved",
            log_id="audio_dump.voice_path_saved",
            kind=kind,
            path=dump.path,
            start_ms=dump.start_ms,
            end_ms=dump.end_ms,
            turn=turn_index,
        )
        return dump

    def voice_path_output_dir(self) -> Path:
        return self.audio_dump_output_dir()

    def start_system_input_dump(self, audio_config: AudioConfig) -> bool:
        if not (
            self.audio_enabled
            and self.config.system_input_dump_enabled
            and self.config.system_input_dump_segment_seconds > 0
        ):
            return False
        with self._system_lock:
            if self._system_thread is not None and self._system_thread.is_alive():
                return False
            self.origin_mono = time.monotonic()
            stop_event = threading.Event()
            self._system_stop_event = stop_event
            thread = threading.Thread(
                target=self._run_system_input_dump,
                args=(audio_config, stop_event),
                name="voiceui-audio-dump-system-input",
                daemon=True,
            )
            self._system_thread = thread
            thread.start()
            return True

    def stop_system_input_dump(self) -> None:
        with self._system_lock:
            stop_event = self._system_stop_event
            thread = self._system_thread
            self._system_stop_event = None
            self._system_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _run_system_input_dump(
        self,
        audio_config: AudioConfig,
        stop_event: threading.Event,
    ) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as exc:
            log_event("audio_dump", "error", log_id="audio_dump.error", error=exc)
            return

        frames = max(1, int(audio_config.sample_rate * audio_config.block_ms / 1000))
        kwargs = {
            "samplerate": audio_config.sample_rate,
            "channels": audio_config.channels,
            "dtype": "int16",
            "blocksize": frames,
        }
        device = resolve_sounddevice_device(sd, audio_config.device, kind="input")
        if device is not None:
            kwargs["device"] = device

        writer = _SegmentedWavDumpWriter(
            manager=self,
            output_dir=self.audio_dump_output_dir(),
            kind="system_input",
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            segment_ms=max(1, int(self.config.system_input_dump_segment_seconds * 1000)),
        )
        try:
            with sd.RawInputStream(**kwargs) as stream:
                log_event(
                    "audio_dump",
                    "system_input_started",
                    log_id="audio_dump.system_input_started",
                    device=audio_config.device,
                    resolved_device=device,
                    channels=audio_config.channels,
                    sample_rate=audio_config.sample_rate,
                    segment_seconds=self.config.system_input_dump_segment_seconds,
                )
                while not stop_event.is_set():
                    data, _overflowed = stream.read(frames)
                    writer.write(bytes(data))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log_event("audio_dump", "error", log_id="audio_dump.error", error=exc)
        finally:
            writer.close()
            log_event(
                "audio_dump",
                "system_input_stopped",
                log_id="audio_dump.system_input_stopped",
            )

    def _write_dump(
        self,
        directory: Path,
        kind: str,
        pcm: bytes,
        *,
        sample_rate: int,
        channels: int,
        start_ms: int,
        end_ms: int,
        turn_index: int | None = None,
    ) -> AudioDumpFile:
        directory.mkdir(parents=True, exist_ok=True)
        start_ms = max(0, int(start_ms))
        end_ms = max(start_ms, int(end_ms))
        path = directory / dump_filename(kind, start_ms, end_ms, turn_index=turn_index)
        write_pcm16_wav(path, pcm, sample_rate, channels=channels)
        return AudioDumpFile(path=path, start_ms=start_ms, end_ms=end_ms)


class _SegmentedWavDumpWriter:
    def __init__(
        self,
        *,
        manager: AudioDumpManager,
        output_dir: Path,
        kind: str,
        sample_rate: int,
        channels: int,
        segment_ms: int,
    ):
        self.manager = manager
        self.output_dir = output_dir
        self.kind = kind
        self.sample_rate = sample_rate
        self.channels = max(1, int(channels))
        self.segment_ms = max(1, int(segment_ms))
        self.segment_frames = max(1, int(sample_rate * self.segment_ms / 1000))
        self._chunks: list[bytes] = []
        self._start_ms: int | None = None
        self._next_start_ms: int | None = 0
        self._duration_frames = 0

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        frame_width = self.channels * 2
        playable_len = len(pcm) - (len(pcm) % frame_width)
        offset = 0
        while offset < playable_len:
            if self._start_ms is None:
                self._start_ms = (
                    self._next_start_ms
                    if self._next_start_ms is not None
                    else self.manager.elapsed_ms()
                )
            remaining_frames = self.segment_frames - self._duration_frames
            remaining_bytes = max(frame_width, remaining_frames * frame_width)
            chunk = pcm[offset : min(playable_len, offset + remaining_bytes)]
            self._chunks.append(chunk)
            self._duration_frames += len(chunk) // frame_width
            offset += len(chunk)
            if self._duration_frames >= self.segment_frames:
                self.close()

    def close(self) -> AudioDumpFile | None:
        if self._start_ms is None or not self._chunks:
            return None
        pcm = b"".join(self._chunks)
        start_ms = self._start_ms
        duration_ms = int(self._duration_frames * 1000 / self.sample_rate)
        end_ms = start_ms + duration_ms
        dump = self.manager._write_dump(
            self.output_dir,
            self.kind,
            pcm,
            sample_rate=self.sample_rate,
            channels=self.channels,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        log_event(
            "audio_dump",
            "segment_saved",
            log_id="audio_dump.segment_saved",
            kind=self.kind,
            path=dump.path,
            start_ms=dump.start_ms,
            end_ms=dump.end_ms,
        )
        self._chunks = []
        self._start_ms = None
        self._next_start_ms = end_ms
        self._duration_frames = 0
        return dump


_CURRENT_MANAGER: AudioDumpManager | None = None
_CURRENT_MANAGER_LOCK = threading.Lock()


def configure_audio_dump(manager: AudioDumpManager | None) -> None:
    global _CURRENT_MANAGER
    with _CURRENT_MANAGER_LOCK:
        _CURRENT_MANAGER = manager


def current_audio_dump_manager() -> AudioDumpManager | None:
    with _CURRENT_MANAGER_LOCK:
        return _CURRENT_MANAGER


def dump_filename(
    kind: str,
    start_ms: int,
    end_ms: int,
    *,
    turn_index: int | None = None,
) -> str:
    safe_kind = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in kind)
    start = _format_relative_time(max(0, start_ms))
    end = _format_relative_time(max(0, end_ms))
    if turn_index is None:
        return f"{safe_kind}_{start}_{end}.wav"
    return f"{safe_kind}_{max(0, int(turn_index)):02d}_{start}_{end}.wav"


def _format_relative_time(ms: int) -> str:
    hours, remainder = divmod(max(0, int(ms)), 60 * 60 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}.{minutes:02d}.{seconds:02d}.{millis:03d}"


def _create_debug_session_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    for index in range(100):
        suffix = "" if index == 0 else f"-{index:02d}"
        path = root / f"{stamp}{suffix}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    path = root / f"{stamp}-{time.monotonic_ns()}"
    path.mkdir()
    return path


def _pcm_duration_ms(pcm: bytes, sample_rate: int, channels: int) -> int:
    if sample_rate <= 0 or channels <= 0:
        return 0
    return int(len(pcm) / 2 / channels / sample_rate * 1000)
