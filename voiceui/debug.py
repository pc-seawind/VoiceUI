from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voiceui.audio_dump import AudioDumpFile, AudioDumpManager
from voiceui.models import DebugConfig, Utterance, WakeEvent


@dataclass(slots=True)
class TurnDebugData:
    node_id: str
    room: str
    wake: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=dict)
    utterance: dict[str, Any] = field(default_factory=dict)
    transcript: str = ""
    reply: str = ""
    routed_to: str = "llm"
    errors: list[str] = field(default_factory=list)


class DebugRecorder:
    def __init__(self, config: DebugConfig, audio_dump: AudioDumpManager | None = None):
        self.config = config
        self.audio_dump = audio_dump or AudioDumpManager(config)
        self._barge_in_index = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def save_turn(
        self,
        data: TurnDebugData,
        utterance: Utterance | None = None,
        wake_audio: WakeEvent | None = None,
    ) -> Path | None:
        if not self.config.enabled:
            return None

        turn_index = self.audio_dump.ensure_turn()
        session_dir = self.audio_dump.debug_session_dir()
        if session_dir is None:
            return None

        if wake_audio is not None:
            data.wake["duration_ms"] = wake_audio.duration_ms
            data.wake["sample_rate"] = wake_audio.sample_rate
            data.wake["bytes"] = len(wake_audio.pcm)
            if wake_audio.pcm and self.config.save_audio:
                dump = self.audio_dump.write_voice_path_dump(
                    None,
                    "wake",
                    wake_audio.pcm,
                    sample_rate=wake_audio.sample_rate,
                    duration_ms=wake_audio.duration_ms,
                )
                _record_dump_metadata(data.wake, dump)

        if utterance and self.config.save_audio:
            dump = self.audio_dump.write_voice_path_dump(
                None,
                "utterance",
                utterance.pcm,
                sample_rate=utterance.sample_rate,
                duration_ms=utterance.duration_ms,
            )
            _record_dump_metadata(data.utterance, dump)

        if self.config.save_metadata:
            metadata = asdict(data)
            metadata["turn"] = turn_index
            _append_session_metadata(session_dir, "turns", metadata)

        self.audio_dump.end_turn(turn_index)
        return session_dir

    def save_barge_in_monitor(
        self,
        *,
        mode: str,
        result: str,
        pcm: bytes,
        sample_rate: int,
        duration_ms: int,
        metadata: dict[str, Any] | None = None,
        extra_wavs: dict[str, tuple[bytes, int, int]] | None = None,
    ) -> Path | None:
        if not self.config.enabled:
            return None
        if not pcm and not extra_wavs and not self.config.save_metadata:
            return None

        self._barge_in_index += 1
        created_turn = self.audio_dump.current_turn_index is None
        turn_index = self.audio_dump.ensure_turn()
        session_dir = self.audio_dump.debug_session_dir()
        if session_dir is None:
            return None

        data: dict[str, Any] = {
            "turn": turn_index,
            "barge_in_index": self._barge_in_index,
            "mode": mode,
            "result": result,
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "bytes": len(pcm),
        }
        if metadata:
            data.update(metadata)

        if pcm and self.config.save_audio:
            dump = self.audio_dump.write_voice_path_dump(
                None,
                "barge_in_monitor",
                pcm,
                sample_rate=sample_rate,
                duration_ms=duration_ms,
            )
            _record_dump_metadata(data, dump)

        if extra_wavs and self.config.save_audio:
            extra_paths: dict[str, str] = {}
            for filename, (extra_pcm, extra_sample_rate, extra_channels) in extra_wavs.items():
                if not extra_pcm:
                    continue
                dump = self.audio_dump.write_voice_path_dump(
                    None,
                    Path(filename).stem,
                    extra_pcm,
                    sample_rate=extra_sample_rate,
                    channels=extra_channels,
                )
                if dump is not None:
                    extra_paths[filename] = str(dump.path)
            if extra_paths:
                data["extra_wav_paths"] = extra_paths

        if self.config.save_metadata:
            _append_session_metadata(session_dir, "barge_in", data)

        if created_turn:
            self.audio_dump.end_turn(turn_index)
        return session_dir


def _record_dump_metadata(target: dict[str, Any], dump: AudioDumpFile | None) -> None:
    if dump is None:
        return
    target["dump_path"] = str(dump.path)
    target["dump_start_ms"] = dump.start_ms
    target["dump_end_ms"] = dump.end_ms


def _append_session_metadata(session_dir: Path, section: str, entry: dict[str, Any]) -> None:
    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {"turns": [], "barge_in": []}
    metadata.setdefault("turns", [])
    metadata.setdefault("barge_in", [])
    metadata.setdefault(section, []).append(entry)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
