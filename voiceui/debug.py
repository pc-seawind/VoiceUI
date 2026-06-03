from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from voiceui.audio import write_pcm16_wav
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
    def __init__(self, config: DebugConfig):
        self.config = config
        self._turn_index = 0
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

        self._turn_index += 1
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        turn_dir = Path(self.config.output_dir) / f"{stamp}-{self._turn_index:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        if wake_audio is not None:
            data.wake["duration_ms"] = wake_audio.duration_ms
            data.wake["sample_rate"] = wake_audio.sample_rate
            data.wake["bytes"] = len(wake_audio.pcm)
            if wake_audio.pcm and self.config.save_audio:
                wav_path = turn_dir / "wake.wav"
                write_pcm16_wav(wav_path, wake_audio.pcm, wake_audio.sample_rate)
                data.wake["wav_path"] = str(wav_path)

        if utterance and self.config.save_audio:
            wav_path = turn_dir / "utterance.wav"
            write_pcm16_wav(wav_path, utterance.pcm, utterance.sample_rate)
            data.utterance["wav_path"] = str(wav_path)

        if self.config.save_metadata:
            metadata_path = turn_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(asdict(data), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return turn_dir

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
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            Path(self.config.output_dir)
            / f"{stamp}-barge-in-{self._barge_in_index:04d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "mode": mode,
            "result": result,
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "bytes": len(pcm),
        }
        if metadata:
            data.update(metadata)

        if pcm and self.config.save_audio:
            wav_path = output_dir / "barge_in_monitor.wav"
            write_pcm16_wav(wav_path, pcm, sample_rate)
            data["wav_path"] = str(wav_path)

        if extra_wavs and self.config.save_audio:
            extra_paths: dict[str, str] = {}
            for filename, (extra_pcm, extra_sample_rate, extra_channels) in extra_wavs.items():
                if not extra_pcm:
                    continue
                wav_path = output_dir / filename
                write_pcm16_wav(
                    wav_path,
                    extra_pcm,
                    extra_sample_rate,
                    channels=extra_channels,
                )
                extra_paths[filename] = str(wav_path)
            if extra_paths:
                data["extra_wav_paths"] = extra_paths

        if self.config.save_metadata:
            metadata_path = output_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return output_dir
