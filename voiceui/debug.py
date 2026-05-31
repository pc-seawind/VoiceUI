from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from voiceui.audio import write_pcm16_wav
from voiceui.models import DebugConfig, Utterance


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

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def save_turn(self, data: TurnDebugData, utterance: Utterance | None = None) -> Path | None:
        if not self.config.enabled:
            return None

        self._turn_index += 1
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        turn_dir = Path(self.config.output_dir) / f"{stamp}-{self._turn_index:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

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
