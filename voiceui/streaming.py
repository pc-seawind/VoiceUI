from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StreamFramePolicy:
    frame_ms: int
    buffer_frames: int
    ready_timeout_seconds: float

    @classmethod
    def from_vad(
        cls,
        vad_config,
        *,
        audio_block_ms: int,
        min_ready_timeout_seconds: float = 0.5,
    ) -> StreamFramePolicy:
        frame_ms = max(1, int(getattr(vad_config, "frame_ms", audio_block_ms) or audio_block_ms))
        pre_roll_ms = max(0, int(getattr(vad_config, "pre_roll_ms", 0) or 0))
        min_speech_ms = max(0, int(getattr(vad_config, "min_speech_ms", 0) or 0))
        covered_ms = pre_roll_ms + min_speech_ms
        buffer_frames = max(1, (covered_ms + frame_ms - 1) // frame_ms)
        ready_timeout = max(min_ready_timeout_seconds, covered_ms / 1000.0)
        return cls(
            frame_ms=frame_ms,
            buffer_frames=buffer_frames,
            ready_timeout_seconds=ready_timeout,
        )

    @classmethod
    def text_audio_default(cls) -> StreamFramePolicy:
        return cls(frame_ms=20, buffer_frames=1, ready_timeout_seconds=0.5)


class BoundedBackpressureQueue(Generic[T]):
    def __init__(self, maxsize: int):
        self._items: queue.Queue[T] = queue.Queue(maxsize=max(1, maxsize))
        self.put_count = 0
        self.get_count = 0
        self.blocked_puts = 0
        self.blocked_put_ms = 0

    def put(self, item: T, *, timeout: float | None = None) -> None:
        started = time.monotonic()
        try:
            self._items.put(item, block=True, timeout=timeout)
        except queue.Full:
            raise
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if elapsed_ms > 0:
            self.blocked_puts += 1
            self.blocked_put_ms += elapsed_ms
        self.put_count += 1

    def get(self, *, timeout: float | None = None) -> T:
        try:
            item = self._items.get(block=True, timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError from exc
        self.get_count += 1
        return item

    def empty(self) -> bool:
        return self._items.empty()

    def stats(self) -> dict[str, int]:
        return {
            "put_count": self.put_count,
            "get_count": self.get_count,
            "blocked_puts": self.blocked_puts,
            "blocked_put_ms": self.blocked_put_ms,
        }
