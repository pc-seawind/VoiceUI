from __future__ import annotations

import unittest
from collections.abc import Iterator

from voiceui.models import VadConfig
from voiceui.vad import EnergyVadRecorder


class FakeAudio:
    sample_rate = 16000
    block_ms = 80

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def chunks(self) -> Iterator[bytes]:
        yield from self._chunks


class VadTests(unittest.TestCase):
    def test_start_timeout_raises_when_no_speech_starts(self) -> None:
        recorder = EnergyVadRecorder(VadConfig(threshold=1000))
        silence = b"\x00\x00" * 1280

        with self.assertRaises(TimeoutError):
            recorder.record(
                FakeAudio([silence, silence]),
                start_timeout_seconds=0.16,
            )

    def test_start_timeout_allows_speech_that_has_started(self) -> None:
        recorder = EnergyVadRecorder(
            VadConfig(
                threshold=1000,
                min_speech_ms=160,
                silence_ms=80,
                pre_roll_ms=80,
            )
        )
        speech = (2000).to_bytes(2, "little", signed=True) * 1280
        silence = b"\x00\x00" * 1280

        utterance = recorder.record(
            FakeAudio([silence, speech, speech, silence]),
            start_timeout_seconds=0.16,
        )

        self.assertGreater(utterance.duration_ms, 0)


if __name__ == "__main__":
    unittest.main()
