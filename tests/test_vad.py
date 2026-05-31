from __future__ import annotations

import sys
import types
import unittest
from collections.abc import Iterator
from unittest.mock import patch

from voiceui.models import VadConfig
from voiceui.vad import EnergyVadRecorder, SileroVadRecorder, WebRtcVadRecorder, _pcm16_frames


class FakeAudio:
    sample_rate = 16000
    block_ms = 80

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def chunks(self) -> Iterator[bytes]:
        yield from self._chunks


class FakeWebRtcVad:
    def __init__(self, mode: int):
        self.mode = mode

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return any(frame)


class FakeSileroModel:
    def __init__(self):
        self.reset_count = 0

    def reset_states(self) -> None:
        self.reset_count += 1

    def __call__(self, samples, sample_rate: int):
        return types.SimpleNamespace(item=lambda: 0.9 if any(samples) else 0.0)


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def set_num_threads(_threads: int) -> None:
        return None

    @staticmethod
    def tensor(samples, dtype=None):
        return samples


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

    def test_pcm16_frames_splits_larger_audio_chunks(self) -> None:
        first_frame = b"\x01\x00" * 320
        second_frame = b"\x02\x00" * 320
        frames = list(_pcm16_frames(FakeAudio([first_frame + second_frame]), frame_ms=20))

        self.assertEqual(frames, [first_frame, second_frame])

    def test_webrtc_vad_records_until_trailing_silence(self) -> None:
        recorder = WebRtcVadRecorder(
            VadConfig(
                engine="webrtc",
                min_speech_ms=40,
                silence_ms=40,
                pre_roll_ms=20,
                frame_ms=20,
            )
        )
        silence = b"\x00\x00" * 320
        speech = (2000).to_bytes(2, "little", signed=True) * 320
        fake_webrtcvad = types.SimpleNamespace(Vad=FakeWebRtcVad)

        with patch.dict(sys.modules, {"webrtcvad": fake_webrtcvad}):
            utterance = recorder.record(
                FakeAudio([silence + speech + speech + speech + silence + silence])
            )

        self.assertEqual(utterance.duration_ms, 80)
        self.assertEqual(utterance.sample_rate, 16000)

    def test_silero_vad_records_until_trailing_silence(self) -> None:
        recorder = SileroVadRecorder(
            VadConfig(
                engine="silero",
                threshold=0.6,
                min_speech_ms=64,
                silence_ms=64,
                pre_roll_ms=32,
            )
        )
        silence = b"\x00\x00" * 512
        speech = (2000).to_bytes(2, "little", signed=True) * 512
        fake_silero = types.SimpleNamespace(load_silero_vad=FakeSileroModel)

        with patch.dict(sys.modules, {"silero_vad": fake_silero, "torch": FakeTorch}):
            utterance = recorder.record(
                FakeAudio([silence + speech + speech + speech + silence + silence])
            )

        self.assertEqual(utterance.duration_ms, 128)
        self.assertEqual(utterance.sample_rate, 16000)


if __name__ == "__main__":
    unittest.main()
