from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from voiceui.audio import pcm16_rms, read_pcm16_wav, select_pcm16_channel, write_pcm16_wav


class AudioTests(unittest.TestCase):
    def test_select_pcm16_channel_extracts_interleaved_samples(self) -> None:
        samples = [1, 10, 2, 20, 3, 30]
        pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)

        left = select_pcm16_channel(pcm, channels=2, selected_channel=0)
        right = select_pcm16_channel(pcm, channels=2, selected_channel=1)

        self.assertEqual(_decode_pcm16(left), [1, 2, 3])
        self.assertEqual(_decode_pcm16(right), [10, 20, 30])

    def test_select_pcm16_channel_rejects_invalid_channel(self) -> None:
        with self.assertRaises(ValueError):
            select_pcm16_channel(b"\x00\x00\x00\x00", channels=2, selected_channel=2)

    def test_pcm16_rms(self) -> None:
        samples = [3, 4]
        pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
        self.assertAlmostEqual(pcm16_rms(pcm), 3.5355, places=3)

    def test_write_and_read_pcm16_wav(self) -> None:
        samples = [100, -100]
        pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            write_pcm16_wav(path, pcm, sample_rate=16000)
            loaded_pcm, sample_rate = read_pcm16_wav(path)

        self.assertEqual(loaded_pcm, pcm)
        self.assertEqual(sample_rate, 16000)


def _decode_pcm16(pcm: bytes) -> list[int]:
    return [
        int.from_bytes(pcm[index : index + 2], "little", signed=True)
        for index in range(0, len(pcm), 2)
    ]


if __name__ == "__main__":
    unittest.main()
