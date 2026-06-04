from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voiceui.audio import (
    RawAudioRecording,
    apply_pcm16_gain_db,
    pcm16_rms,
    read_pcm16_wav,
    resolve_sounddevice_device,
    select_pcm16_channel,
    write_pcm16_wav,
)


class AudioTests(unittest.TestCase):
    def test_resolve_sounddevice_device_uses_full_wasapi_input_display_name(self) -> None:
        sd = _FakeSoundDevice()

        device = resolve_sounddevice_device(
            sd,
            "回音消除话筒 (reSpeaker XVF3800 4-Mic Array), "
            "Windows WASAPI (2 in, 0 out)",
            kind="input",
        )

        self.assertEqual(device, 1)

    def test_resolve_sounddevice_device_uses_full_wasapi_output_display_name(self) -> None:
        sd = _FakeSoundDevice()

        device = resolve_sounddevice_device(
            sd,
            "回音消除话筒 (reSpeaker XVF3800 4-Mic Array), "
            "Windows WASAPI (0 in, 2 out)",
            kind="output",
        )

        self.assertEqual(device, 2)

    def test_resolve_sounddevice_device_distinguishes_second_xvf3800(self) -> None:
        sd = _FakeSoundDevice()

        device = resolve_sounddevice_device(
            sd,
            "回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array), "
            "Windows WASAPI (2 in, 0 out)",
            kind="input",
        )

        self.assertEqual(device, 3)

    def test_resolve_sounddevice_device_prefers_wasapi_for_bare_name(self) -> None:
        sd = _FakeSoundDevice()

        device = resolve_sounddevice_device(
            sd,
            "回音消除话筒 (reSpeaker XVF3800 4-Mic Array)",
            kind="input",
        )

        self.assertEqual(device, 1)

    def test_resolve_sounddevice_device_rejects_ambiguous_substring(self) -> None:
        sd = _FakeSoundDevice()

        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            resolve_sounddevice_device(
                sd,
                "reSpeaker XVF3800 4-Mic Array",
                kind="input",
            )

    def test_resolve_sounddevice_device_keeps_numeric_index_fallback(self) -> None:
        sd = _FakeSoundDevice()

        self.assertEqual(resolve_sounddevice_device(sd, "24", kind="input"), 24)
        self.assertEqual(resolve_sounddevice_device(sd, 22, kind="output"), 22)

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

    def test_apply_pcm16_gain_db_amplifies_and_clips(self) -> None:
        samples = [1000, -1000, 4000, -4000]
        pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)

        amplified = apply_pcm16_gain_db(pcm, 20.0)

        self.assertEqual(_decode_pcm16(amplified), [10000, -10000, 32767, -32768])

    def test_apply_pcm16_gain_db_zero_gain_returns_same_bytes(self) -> None:
        pcm = b"\x01\x00\x02\x00"

        self.assertIs(apply_pcm16_gain_db(pcm, 0.0), pcm)

    def test_write_and_read_pcm16_wav(self) -> None:
        samples = [100, -100]
        pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            write_pcm16_wav(path, pcm, sample_rate=16000)
            loaded_pcm, sample_rate = read_pcm16_wav(path)

        self.assertEqual(loaded_pcm, pcm)
        self.assertEqual(sample_rate, 16000)

    def test_write_and_read_stereo_pcm16_wav_selected_channel(self) -> None:
        samples = [1, 10, 2, 20]
        pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            write_pcm16_wav(path, pcm, sample_rate=16000, channels=2)
            loaded_pcm, sample_rate = read_pcm16_wav(path, selected_channel=1)

        self.assertEqual(_decode_pcm16(loaded_pcm), [10, 20])
        self.assertEqual(sample_rate, 16000)

    def test_raw_audio_recording_keeps_interleaved_audio(self) -> None:
        recording = RawAudioRecording(sample_rate=16000, channels=2)

        recording.append(b"\x01\x00\x02\x00")
        recording.append(b"\x03\x00\x04\x00")

        self.assertEqual(recording.pcm(), b"\x01\x00\x02\x00\x03\x00\x04\x00")
        self.assertEqual(recording.duration_ms(), 0)


def _decode_pcm16(pcm: bytes) -> list[int]:
    return [
        int.from_bytes(pcm[index : index + 2], "little", signed=True)
        for index in range(0, len(pcm), 2)
    ]


class _FakeSoundDevice:
    def query_hostapis(self):
        return [
            {"name": "MME"},
            {"name": "Windows WASAPI"},
        ]

    def query_devices(self):
        return [
            {
                "name": "回音消除话筒 (reSpeaker XVF3800 4-Mic Array)",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
            {
                "name": "回音消除话筒 (reSpeaker XVF3800 4-Mic Array)",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
            {
                "name": "回音消除话筒 (reSpeaker XVF3800 4-Mic Array)",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {
                "name": "回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array)",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
            {
                "name": "回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array)",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
        ]


if __name__ == "__main__":
    unittest.main()
