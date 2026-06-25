from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from voiceui.audio_dump import AudioDumpManager, _SegmentedWavDumpWriter, dump_filename
from voiceui.logs import reset_logging
from voiceui.models import DebugConfig


class AudioDumpTests(unittest.TestCase):
    def test_dump_filename_contains_relative_start_and_end_ms(self) -> None:
        self.assertEqual(
            dump_filename("wake", 12, 345, turn_index=1),
            "wake_01_00.00.00.012_00.00.00.345.wav",
        )
        self.assertEqual(
            dump_filename("system_input", 0, 30_000),
            "system_input_00.00.00.000_00.00.30.000.wav",
        )

    def test_voice_path_dump_uses_time_window_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AudioDumpManager(DebugConfig(enabled=True, output_dir=temp_dir))

            dump = manager.write_voice_path_dump(
                temp_dir,
                "utterance",
                b"\x00\x00" * 160,
                sample_rate=16000,
                start_ms=100,
                end_ms=110,
            )

            self.assertIsNotNone(dump)
            assert dump is not None
            self.assertEqual(dump.path.name, "utterance_01_00.00.00.100_00.00.00.110.wav")
            with wave.open(str(dump.path), "rb") as wav:
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getnchannels(), 1)

    def test_voice_path_default_output_uses_flat_session_audio_dump_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AudioDumpManager(DebugConfig(enabled=True, output_dir=temp_dir))

            dump = manager.write_voice_path_dump(
                None,
                "tts_output",
                b"\x00\x00" * 160,
                sample_rate=16000,
                start_ms=100,
                end_ms=110,
            )

            self.assertIsNotNone(dump)
            assert dump is not None
            self.assertEqual(dump.path.parent.name, "audio_dumps")
            self.assertEqual(dump.path.parent.parent, manager.debug_session_dir())
            self.assertEqual(
                list(dump.path.parent.glob("tts_output_01_*_*.wav")),
                [dump.path],
            )
            self.assertFalse((dump.path.parent / "voice_path").exists())
            self.assertFalse((dump.path.parent / "system_input").exists())


    def test_turn_scoped_sessions_are_created_per_turn_not_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AudioDumpManager(
                DebugConfig(enabled=True, output_dir=temp_dir, session_scope="turn")
            )

            self.assertEqual(manager.debug_log_path(), Path(temp_dir) / "debug.log")
            self.assertIsNone(manager.debug_session_dir())
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

            first_turn = manager.begin_turn()
            first_session = manager.debug_session_dir()
            self.assertIsNotNone(first_session)
            assert first_session is not None
            self.assertEqual(manager.debug_log_path(), first_session / "debug.log")
            manager.end_turn(first_turn)
            self.assertEqual(manager.debug_log_path(), Path(temp_dir) / "debug.log")

            second_turn = manager.begin_turn()
            second_session = manager.debug_session_dir()
            self.assertIsNotNone(second_session)
            assert second_session is not None
            manager.end_turn(second_turn)

            self.assertNotEqual(first_session, second_session)
            self.assertEqual(
                sorted(path.name for path in Path(temp_dir).iterdir() if path.is_dir()),
                sorted([first_session.name, second_session.name, "text_records"]),
            )
        reset_logging()

    def test_segmented_writer_splits_system_input_dump(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AudioDumpManager(DebugConfig(enabled=True, output_dir=temp_dir))
            writer = _SegmentedWavDumpWriter(
                manager=manager,
                output_dir=Path(temp_dir),
                kind="system_input",
                sample_rate=1000,
                channels=1,
                segment_ms=30_000,
            )

            writer.write(b"\x00\x00" * 30_000)
            writer.write(b"\x01\x00" * 5_000)
            writer.close()

            dumps = sorted(Path(temp_dir).glob("system_input_*.wav"))
            self.assertEqual(len(dumps), 2)
            self.assertEqual(
                [dump.name for dump in dumps],
                [
                    "system_input_00.00.00.000_00.00.30.000.wav",
                    "system_input_00.00.30.000_00.00.35.000.wav",
                ],
            )
            with wave.open(str(dumps[0]), "rb") as wav:
                self.assertEqual(wav.getnframes(), 30_000)
            with wave.open(str(dumps[1]), "rb") as wav:
                self.assertEqual(wav.getnframes(), 5_000)

    def test_segmented_writer_splits_large_single_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AudioDumpManager(DebugConfig(enabled=True, output_dir=temp_dir))
            writer = _SegmentedWavDumpWriter(
                manager=manager,
                output_dir=Path(temp_dir),
                kind="system_input",
                sample_rate=1000,
                channels=1,
                segment_ms=30_000,
            )

            writer.write(b"\x00\x00" * 65_000)
            writer.close()

            dumps = sorted(Path(temp_dir).glob("system_input_*.wav"))
            self.assertEqual(len(dumps), 3)
            self.assertEqual(
                [dump.name for dump in dumps],
                [
                    "system_input_00.00.00.000_00.00.30.000.wav",
                    "system_input_00.00.30.000_00.01.00.000.wav",
                    "system_input_00.01.00.000_00.01.05.000.wav",
                ],
            )
            frame_counts = []
            for dump in dumps:
                with wave.open(str(dump), "rb") as wav:
                    frame_counts.append(wav.getnframes())
            self.assertEqual(frame_counts, [30_000, 30_000, 5_000])


if __name__ == "__main__":
    unittest.main()
