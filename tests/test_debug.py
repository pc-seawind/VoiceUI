from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from voiceui.debug import DebugRecorder, TurnDebugData
from voiceui.models import DebugConfig, Utterance, WakeEvent


class DebugTests(unittest.TestCase):
    def test_debug_recorder_writes_metadata_and_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DebugRecorder(
                DebugConfig(
                    enabled=True,
                    output_dir=temp_dir,
                    save_audio=True,
                    save_metadata=True,
                )
            )
            utterance = Utterance(pcm=b"\x00\x00" * 160, sample_rate=16000, duration_ms=10)
            wake = WakeEvent(
                engine="openwakeword",
                confidence=0.7,
                label="alexa",
                pcm=b"\x01\x00" * 320,
                sample_rate=16000,
                duration_ms=20,
            )
            data = TurnDebugData(
                node_id="node",
                room="room",
                timings_ms={"stt": 10},
                transcript="hello",
                reply="hi",
            )

            turn_dir = recorder.save_turn(data, utterance, wake_audio=wake)

            self.assertIsNotNone(turn_dir)
            assert turn_dir is not None
            audio_dir = turn_dir / "audio_dumps"
            self.assertEqual(len(list(audio_dir.glob("wake_01_*.wav"))), 1)
            self.assertEqual(len(list(audio_dir.glob("utterance_01_*.wav"))), 1)
            metadata_file = json.loads(
                (turn_dir / "metadata.json").read_text(encoding="utf-8")
            )
            metadata = metadata_file["turns"][0]
            self.assertEqual(len(metadata_file["turns"]), 1)
            self.assertEqual(metadata_file["barge_in"], [])
            self.assertEqual(metadata["turn"], 1)
            self.assertEqual(metadata["transcript"], "hello")
            self.assertEqual(metadata["reply"], "hi")
            self.assertEqual(metadata["wake"]["duration_ms"], 20)
            self.assertEqual(metadata["wake"]["sample_rate"], 16000)
            self.assertIn("dump_path", metadata["wake"])
            self.assertIn("dump_start_ms", metadata["wake"])
            self.assertIn("dump_end_ms", metadata["wake"])
            self.assertIn("dump_path", metadata["utterance"])
            self.assertIn("dump_start_ms", metadata["utterance"])
            self.assertIn("dump_end_ms", metadata["utterance"])

    def test_disabled_debug_recorder_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DebugRecorder(DebugConfig(enabled=False, output_dir=temp_dir))

            turn_dir = recorder.save_turn(TurnDebugData(node_id="node", room="room"))

            self.assertIsNone(turn_dir)
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_debug_recorder_appends_turns_to_one_metadata_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DebugRecorder(
                DebugConfig(
                    enabled=True,
                    output_dir=temp_dir,
                    save_audio=True,
                    save_metadata=True,
                )
            )

            session_dir = recorder.save_turn(
                TurnDebugData(node_id="node", room="room", transcript="one"),
                Utterance(pcm=b"\x00\x00" * 160, sample_rate=16000, duration_ms=10),
            )
            recorder.save_turn(
                TurnDebugData(node_id="node", room="room", transcript="two"),
                Utterance(pcm=b"\x00\x00" * 160, sample_rate=16000, duration_ms=10),
            )

            assert session_dir is not None
            metadata_files = list(session_dir.glob("*metadata*.json"))
            self.assertEqual([path.name for path in metadata_files], ["metadata.json"])
            metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [(turn["turn"], turn["transcript"]) for turn in metadata["turns"]],
                [(1, "one"), (2, "two")],
            )
            audio_files = sorted((session_dir / "audio_dumps").glob("utterance_*.wav"))
            self.assertEqual(audio_files[0].name.split("_")[1], "01")
            self.assertEqual(audio_files[1].name.split("_")[1], "02")

    def test_debug_recorder_writes_barge_in_monitor_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DebugRecorder(
                DebugConfig(
                    enabled=True,
                    output_dir=temp_dir,
                    save_audio=True,
                    save_metadata=True,
                )
            )

            debug_dir = recorder.save_barge_in_monitor(
                mode="stream",
                result="no_speech",
                pcm=b"\x00\x00" * 160,
                sample_rate=16000,
                duration_ms=10,
                metadata={"vad_engine": "silero"},
                extra_wavs={
                    "raw.wav": (b"\x01\x00\x02\x00" * 160, 16000, 2),
                    "raw_ch0.wav": (b"\x01\x00" * 160, 16000, 1),
                },
            )

            self.assertIsNotNone(debug_dir)
            assert debug_dir is not None
            audio_dir = debug_dir / "audio_dumps"
            self.assertEqual(len(list(audio_dir.glob("barge_in_monitor_01_*.wav"))), 1)
            raw_dumps = [
                path
                for path in audio_dir.glob("raw_01_*.wav")
                if not path.name.startswith("raw_ch")
            ]
            self.assertEqual(len(raw_dumps), 1)
            self.assertEqual(len(list(audio_dir.glob("raw_ch0_01_*.wav"))), 1)
            metadata_file = json.loads(
                (debug_dir / "metadata.json").read_text(encoding="utf-8")
            )
            metadata = metadata_file["barge_in"][0]
            self.assertEqual(metadata_file["turns"], [])
            self.assertEqual(len(metadata_file["barge_in"]), 1)
            self.assertEqual(metadata["turn"], 1)
            self.assertEqual(metadata["barge_in_index"], 1)
            self.assertEqual(metadata["mode"], "stream")
            self.assertEqual(metadata["result"], "no_speech")
            self.assertEqual(metadata["vad_engine"], "silero")
            self.assertIn("dump_path", metadata)
            self.assertIn("dump_start_ms", metadata)
            self.assertIn("dump_end_ms", metadata)
            self.assertIn("extra_wav_paths", metadata)

    def test_standalone_barge_in_monitor_increments_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DebugRecorder(
                DebugConfig(
                    enabled=True,
                    output_dir=temp_dir,
                    save_audio=True,
                    save_metadata=True,
                )
            )

            debug_dir = recorder.save_barge_in_monitor(
                mode="stream",
                result="no_speech",
                pcm=b"\x00\x00" * 160,
                sample_rate=16000,
                duration_ms=10,
            )
            recorder.save_barge_in_monitor(
                mode="stream",
                result="no_speech",
                pcm=b"\x00\x00" * 160,
                sample_rate=16000,
                duration_ms=10,
            )

            assert debug_dir is not None
            metadata_file = json.loads(
                (debug_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [
                    (entry["turn"], entry["barge_in_index"])
                    for entry in metadata_file["barge_in"]
                ],
                [(1, 1), (2, 2)],
            )
            audio_names = [
                path.name.split("_")[3]
                for path in sorted((debug_dir / "audio_dumps").glob("barge_in_monitor_*.wav"))
            ]
            self.assertEqual(audio_names, ["01", "02"])


if __name__ == "__main__":
    unittest.main()
