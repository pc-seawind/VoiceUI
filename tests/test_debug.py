from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from voiceui.debug import DebugRecorder, TurnDebugData
from voiceui.models import DebugConfig, Utterance


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
            data = TurnDebugData(
                node_id="node",
                room="room",
                timings_ms={"stt": 10},
                transcript="hello",
                reply="hi",
            )

            turn_dir = recorder.save_turn(data, utterance)

            self.assertIsNotNone(turn_dir)
            assert turn_dir is not None
            self.assertTrue((turn_dir / "utterance.wav").exists())
            metadata = json.loads((turn_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["transcript"], "hello")
            self.assertEqual(metadata["reply"], "hi")
            self.assertIn("wav_path", metadata["utterance"])

    def test_disabled_debug_recorder_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DebugRecorder(DebugConfig(enabled=False, output_dir=temp_dir))

            turn_dir = recorder.save_turn(TurnDebugData(node_id="node", room="room"))

            self.assertIsNone(turn_dir)
            self.assertEqual(list(Path(temp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
