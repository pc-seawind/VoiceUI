from __future__ import annotations

import io
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from voiceui.__main__ import _save_wake_debug, main
from voiceui.audio_dump import AudioDumpManager, configure_audio_dump
from voiceui.logs import log_event, reset_logging
from voiceui.models import AssistantConfig, DebugConfig, WakeEvent


class _FakeAudioInput:
    sample_rate = 16000
    block_ms = 80
    selected_channel = 0


class MainTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_audio_dump(None)
        reset_logging()

    def test_wake_test_keeps_listening_and_only_prints_wake_logs(self) -> None:
        config = AssistantConfig()
        detector = MagicMock()
        detector.wait.side_effect = [
            WakeEvent(engine="wekws_mha", confidence=0.998, label="hey_leela", pcm=b"1"),
            WakeEvent(engine="wekws_mha", confidence=0.999, label="hello_leela", pcm=b"2"),
            KeyboardInterrupt(),
        ]
        audio_dump = MagicMock()
        audio_dump.debug_log_path.return_value = None
        audio_dump.text_record_dir.return_value = None
        wake_paths = [Path("wake_01.wav"), Path("wake_02.wav")]

        def create_audio(*_args, **_kwargs):
            log_event("audio", "stream_opened", log_id="audio.stream_opened", device="test")
            return _FakeAudioInput()

        output = io.StringIO()
        with (
            patch("voiceui.__main__.load_dotenv"),
            patch("voiceui.__main__.load_config", return_value=config),
            patch("voiceui.__main__.AudioDumpManager", return_value=audio_dump),
            patch("voiceui.__main__.create_audio_input", side_effect=create_audio),
            patch("voiceui.__main__.create_wake_detector", return_value=detector),
            patch(
                "voiceui.__main__._save_wake_debug",
                side_effect=wake_paths,
            ) as save_wake_debug,
            redirect_stdout(output),
        ):
            result = main(["--config", "test.yaml", "--wake-test"])

        self.assertEqual(result, 130)
        self.assertEqual(detector.wait.call_count, 3)
        self.assertEqual(save_wake_debug.call_count, 2)
        self.assertTrue(config.debug.enabled)
        self.assertTrue(config.debug.save_audio)
        self.assertTrue(config.debug.voice_path_dump_enabled)
        stdout_text = output.getvalue()
        self.assertEqual(stdout_text.count("module=wake | event=detected"), 2)
        self.assertEqual(stdout_text.count("module=wake | event=test_waiting"), 3)
        self.assertEqual(stdout_text.count("module=wake | event=audio_saved"), 2)
        self.assertNotIn("module=audio", stdout_text)

    def test_save_wake_debug_writes_exact_wake_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AssistantConfig(
                debug=DebugConfig(enabled=True, output_dir=temp_dir, save_audio=True)
            )
            audio_dump = AudioDumpManager(config.debug)
            wake = WakeEvent(
                engine="wekws_mha",
                confidence=0.999,
                label="hello_leela",
                pcm=b"\x00\x00" * 32000,
                sample_rate=16000,
                duration_ms=2000,
            )

            with redirect_stdout(io.StringIO()):
                path = _save_wake_debug(
                    config,
                    wake,
                    wake_ms=2400,
                    audio_dump=audio_dump,
                )

            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.exists())
            self.assertTrue(path.name.startswith("wake_01_"))
            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16000)
                self.assertEqual(wav_file.getnframes(), 32000)


if __name__ == "__main__":
    unittest.main()
