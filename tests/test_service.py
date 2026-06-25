from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voiceui.config import AUTO_CONFIG
from voiceui.models import AssistantConfig, DebugConfig, InputConfig
from voiceui.service import _DEFAULT_SERVICE_CONFIG, _prepare_service_config


class ServiceTests(unittest.TestCase):

    def test_service_config_default_is_auto(self) -> None:
        self.assertEqual(_DEFAULT_SERVICE_CONFIG, AUTO_CONFIG)

    def test_service_config_defaults_to_audio_mode_without_audio_dumps(self) -> None:
        config = AssistantConfig(
            input=InputConfig(mode="text"),
            debug=DebugConfig(
                enabled=False,
                save_audio=True,
                system_input_dump_enabled=True,
                voice_path_dump_enabled=True,
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            _prepare_service_config(config, output_dir=Path(temp_dir), audio_dump=False)

            self.assertEqual(config.input.mode, "audio")
            self.assertTrue(config.debug.enabled)
            self.assertEqual(config.debug.output_dir, temp_dir)
            self.assertFalse(config.debug.save_audio)
            self.assertTrue(config.debug.save_metadata)
            self.assertEqual(config.debug.session_scope, "run")
            self.assertFalse(config.debug.system_input_dump_enabled)
            self.assertFalse(config.debug.voice_path_dump_enabled)

    def test_service_config_can_enable_audio_dumps_explicitly(self) -> None:
        config = AssistantConfig(debug=DebugConfig(enabled=False, save_audio=False))

        _prepare_service_config(config, output_dir=None, audio_dump=True)

        self.assertTrue(config.debug.enabled)
        self.assertTrue(config.debug.save_audio)
        self.assertTrue(config.debug.system_input_dump_enabled)
        self.assertTrue(config.debug.voice_path_dump_enabled)

    def test_service_config_can_enable_voice_path_dumps_without_system_input_dump(
        self,
    ) -> None:
        config = AssistantConfig(debug=DebugConfig(enabled=False, save_audio=False))

        _prepare_service_config(
            config,
            output_dir=None,
            audio_dump=False,
            voice_path_dump=True,
        )

        self.assertTrue(config.debug.enabled)
        self.assertTrue(config.debug.save_audio)
        self.assertFalse(config.debug.system_input_dump_enabled)
        self.assertTrue(config.debug.voice_path_dump_enabled)

    def test_service_config_can_enable_system_input_dump_without_voice_path_dump(
        self,
    ) -> None:
        config = AssistantConfig(debug=DebugConfig(enabled=False, save_audio=False))

        _prepare_service_config(
            config,
            output_dir=None,
            audio_dump=False,
            system_input_dump=True,
        )

        self.assertTrue(config.debug.enabled)
        self.assertTrue(config.debug.save_audio)
        self.assertTrue(config.debug.system_input_dump_enabled)
        self.assertFalse(config.debug.voice_path_dump_enabled)


if __name__ == "__main__":
    unittest.main()
