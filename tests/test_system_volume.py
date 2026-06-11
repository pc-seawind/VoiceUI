from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from voiceui.system_volume import (
    _linux_wpctl_target,
    _percent_to_scalar,
    get_system_output_volume,
    resolve_output_device_name,
    set_system_output_volume,
)


class SystemVolumeTests(unittest.TestCase):
    def test_percent_to_scalar_accepts_percent_or_fraction(self) -> None:
        self.assertEqual(_percent_to_scalar(30), 0.3)
        self.assertEqual(_percent_to_scalar(0.3), 0.3)
        self.assertEqual(_percent_to_scalar(1), 0.01)
        self.assertEqual(_percent_to_scalar(-10), -0.1)

    def test_resolve_output_device_name_uses_sounddevice_for_index(self) -> None:
        fake_sounddevice = types.SimpleNamespace(
            query_devices=lambda _device, _kind: {"name": "Speaker"}
        )

        with patch.dict("sys.modules", {"sounddevice": fake_sounddevice}):
            name = resolve_output_device_name(20)

        self.assertEqual(name, "Speaker")

    def test_resolve_output_device_name_resolves_wasapi_display_name(self) -> None:
        class FakeSoundDevice:
            def query_hostapis(self):
                return [{"name": "Windows WASAPI"}]

            def query_devices(self, *args):
                devices = [
                    {
                        "name": "回音消除话筒 (reSpeaker XVF3800 4-Mic Array)",
                        "hostapi": 0,
                        "max_input_channels": 0,
                        "max_output_channels": 2,
                    }
                ]
                if args:
                    return devices[int(args[0])]
                return devices

        with patch.dict("sys.modules", {"sounddevice": FakeSoundDevice()}):
            name = resolve_output_device_name(
                "回音消除话筒 (reSpeaker XVF3800 4-Mic Array), "
                "Windows WASAPI (0 in, 2 out)"
            )

        self.assertEqual(name, "回音消除话筒 (reSpeaker XVF3800 4-Mic Array)")

    def test_set_system_output_volume_parses_powershell_json(self) -> None:
        completed = types.SimpleNamespace(
            returncode=0,
            stdout='{"device":"Speaker","before_percent":10,"after_percent":30}\n',
            stderr="",
        )

        with patch("voiceui.system_volume.sys.platform", "win32"):
            with patch("voiceui.system_volume.resolve_output_device_name", return_value="Speaker"):
                with patch("voiceui.system_volume.subprocess.run", return_value=completed) as run:
                    result = set_system_output_volume(
                        device=20,
                        volume_percent=30,
                        muted=False,
                    )

        self.assertEqual(result["device"], "Speaker")
        self.assertEqual(result["after_percent"], 30)
        command = run.call_args.args[0]
        self.assertIn("-EncodedCommand", command)

    def test_linux_get_system_output_volume_uses_wpctl(self) -> None:
        outputs = {
            ("get-volume", "@DEFAULT_AUDIO_SINK@"): types.SimpleNamespace(
                returncode=0, stdout="Volume: 0.80\n", stderr=""
            )
        }

        def fake_run(command, **_kwargs):
            return outputs[tuple(command[1:])]

        with patch("voiceui.system_volume.sys.platform", "linux"):
            with patch("voiceui.system_volume.subprocess.run", side_effect=fake_run):
                result = get_system_output_volume()

        self.assertEqual(result["after_percent"], 80.0)
        self.assertFalse(result["after_muted"])

    def test_linux_set_system_output_volume_uses_wpctl_relative(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            if command[1] == "get-volume":
                return types.SimpleNamespace(returncode=0, stdout="Volume: 0.80\n", stderr="")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("voiceui.system_volume.sys.platform", "linux"):
            with patch("voiceui.system_volume.subprocess.run", side_effect=fake_run):
                result = set_system_output_volume(relative_percent=-20)

        self.assertIn(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "0.6000"], calls)
        self.assertEqual(result["before_percent"], 80.0)
        self.assertEqual(result["after_percent"], 80.0)

    def test_linux_wpctl_target_matches_status_description(self) -> None:
        status = """Audio
 ├─ Sinks:
 │  *   57. reSpeaker XVF3800 4-Mic Array Analog Stereo [vol: 0.80]
"""
        completed = types.SimpleNamespace(returncode=0, stdout=status, stderr="")

        with patch("voiceui.system_volume.subprocess.run", return_value=completed):
            target = _linux_wpctl_target("reSpeaker XVF3800 4-Mic Array")

        self.assertEqual(target, "57")



if __name__ == "__main__":
    unittest.main()
