from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voiceui.models import XiaomiMiotConfig
from voiceui.xiaomi_miot import (
    XiaomiMiotClient,
    XiaomiMiotToken,
    coerce_miot_action_inputs,
    coerce_miot_value,
    load_xiaomi_miot_token,
    miot_values_equal,
    parse_miot_spec_lite,
    save_xiaomi_miot_token,
    split_miot_iid,
)


class XiaomiMiotTests(unittest.TestCase):
    def test_split_miot_iid_accepts_prop_and_action(self) -> None:
        self.assertEqual(split_miot_iid("prop.0.2.1"), ("prop", 2, 1))
        self.assertEqual(split_miot_iid("action.0.3.4"), ("action", 3, 4))

        with self.assertRaises(RuntimeError):
            split_miot_iid("prop.2.1")

    def test_coerce_miot_values_by_spec_format(self) -> None:
        self.assertEqual(coerce_miot_value("1", {"format": "uint8"}), 1)
        self.assertEqual(coerce_miot_value("26.5", {"format": "float"}), 26.5)
        self.assertEqual(coerce_miot_value(123, {"format": "string"}), "123")
        self.assertTrue(coerce_miot_value("yes", {"format": "bool"}))
        self.assertFalse(coerce_miot_value("0", {"format": "bool"}))

    def test_coerce_action_inputs_accepts_list_or_json_array(self) -> None:
        self.assertEqual(coerce_miot_action_inputs([1, "x"]), [1, "x"])
        self.assertEqual(coerce_miot_action_inputs('[1,"x"]'), [1, "x"])
        self.assertEqual(coerce_miot_action_inputs("raw"), ["raw"])

        with self.assertRaises(RuntimeError):
            coerce_miot_action_inputs(1)

    def test_miot_values_equal_compares_bool_and_number_readbacks(self) -> None:
        self.assertTrue(miot_values_equal(True, 1))
        self.assertTrue(miot_values_equal(False, "0"))
        self.assertTrue(miot_values_equal(26, "26"))
        self.assertFalse(miot_values_equal(True, 0))

    def test_parse_miot_spec_lite_builds_property_and_action_iids(self) -> None:
        instance = {
            "type": "urn:miot-spec-v2:device:light:0000A001:test:1",
            "description": "Light",
            "services": [
                {
                    "iid": 1,
                    "type": "urn:miot-spec-v2:service:device-information:00007801:miot:1",
                    "description": "Device Information",
                    "properties": [],
                },
                {
                    "iid": 2,
                    "type": "urn:miot-spec-v2:service:light:00007802:test:1",
                    "description": "Light",
                    "properties": [
                        {
                            "iid": 1,
                            "type": "urn:miot-spec-v2:property:on:00000006:test:1",
                            "description": "Power",
                            "format": "bool",
                            "access": ["read", "write", "notify"],
                        },
                        {
                            "iid": 2,
                            "type": "urn:miot-spec-v2:property:brightness:0000000D:test:1",
                            "description": "Brightness",
                            "format": "uint8",
                            "access": ["read", "write"],
                            "value-range": [1, 100, 1],
                        },
                    ],
                    "actions": [
                        {
                            "iid": 1,
                            "type": "urn:miot-spec-v2:action:toggle:00002801:test:1",
                            "description": "Toggle",
                            "in": [1],
                        }
                    ],
                },
            ],
        }

        spec = parse_miot_spec_lite(instance, instance["type"])

        self.assertEqual(spec["description"], "Light")
        self.assertNotIn("prop.0.1.1", spec["items"])
        self.assertTrue(spec["items"]["prop.0.2.1"]["writeable"])
        self.assertEqual(spec["items"]["prop.0.2.2"]["value_range"]["max"], 100)
        self.assertEqual(spec["items"]["action.0.2.1"]["inputs"][0]["format"], "bool")

    def test_token_loads_from_json_env_without_printing_secret(self) -> None:
        token_json = json.dumps(
            {
                "access_token": "access-secret",
                "refresh_token": "refresh-secret",
                "expires_ts": 123,
            }
        )

        with patch.dict(os.environ, {"XIAOMI_MIOT_TOKEN_JSON": token_json}, clear=True):
            token = load_xiaomi_miot_token(XiaomiMiotConfig())

        self.assertEqual(token.access_token, "access-secret")
        self.assertEqual(token.refresh_token, "refresh-secret")
        self.assertEqual(token.expires_ts, 123)

    def test_token_round_trips_to_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            token_path = Path(temp_dir) / "miot_token.json"
            config = XiaomiMiotConfig(token_file=str(token_path))
            save_xiaomi_miot_token(
                config,
                XiaomiMiotToken(
                    access_token="access-secret",
                    refresh_token="refresh-secret",
                    expires_ts=456,
                ),
            )

            with patch.dict(os.environ, {}, clear=True):
                token = load_xiaomi_miot_token(config)

        self.assertEqual(token.access_token, "access-secret")
        self.assertEqual(token.refresh_token, "refresh-secret")
        self.assertEqual(token.expires_ts, 456)

    def test_control_device_fuzzy_matches_unique_room_light(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = XiaomiMiotClient(
                XiaomiMiotConfig(
                    token_file=str(Path(temp_dir) / "token.json"),
                    uuid_file=str(Path(temp_dir) / "uuid"),
                    cache_dir=str(Path(temp_dir) / "cache"),
                    control_verify_delay_seconds=0,
                ),
                token=XiaomiMiotToken(access_token="access-secret"),
            )
        devices = {
            "did-light": {
                "did": "did-light",
                "name": "书房吸顶灯",
                "room_name": "书房",
                "home_name": "家",
                "device_class": "light",
                "online": True,
                "model": "yeelink.light.ceiling",
                "urn": "urn:test",
            },
            "did-kitchen": {
                "did": "did-kitchen",
                "name": "厨房吸顶灯",
                "room_name": "厨房",
                "home_name": "家",
                "device_class": "light",
                "online": True,
                "model": "yeelink.light.ceiling",
                "urn": "urn:test",
            },
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "on",
                    "description": "Light - Power",
                    "format": "bool",
                    "readable": True,
                    "writeable": True,
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                    send_ctrl.return_value = {
                        "status": "verified",
                        "did": "did-light",
                        "iid": "prop.0.2.1",
                        "readback_value": True,
                    }
                    result = client.control_device(request="帮我把书房的灯灯打开")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["device"]["name"], "书房吸顶灯")
        send_ctrl.assert_called_once_with("did-light", "prop.0.2.1", True, verify=True)

    def test_control_device_returns_ambiguous_for_multiple_room_lights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = XiaomiMiotClient(
                XiaomiMiotConfig(
                    token_file=str(Path(temp_dir) / "token.json"),
                    uuid_file=str(Path(temp_dir) / "uuid"),
                    cache_dir=str(Path(temp_dir) / "cache"),
                ),
                token=XiaomiMiotToken(access_token="access-secret"),
            )
        devices = {
            "did-1": {
                "did": "did-1",
                "name": "书房吸顶灯",
                "room_name": "书房",
                "device_class": "light",
                "online": True,
            },
            "did-2": {
                "did": "did-2",
                "name": "书房台灯",
                "room_name": "书房",
                "device_class": "light",
                "online": True,
            },
        }

        with patch.object(client, "get_devices", return_value=devices):
            result = client.control_device(request="帮我把书房的灯打开")

        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(len(result["candidates"]), 2)

    def test_send_ctrl_rpc_verifies_readable_property(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = XiaomiMiotClient(
                XiaomiMiotConfig(
                    token_file=str(Path(temp_dir) / "token.json"),
                    uuid_file=str(Path(temp_dir) / "uuid"),
                    cache_dir=str(Path(temp_dir) / "cache"),
                    control_verify_delay_seconds=0,
                ),
                token=XiaomiMiotToken(access_token="access-secret"),
            )
        spec_item = {
            "iid": "prop.0.2.1",
            "kind": "property",
            "format": "bool",
            "readable": True,
            "writeable": True,
        }

        with patch.object(client, "_find_spec_item", return_value=spec_item):
            with patch.object(client, "_api_post") as api_post:
                api_post.side_effect = [
                    {"code": 0, "result": [{"code": 0}]},
                    {"code": 0, "result": [{"value": False}]},
                    {"code": 0, "result": [{"value": False}]},
                ]
                with self.assertRaisesRegex(RuntimeError, "verification failed"):
                    client.send_ctrl_rpc("did-light", "prop.0.2.1", True)


if __name__ == "__main__":
    unittest.main()
