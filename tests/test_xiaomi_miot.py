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

    def test_read_device_property_selects_air_quality_property(self) -> None:
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
            "did-purifier": {
                "did": "did-purifier",
                "name": "Living Room Air Purifier",
                "room_name": "Living Room",
                "home_name": "Home",
                "device_class": "airpurifier",
                "online": True,
                "model": "zhimi.airpurifier.test",
                "urn": "urn:test",
            }
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "pm2.5-density",
                    "description": "Air Quality - PM2.5 Density",
                    "service": "Air Quality",
                    "format": "uint16",
                    "readable": True,
                    "writeable": False,
                    "unit": "ug/m3",
                },
                "prop.0.2.2": {
                    "iid": "prop.0.2.2",
                    "kind": "property",
                    "name": "temperature",
                    "description": "Temperature",
                    "service": "Environment",
                    "format": "float",
                    "readable": True,
                    "writeable": False,
                    "unit": "C",
                },
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc") as send_get:
                    send_get.return_value = {
                        "did": "did-purifier",
                        "iid": "prop.0.2.1",
                        "value": 12,
                    }
                    result = client.read_device_property(
                        request="check home air purifier air quality"
                    )

        self.assertEqual(result["status"], "property_read")
        self.assertEqual(result["iid"], "prop.0.2.1")
        self.assertEqual(result["value"], 12)
        self.assertIn("Living Room Air Purifier", result["direct_response"])
        send_get.assert_called_once_with("did-purifier", "prop.0.2.1")

    def test_read_device_property_summarizes_group_power_state(self) -> None:
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
            "did-study": {
                "did": "did-study",
                "name": "书房空调",
                "room_name": "书房",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
            "did-living": {
                "did": "did-living",
                "name": "客厅空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "on",
                    "description": "Air Conditioner - Power",
                    "service": "Air Conditioner",
                    "format": "bool",
                    "readable": True,
                    "writeable": True,
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc") as send_get:
                    send_get.side_effect = [
                        {"did": "did-study", "iid": "prop.0.2.1", "value": True},
                        {"did": "did-living", "iid": "prop.0.2.1", "value": False},
                    ]
                    result = client.read_device_property(request="哪个空调开着")

        self.assertEqual(result["status"], "property_read_group")
        self.assertEqual(len(result["readings"]), 2)
        self.assertIn("书房空调", result["direct_response"])

    def test_control_device_rejects_scheduled_iot_without_executing(self) -> None:
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
            "did-ac": {
                "did": "did-ac",
                "name": "客厅空调",
                "room_name": "客厅",
                "home_name": "家",
                "device_class": "aircondition",
                "online": True,
                "model": "xiaomi.aircondition.test",
                "urn": "urn:test",
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec") as get_spec:
                with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                    result = client.control_device(request="十分钟后关闭空调")

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["decision"], "unsupported")
        self.assertEqual(result["operation"], "schedule")
        self.assertIn("不能定时控制", result["direct_response"])
        get_spec.assert_not_called()
        send_ctrl.assert_not_called()

    def test_control_device_rejects_scene_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = XiaomiMiotClient(
                XiaomiMiotConfig(
                    token_file=str(Path(temp_dir) / "token.json"),
                    uuid_file=str(Path(temp_dir) / "uuid"),
                    cache_dir=str(Path(temp_dir) / "cache"),
                ),
                token=XiaomiMiotToken(access_token="access-secret"),
            )

        with patch.object(client, "get_devices", return_value={}):
            with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                result = client.control_device(request="执行离家场景")

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["decision"], "unsupported")
        self.assertEqual(result["operation"], "scene")
        self.assertIn("不能执行米家场景", result["direct_response"])
        send_ctrl.assert_not_called()

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
        self.assertIsInstance(result["candidates"][0]["match_score"], int)
        self.assertIn("area", result["candidates"][0]["hit_fields"])
        self.assertIn("device_class", result["candidates"][0]["hit_fields"])

    def test_control_device_reports_capability_gap_when_no_control_item_exists(self) -> None:
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
            "did-light": {
                "did": "did-light",
                "name": "书房灯",
                "room_name": "书房",
                "device_class": "light",
                "online": True,
                "urn": "urn:test",
            },
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value={"items": {}}):
                result = client.control_device(request="打开书房灯")

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["capability_gap"], "no_control_item")
        self.assertEqual(result["device"]["name"], "书房灯")

    def test_control_device_sets_temperature_value(self) -> None:
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
            "did-ac": {
                "did": "did-ac",
                "name": "客厅空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            }
        }
        spec = {
            "items": {
                "prop.0.2.2": {
                    "iid": "prop.0.2.2",
                    "kind": "property",
                    "name": "target-temperature",
                    "description": "Air Conditioner - Target Temperature",
                    "service": "Air Conditioner",
                    "format": "uint8",
                    "readable": True,
                    "writeable": True,
                    "unit": "celsius",
                    "value_range": {"min": 16, "max": 30, "step": 1},
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                    send_ctrl.return_value = {
                        "status": "verified",
                        "did": "did-ac",
                        "iid": "prop.0.2.2",
                        "readback_value": 26,
                    }
                    result = client.control_device(request="把客厅空调调到26度")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["action"], "set_value")
        self.assertEqual(result["target_value"], 26)
        send_ctrl.assert_called_once_with("did-ac", "prop.0.2.2", 26, verify=True)

    def test_control_device_lowers_temperature_relatively(self) -> None:
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
            "did-ac": {
                "did": "did-ac",
                "name": "客厅空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            }
        }
        spec = {
            "items": {
                "prop.0.2.4": {
                    "iid": "prop.0.2.4",
                    "kind": "property",
                    "name": "target-temperature",
                    "description": "Air Conditioner - Target Temperature",
                    "service": "Air Conditioner",
                    "format": "float",
                    "readable": True,
                    "writeable": True,
                    "unit": "celsius",
                    "value_range": {"min": 16, "max": 31, "step": 0.5},
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc", return_value={"value": 26.0}):
                    with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                        send_ctrl.return_value = {
                            "status": "verified",
                            "did": "did-ac",
                            "iid": "prop.0.2.4",
                            "readback_value": 25.5,
                        }
                        result = client.control_device(request="把客厅空调温度降低0.5度")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["action"], "set_value")
        self.assertEqual(result["previous_value"], 26.0)
        self.assertEqual(result["target_value"], 25.5)
        send_ctrl.assert_called_once_with("did-ac", "prop.0.2.4", 25.5, verify=True)

    def test_control_device_lowers_temperature_with_jian_phrase(self) -> None:
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
            "did-ac": {
                "did": "did-ac",
                "name": "书房空调",
                "room_name": "书房",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            }
        }
        spec = {
            "items": {
                "prop.0.2.4": {
                    "iid": "prop.0.2.4",
                    "kind": "property",
                    "name": "target-temperature",
                    "description": "Air Conditioner - Target Temperature",
                    "service": "Air Conditioner",
                    "format": "float",
                    "readable": True,
                    "writeable": True,
                    "unit": "celsius",
                    "value_range": {"min": 16, "max": 31, "step": 0.5},
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc", return_value={"value": 27.0}):
                    with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                        send_ctrl.return_value = {
                            "status": "verified",
                            "did": "did-ac",
                            "iid": "prop.0.2.4",
                            "readback_value": 26.5,
                        }
                        result = client.control_device(request="书房的空调温度减0.5度")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["action"], "set_value")
        self.assertEqual(result["previous_value"], 27.0)
        self.assertEqual(result["target_value"], 26.5)
        send_ctrl.assert_called_once_with("did-ac", "prop.0.2.4", 26.5, verify=True)

    def test_control_device_sets_brightness_value(self) -> None:
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
                "name": "客厅吸顶灯",
                "room_name": "客厅",
                "device_class": "light",
                "online": True,
                "urn": "urn:test",
            }
        }
        spec = {
            "items": {
                "prop.0.2.3": {
                    "iid": "prop.0.2.3",
                    "kind": "property",
                    "name": "brightness",
                    "description": "Light - Brightness",
                    "service": "Light",
                    "format": "uint8",
                    "readable": True,
                    "writeable": True,
                    "value_range": {"min": 1, "max": 100, "step": 1},
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                    send_ctrl.return_value = {
                        "status": "verified",
                        "did": "did-light",
                        "iid": "prop.0.2.3",
                        "readback_value": 60,
                    }
                    result = client.control_device(request="把客厅灯亮度调到60%")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["target_value"], 60)
        send_ctrl.assert_called_once_with("did-light", "prop.0.2.3", 60, verify=True)

    def test_control_device_sets_curtain_position_value(self) -> None:
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
            "did-curtain": {
                "did": "did-curtain",
                "name": "客厅窗帘",
                "room_name": "客厅",
                "device_class": "curtain",
                "online": True,
                "urn": "urn:test",
            }
        }
        spec = {
            "items": {
                "prop.0.2.4": {
                    "iid": "prop.0.2.4",
                    "kind": "property",
                    "name": "target-position",
                    "description": "Curtain - Target Position",
                    "service": "Curtain",
                    "format": "uint8",
                    "readable": True,
                    "writeable": True,
                    "value_range": {"min": 0, "max": 100, "step": 1},
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                    send_ctrl.return_value = {
                        "status": "verified",
                        "did": "did-curtain",
                        "iid": "prop.0.2.4",
                        "readback_value": 50,
                    }
                    result = client.control_device(request="客厅窗帘开一半")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["target_value"], 50)
        send_ctrl.assert_called_once_with("did-curtain", "prop.0.2.4", 50, verify=True)

    def test_control_device_sets_mode_value_list(self) -> None:
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
            "did-ac": {
                "did": "did-ac",
                "name": "客厅空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            }
        }
        spec = {
            "items": {
                "prop.0.2.5": {
                    "iid": "prop.0.2.5",
                    "kind": "property",
                    "name": "mode",
                    "description": "Air Conditioner - Mode",
                    "service": "Air Conditioner",
                    "format": "uint8",
                    "readable": True,
                    "writeable": True,
                    "value_list": [
                        {"value": 1, "description": "Auto"},
                        {"value": 2, "description": "Cool"},
                        {"value": 3, "description": "Heat"},
                    ],
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                    send_ctrl.return_value = {
                        "status": "verified",
                        "did": "did-ac",
                        "iid": "prop.0.2.5",
                        "readback_value": 2,
                    }
                    result = client.control_device(request="客厅空调开制冷模式")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["target_value"], 2)
        send_ctrl.assert_called_once_with("did-ac", "prop.0.2.5", 2, verify=True)

    def test_control_device_closes_only_currently_on_candidate_without_state_words(self) -> None:
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
            "did-study": {
                "did": "did-study",
                "name": "书房空调",
                "room_name": "书房",
                "home_name": "家",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
            "did-living": {
                "did": "did-living",
                "name": "客厅空调",
                "room_name": "客厅",
                "home_name": "家",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "on",
                    "description": "Air Conditioner - Power",
                    "service": "Air Conditioner",
                    "format": "bool",
                    "readable": True,
                    "writeable": True,
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc") as send_get:
                    with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                        send_get.side_effect = [
                            {"did": "did-study", "iid": "prop.0.2.1", "value": True},
                            {"did": "did-living", "iid": "prop.0.2.1", "value": False},
                        ]
                        send_ctrl.return_value = {
                            "status": "verified",
                            "did": "did-study",
                            "iid": "prop.0.2.1",
                            "readback_value": False,
                        }
                        result = client.control_device(request="关闭空调")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["device"]["name"], "书房空调")
        self.assertEqual(
            [call_args.args for call_args in send_get.call_args_list],
            [("did-study", "prop.0.2.1"), ("did-living", "prop.0.2.1")],
        )
        send_ctrl.assert_called_once_with("did-study", "prop.0.2.1", False, verify=True)

    def test_control_device_uses_room_descriptor_to_close_only_on_candidate(self) -> None:
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
            "did-ceiling": {
                "did": "did-ceiling",
                "name": "书房吸顶灯",
                "room_name": "书房",
                "home_name": "家",
                "device_class": "light",
                "online": True,
                "urn": "urn:test",
            },
            "did-desk": {
                "did": "did-desk",
                "name": "书房台灯",
                "room_name": "书房",
                "home_name": "家",
                "device_class": "light",
                "online": True,
                "urn": "urn:test",
            },
            "did-kitchen": {
                "did": "did-kitchen",
                "name": "厨房吸顶灯",
                "room_name": "厨房",
                "home_name": "家",
                "device_class": "light",
                "online": True,
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
                    "service": "Light",
                    "format": "bool",
                    "readable": True,
                    "writeable": True,
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc") as send_get:
                    with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                        send_get.side_effect = [
                            {"did": "did-ceiling", "iid": "prop.0.2.1", "value": False},
                            {"did": "did-desk", "iid": "prop.0.2.1", "value": True},
                        ]
                        send_ctrl.return_value = {
                            "status": "verified",
                            "did": "did-desk",
                            "iid": "prop.0.2.1",
                            "readback_value": False,
                        }
                        result = client.control_device(request="关闭书房的灯")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["device"]["name"], "书房台灯")
        self.assertEqual(
            [call_args.args for call_args in send_get.call_args_list],
            [("did-ceiling", "prop.0.2.1"), ("did-desk", "prop.0.2.1")],
        )
        send_ctrl.assert_called_once_with("did-desk", "prop.0.2.1", False, verify=True)

    def test_control_device_matches_room_alias(self) -> None:
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
            "did-living": {
                "did": "did-living",
                "name": "客厅空调",
                "room_name": "客厅",
                "home_name": "家",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
            "did-study": {
                "did": "did-study",
                "name": "书房空调",
                "room_name": "书房",
                "home_name": "家",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "on",
                    "description": "Air Conditioner - Power",
                    "service": "Air Conditioner",
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
                        "did": "did-living",
                        "iid": "prop.0.2.1",
                        "readback_value": False,
                    }
                    result = client.control_device(request="关闭大厅的空调")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["device"]["name"], "客厅空调")
        send_ctrl.assert_called_once_with("did-living", "prop.0.2.1", False, verify=True)

    def test_control_device_matches_generic_bedroom_when_unique(self) -> None:
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
            "did-bedroom": {
                "did": "did-bedroom",
                "name": "主卧吸顶灯",
                "room_name": "主卧",
                "home_name": "家",
                "device_class": "light",
                "online": True,
                "urn": "urn:test",
            },
            "did-living": {
                "did": "did-living",
                "name": "客厅吸顶灯",
                "room_name": "客厅",
                "home_name": "家",
                "device_class": "light",
                "online": True,
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
                    "service": "Light",
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
                        "did": "did-bedroom",
                        "iid": "prop.0.2.1",
                        "readback_value": False,
                    }
                    result = client.control_device(request="关闭卧室的灯")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["device"]["name"], "主卧吸顶灯")
        send_ctrl.assert_called_once_with("did-bedroom", "prop.0.2.1", False, verify=True)

    def test_control_device_opens_only_currently_off_candidate(self) -> None:
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
            "did-ceiling": {
                "did": "did-ceiling",
                "name": "书房吸顶灯",
                "room_name": "书房",
                "device_class": "light",
                "online": True,
                "urn": "urn:test",
            },
            "did-desk": {
                "did": "did-desk",
                "name": "书房台灯",
                "room_name": "书房",
                "device_class": "light",
                "online": True,
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
                    "service": "Light",
                    "format": "bool",
                    "readable": True,
                    "writeable": True,
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc") as send_get:
                    with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                        send_get.side_effect = [
                            {"did": "did-ceiling", "iid": "prop.0.2.1", "value": True},
                            {"did": "did-desk", "iid": "prop.0.2.1", "value": False},
                        ]
                        send_ctrl.return_value = {
                            "status": "verified",
                            "did": "did-desk",
                            "iid": "prop.0.2.1",
                            "readback_value": True,
                        }
                        result = client.control_device(request="打开书房的灯")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["device"]["name"], "书房台灯")
        send_ctrl.assert_called_once_with("did-desk", "prop.0.2.1", True, verify=True)

    def test_control_device_group_command_controls_all_matches_without_confirmation(self) -> None:
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
            "did-study": {
                "did": "did-study",
                "name": "书房空调",
                "room_name": "书房",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
            "did-living": {
                "did": "did-living",
                "name": "客厅空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "on",
                    "description": "Air Conditioner - Power",
                    "service": "Air Conditioner",
                    "format": "bool",
                    "readable": True,
                    "writeable": True,
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                    send_ctrl.side_effect = [
                        {"status": "verified", "did": "did-study", "iid": "prop.0.2.1"},
                        {"status": "verified", "did": "did-living", "iid": "prop.0.2.1"},
                    ]
                    result = client.control_device(request="关闭所有空调")

        self.assertEqual(result["status"], "group_executed")
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(
            [call_args.args for call_args in send_ctrl.call_args_list],
            [
                ("did-study", "prop.0.2.1", False),
                ("did-living", "prop.0.2.1", False),
            ],
        )

    def test_control_device_group_command_filters_by_current_state(self) -> None:
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
            "did-study": {
                "did": "did-study",
                "name": "书房空调",
                "room_name": "书房",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
            "did-living": {
                "did": "did-living",
                "name": "客厅空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "on",
                    "description": "Air Conditioner - Power",
                    "service": "Air Conditioner",
                    "format": "bool",
                    "readable": True,
                    "writeable": True,
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc") as send_get:
                    with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                        send_get.side_effect = [
                            {"did": "did-study", "iid": "prop.0.2.1", "value": True},
                            {"did": "did-living", "iid": "prop.0.2.1", "value": False},
                        ]
                        send_ctrl.return_value = {
                            "status": "verified",
                            "did": "did-study",
                            "iid": "prop.0.2.1",
                        }
                        result = client.control_device(request="把家里开着的空调都关了")

        self.assertEqual(result["status"], "group_executed")
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["skipped_count"], 1)
        send_ctrl.assert_called_once_with("did-study", "prop.0.2.1", False, verify=True)

    def test_control_device_does_not_treat_chengdu_as_group_quantifier(self) -> None:
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
            "did-chengdu": {
                "did": "did-chengdu",
                "name": "成都空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
            "did-beijing": {
                "did": "did-beijing",
                "name": "北京空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "on",
                    "description": "Air Conditioner - Power",
                    "service": "Air Conditioner",
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
                        "did": "did-chengdu",
                        "iid": "prop.0.2.1",
                        "readback_value": False,
                    }
                    result = client.control_device(request="关闭成都空调")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["device"]["name"], "成都空调")
        self.assertNotIn("success_count", result)
        send_ctrl.assert_called_once_with("did-chengdu", "prop.0.2.1", False, verify=True)

    def test_control_device_stays_ambiguous_when_multiple_candidates_are_on(self) -> None:
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
            "did-study": {
                "did": "did-study",
                "name": "书房空调",
                "room_name": "书房",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
            "did-living": {
                "did": "did-living",
                "name": "客厅空调",
                "room_name": "客厅",
                "device_class": "aircondition",
                "online": True,
                "urn": "urn:test",
            },
        }
        spec = {
            "items": {
                "prop.0.2.1": {
                    "iid": "prop.0.2.1",
                    "kind": "property",
                    "name": "on",
                    "description": "Air Conditioner - Power",
                    "service": "Air Conditioner",
                    "format": "bool",
                    "readable": True,
                    "writeable": True,
                }
            }
        }

        with patch.object(client, "get_devices", return_value=devices):
            with patch.object(client, "get_device_spec", return_value=spec):
                with patch.object(client, "send_get_rpc") as send_get:
                    with patch.object(client, "send_ctrl_rpc") as send_ctrl:
                        send_get.side_effect = [
                            {"did": "did-study", "iid": "prop.0.2.1", "value": True},
                            {"did": "did-living", "iid": "prop.0.2.1", "value": True},
                        ]
                        result = client.control_device(request="关闭空调")

        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(
            [candidate["name"] for candidate in result["candidates"]],
            ["书房空调", "客厅空调"],
        )
        send_ctrl.assert_not_called()

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
