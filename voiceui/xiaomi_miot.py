from __future__ import annotations

import base64
import difflib
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from voiceui.logs import log_event
from voiceui.models import XiaomiMiotConfig

PROJECT_CODE = "mico"
MIHOME_HTTP_API_TIMEOUT = 30
MIHOME_HTTP_USER_AGENT = "mico/docker"
MIHOME_HTTP_X_CLIENT_BIZID = "micoapi"
MIHOME_HTTP_X_ENCRYPT_TYPE = "1"
OAUTH2_CLIENT_ID = "2882303761520431603"
OAUTH2_AUTH_URL = "https://account.xiaomi.com/oauth2/authorize"
OAUTH2_API_HOST_DEFAULT = "mico.api.mijia.tech"
TOKEN_EXPIRES_TS_RATIO = 0.7

MIHOME_HTTP_API_PUBKEY = (
    "-----BEGIN PUBLIC KEY-----"
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzH220YGgZOlXJ4eSleFb"
    "Beylq4qHsVNzhPTUTy/caDb4a3GzqH6SX4GiYRilZZZrjjU2ckkr8GM66muaIuJw"
    "r8ZB9SSY3Hqwo32tPowpyxobTN1brmqGK146X6JcFWK/QiUYVXZlcHZuMgXLlWyn"
    "zTMVl2fq7wPbzZwOYFxnSRh8YEnXz6edHAqJqLEqZMP00bNFBGP+yc9xmc7ySSyw"
    "OgW/muVzfD09P2iWhl3x8N+fBBWpuI5HjvyQuiX8CZg3xpEeCV8weaprxMxR0epM"
    "3l7T6rJuPXR1D7yhHaEQj2+dyrZTeJO8D8SnOgzV5j4bp1dTunlzBXGYVjqDsRhZ"
    "qQIDAQAB"
    "-----END PUBLIC KEY-----"
)


class XiaomiMiotError(RuntimeError):
    """Raised when the Xiaomi Home cloud API cannot satisfy a request."""


@dataclass(slots=True)
class XiaomiMiotToken:
    access_token: str
    refresh_token: str = ""
    expires_ts: int = 0
    user_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class XiaomiMiotClient:
    def __init__(self, config: XiaomiMiotConfig, token: XiaomiMiotToken | None = None):
        self.config = config
        self._token = token
        self._uuid = _load_or_create_uuid(config)
        self._device_id = f"{PROJECT_CODE}.{self._uuid}"
        self._state = hashlib.sha1(f"d={self._device_id}".encode()).hexdigest()
        self._host = _api_host(config.cloud_server)
        self._base_url = f"https://{self._host}"
        self._aes_key: bytes | None = None
        self._client_secret_b64 = ""
        self._devices_cache: dict[str, dict[str, Any]] | None = None
        self._homes_cache: dict[str, dict[str, Any]] | None = None
        self._spec_cache: dict[str, dict[str, Any]] = {}
        self._urn_by_model_cache: dict[str, str | None] = {}

    def generate_auth_url(self, skip_confirm: bool = False) -> dict[str, str]:
        params = {
            "redirect_uri": self.config.redirect_uri,
            "client_id": OAUTH2_CLIENT_ID,
            "response_type": "code",
            "device_id": self._device_id,
            "state": self._state,
            "skip_confirm": skip_confirm,
        }
        return {
            "auth_url": f"{OAUTH2_AUTH_URL}?{urllib.parse.urlencode(params)}",
            "state": self._state,
        }

    def exchange_code(self, code: str, state: str | None = None) -> XiaomiMiotToken:
        code = str(code or "").strip()
        if not code:
            raise XiaomiMiotError("code is required")
        if state and state != self._state:
            raise XiaomiMiotError("OAuth state does not match this VoiceUI device id")
        token = self._exchange_token(
            {
                "client_id": OAUTH2_CLIENT_ID,
                "redirect_uri": self.config.redirect_uri,
                "code": code,
                "device_id": self._device_id,
            }
        )
        self._token = token
        save_xiaomi_miot_token(self.config, token)
        return token

    def refresh_access_token(self, refresh_token: str | None = None) -> XiaomiMiotToken:
        refresh = str(refresh_token or (self._token.refresh_token if self._token else "")).strip()
        if not refresh:
            raise XiaomiMiotError("refresh_token is required")
        token = self._exchange_token(
            {
                "client_id": OAUTH2_CLIENT_ID,
                "redirect_uri": self.config.redirect_uri,
                "refresh_token": refresh,
            }
        )
        self._token = token
        save_xiaomi_miot_token(self.config, token)
        return token

    def get_homes(
        self,
        fetch_share_home: bool | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        if self._homes_cache is not None and not refresh:
            return self._homes_cache
        include_shared = (
            self.config.fetch_share_home if fetch_share_home is None else fetch_share_home
        )
        response = self._api_post(
            "/app/v2/homeroom/gethome",
            {
                "limit": 150,
                "fetch_share": include_shared,
                "fetch_share_dev": include_shared,
                "plat_form": 0,
                "app_ver": 9,
            },
        )
        result = _expect_mapping(response.get("result"), "MIoT homes response missing result")
        homes: dict[str, dict[str, Any]] = {}
        for home in [*result.get("homelist", []), *result.get("share_home_list", [])]:
            if not isinstance(home, dict):
                continue
            home_id = str(home.get("id") or "")
            if not home_id or not home.get("name"):
                continue
            rooms: dict[str, dict[str, Any]] = {}
            for room in home.get("roomlist", []) or []:
                if not isinstance(room, dict) or not room.get("id"):
                    continue
                room_id = str(room["id"])
                rooms[room_id] = {
                    "room_id": room_id,
                    "room_name": str(room.get("name") or ""),
                    "dids": [str(did) for did in room.get("dids", []) or []],
                }
            homes[home_id] = {
                "home_id": home_id,
                "home_name": str(home.get("name") or ""),
                "share_home": home.get("shareflag", 0) == 1,
                "uid": str(home.get("uid") or ""),
                "dids": [str(did) for did in home.get("dids", []) or []],
                "room_list": rooms,
            }
        if result.get("has_more") and result.get("max_id"):
            more_homes = self._get_dev_room_page(str(result["max_id"]))
            _merge_home_room_pages(homes, more_homes)
        self._homes_cache = homes
        return homes

    def get_area_info(self) -> list[dict[str, Any]]:
        areas: list[dict[str, Any]] = []
        for home_id, home in self.get_homes().items():
            if home.get("dids"):
                areas.append(
                    {
                        "area_id": home_id,
                        "area_name": home.get("home_name") or "Xiaomi Home",
                        "kind": "home",
                    }
                )
            for room_id, room in home.get("room_list", {}).items():
                if room.get("dids"):
                    areas.append(
                        {
                            "area_id": room_id,
                            "area_name": f"{home.get('home_name')}-{room.get('room_name')}",
                            "kind": "room",
                            "home_id": home_id,
                        }
                    )
        return areas

    def get_device_classes(self) -> list[str]:
        classes = {
            str(device.get("device_class") or "")
            for device in self.get_devices().values()
            if device.get("device_class")
        }
        return sorted(classes)

    def get_devices(
        self,
        area_id: str | None = None,
        device_class: str | None = None,
        refresh: bool = False,
    ) -> dict[str, dict[str, Any]]:
        if self._devices_cache is None or refresh:
            self._devices_cache = self._load_devices()
        devices = self._devices_cache
        filtered: dict[str, dict[str, Any]] = {}
        for did, device in devices.items():
            if area_id and area_id not in (device.get("room_id"), device.get("home_id")):
                continue
            if device_class and device.get("device_class") != device_class:
                continue
            filtered[did] = dict(device)
        return filtered

    def get_device_spec(self, did: str) -> dict[str, Any]:
        did = str(did or "").strip()
        if not did:
            raise XiaomiMiotError("did is required")
        devices = self.get_devices()
        device = devices.get(did)
        if device is None:
            raise XiaomiMiotError("Unknown device id. Call xiaomi_miot_get_devices first.")
        urn = str(device.get("urn") or "").strip()
        if not urn:
            raise XiaomiMiotError(f"Device {did} does not have a MIoT spec urn")
        spec = self._get_spec_by_urn(urn)
        return {
            "did": did,
            "name": device.get("name"),
            "model": device.get("model"),
            **spec,
        }

    def control_device(
        self,
        *,
        request: str = "",
        area: str = "",
        device: str = "",
        device_class: str = "",
        action: str = "",
        property_query: str = "",
        value: Any = None,
        relative_delta: Any = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        devices = list(self.get_devices().values())
        command = _miot_command_from_arguments(
            devices=devices,
            request=request,
            area=area,
            device=device,
            device_class=device_class,
            action=action,
            property_query=property_query,
            value=value,
            relative_delta=relative_delta,
        )
        unsupported_operation = _unsupported_miot_operation_response(command)
        if unsupported_operation is not None:
            return unsupported_operation
        matches = _resolve_device_matches(devices, command)
        if not matches and command.get("device_class") and command.get("device"):
            relaxed_command = {**command, "device_class": ""}
            matches = _resolve_device_matches(devices, relaxed_command)
        if not matches:
            _log_miot_resolver(command, "not_found", 0, reason="no_matches")
            return {
                "status": "not_found",
                "decision": "not_found",
                "message": "没有找到匹配的米家设备。",
                "query": command,
            }

        if _is_group_control_command(command):
            return self._execute_group_control(matches, command, dry_run=dry_run)

        best_score = matches[0][0]
        best_matches = [match for match in matches if best_score - match[0] <= 4]
        if len(best_matches) > 1:
            state_matches = self._resolve_power_state_control_matches(best_matches, command)
            if state_matches is not None:
                if len(state_matches) == 1:
                    best_matches = state_matches
                elif state_matches:
                    _log_miot_resolver(
                        command,
                        "ambiguous",
                        len(state_matches),
                        reason="multiple_state_matches",
                    )
                    return {
                        "status": "ambiguous",
                        "decision": "ambiguous",
                        "message": "当前有多个符合状态的匹配设备，需要用户指定其中一个。",
                        "candidates": _public_match_candidates(state_matches[:5], command),
                        "query": command,
                    }
                else:
                    state_text = _target_power_state_description(command)
                    _log_miot_resolver(
                        command,
                        "not_found",
                        len(best_matches),
                        reason="no_state_matches",
                    )
                    return {
                        "status": "not_found",
                        "decision": "not_found",
                        "message": f"没有发现当前{state_text}的匹配设备。",
                        "query": command,
                    }
            if len(best_matches) == 1:
                target_device = dict(best_matches[0][1])
            else:
                _log_miot_resolver(
                    command,
                    "ambiguous",
                    len(best_matches),
                    reason="multiple_matches",
                )
                return {
                    "status": "ambiguous",
                    "decision": "ambiguous",
                    "message": "找到多个匹配的米家设备，需要用户指定其中一个。",
                    "candidates": _public_match_candidates(best_matches[:5], command),
                    "query": command,
                }
        else:
            target_device = dict(best_matches[0][1])

        if not target_device.get("online", True):
            _log_miot_resolver(command, "offline", 1, reason="target_offline")
            return {
                "status": "offline",
                "decision": "unsupported",
                "message": f"{target_device.get('name') or '设备'} 当前离线，无法控制。",
                "device": _public_device(target_device),
            }

        spec = self.get_device_spec(str(target_device["did"]))
        control = _select_control_item(spec, command)
        if control is None:
            _log_miot_resolver(command, "unsupported", 1, reason="no_control_item")
            return {
                "status": "unsupported",
                "decision": "unsupported",
                "capability_gap": "no_control_item",
                "message": "匹配到了设备，但没有找到适合该指令的可写 MIoT 属性或动作。",
                "device": _public_device(target_device),
                "query": command,
            }
        control = self._resolve_relative_control_value(str(target_device["did"]), control)
        if dry_run:
            _log_miot_resolver(command, "resolved", 1, reason="dry_run")
            return {
                "status": "resolved",
                "decision": "resolved",
                "device": _public_device(target_device),
                "iid": control["iid"],
                "target_value": control["value"],
                "previous_value": control.get("previous_value"),
                "action": command.get("action"),
                "item": control["item"],
            }

        result = self.send_ctrl_rpc(
            str(target_device["did"]),
            str(control["iid"]),
            control["value"],
            verify=True,
        )
        _log_miot_resolver(command, "executed", 1, reason="single_match")
        return {
            **result,
            "decision": "executed",
            "device": _public_device(target_device),
            "target_value": control["value"],
            "previous_value": control.get("previous_value"),
            "action": command.get("action"),
            "item": control["item"],
        }

    def _resolve_relative_control_value(
        self,
        did: str,
        control: dict[str, Any],
    ) -> dict[str, Any]:
        relative_delta = control.get("relative_delta")
        if not isinstance(relative_delta, (int, float)) or isinstance(relative_delta, bool):
            return control
        item = control.get("item") if isinstance(control.get("item"), dict) else {}
        current = self.send_get_rpc(did, str(control["iid"])).get("value")
        target_value = _apply_relative_numeric_delta(current, float(relative_delta), item)
        return {
            **control,
            "value": target_value,
            "previous_value": current,
        }

    def _execute_group_control(
        self,
        matches: list[tuple[int, dict[str, Any]]],
        command: dict[str, Any],
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        state_filter = _state_filter_for_group_control(command)
        successes: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        for _score, raw_device in matches:
            device = dict(raw_device)
            public_device = _public_device(device)
            if not device.get("online", True):
                failures.append({"device": public_device, "reason": "offline"})
                continue

            if state_filter is not None:
                state = self._read_device_power_state(device)
                if state is None:
                    failures.append({"device": public_device, "reason": "state_unreadable"})
                    continue
                if state is not state_filter:
                    skipped.append({"device": public_device, "reason": "state_mismatch"})
                    continue

            try:
                spec = self.get_device_spec(str(device["did"]))
                control = _select_control_item(spec, command)
                if control is None:
                    failures.append({"device": public_device, "reason": "unsupported"})
                    continue
                control = self._resolve_relative_control_value(str(device["did"]), control)
                if dry_run:
                    successes.append(
                        {
                            "device": public_device,
                            "iid": control["iid"],
                            "target_value": control["value"],
                            "previous_value": control.get("previous_value"),
                            "item": control["item"],
                        }
                    )
                    continue
                result = self.send_ctrl_rpc(
                    str(device["did"]),
                    str(control["iid"]),
                    control["value"],
                    verify=True,
                )
                successes.append(
                    {
                        **result,
                        "device": public_device,
                        "target_value": control["value"],
                        "previous_value": control.get("previous_value"),
                        "item": control["item"],
                    }
                )
            except RuntimeError as exc:
                failures.append({"device": public_device, "reason": str(exc)})

        if not successes and not failures:
            state_text = _target_power_state_description(command)
            _log_miot_resolver(
                command,
                "not_found",
                len(matches),
                reason="group_state_no_matches",
            )
            return {
                "status": "not_found",
                "decision": "not_found",
                "message": f"没有发现当前{state_text}的匹配设备。",
                "query": command,
                "skipped": skipped,
            }

        decision = "resolved" if dry_run else "executed"
        _log_miot_resolver(
            command,
            decision,
            len(matches),
            reason="group_control",
            success_count=len(successes),
            failure_count=len(failures),
        )
        return {
            "status": "group_resolved" if dry_run else "group_executed",
            "decision": decision,
            "action": command.get("action"),
            "target_value": command.get("value"),
            "success_count": len(successes),
            "failure_count": len(failures),
            "skipped_count": len(skipped),
            "devices": [entry["device"] for entry in successes],
            "failures": failures,
            "skipped": skipped,
            "query": command,
            "direct_response": _format_group_control_response(
                command,
                successes,
                failures,
                dry_run=dry_run,
            ),
        }

    def _resolve_power_state_control_matches(
        self,
        matches: list[tuple[int, dict[str, Any]]],
        command: dict[str, Any],
    ) -> list[tuple[int, dict[str, Any]]] | None:
        target_state = _target_power_state_for_control(command)
        if target_state is None:
            return None

        known_matches: list[tuple[int, dict[str, Any]]] = []
        state_matches: list[tuple[int, dict[str, Any]]] = []
        for score, device in matches:
            if not device.get("online", True):
                continue
            state = self._read_device_power_state(device)
            if state is None:
                return None
            known_matches.append((score, device))
            if state is target_state:
                state_matches.append((score, device))
        if not known_matches:
            return None
        return state_matches

    def _read_device_power_state(self, device: dict[str, Any]) -> bool | None:
        did = str(device.get("did") or "")
        if not did:
            return None
        try:
            spec = self.get_device_spec(did)
            item = _select_power_state_property(spec)
            if item is None:
                return None
            readback = self.send_get_rpc(did, str(item["iid"]))
        except (KeyError, TypeError, RuntimeError):
            return None

        if not isinstance(readback, dict):
            return None
        value = readback.get("value")
        if miot_values_equal(True, value):
            return True
        if miot_values_equal(False, value):
            return False
        return None

    def read_device_property(
        self,
        *,
        request: str = "",
        area: str = "",
        device: str = "",
        device_class: str = "",
        property_query: str = "",
    ) -> dict[str, Any]:
        devices = list(self.get_devices().values())
        command = _miot_read_command_from_arguments(
            devices=devices,
            request=request,
            area=area,
            device=device,
            device_class=device_class,
            property_query=property_query,
        )
        matches = _resolve_device_matches(devices, command)
        if not matches and command.get("device_class") and command.get("device"):
            relaxed_command = {**command, "device_class": ""}
            matches = _resolve_device_matches(devices, relaxed_command)
        if not matches:
            _log_miot_resolver(command, "not_found", 0, reason="read_no_matches")
            return {
                "status": "not_found",
                "decision": "not_found",
                "message": "没有找到匹配的米家设备。",
                "query": command,
            }

        best_score = matches[0][0]
        best_matches = [match for match in matches if best_score - match[0] <= 4]
        if len(best_matches) > 1:
            group_read = self._read_group_power_property(best_matches, command)
            if group_read is not None:
                return group_read
            _log_miot_resolver(
                command,
                "ambiguous",
                len(best_matches),
                reason="read_multiple_matches",
            )
            return {
                "status": "ambiguous",
                "decision": "ambiguous",
                "message": "找到多个匹配的米家设备，需要用户指定其中一个。",
                "candidates": _public_match_candidates(best_matches[:5], command),
                "query": command,
            }

        target_device = dict(best_matches[0][1])
        if not target_device.get("online", True):
            _log_miot_resolver(command, "offline", 1, reason="read_target_offline")
            return {
                "status": "offline",
                "decision": "unsupported",
                "message": f"{target_device.get('name') or '设备'} 当前离线，无法读取。",
                "device": _public_device(target_device),
            }

        spec = self.get_device_spec(str(target_device["did"]))
        item = _select_read_property(spec, command)
        if item is None:
            _log_miot_resolver(command, "unsupported", 1, reason="read_no_item")
            return {
                "status": "unsupported",
                "decision": "unsupported",
                "capability_gap": "no_readable_property",
                "message": "匹配到了设备，但没有找到适合读取的 MIoT 属性。",
                "device": _public_device(target_device),
                "query": command,
            }

        readback = self.send_get_rpc(str(target_device["did"]), str(item["iid"]))
        value = readback.get("value")
        _log_miot_resolver(command, "resolved", 1, reason="property_read")
        return {
            "status": "property_read",
            "decision": "resolved",
            "device": _public_device(target_device),
            "iid": item["iid"],
            "property": item,
            "value": value,
            "unit": item.get("unit"),
            "readback": readback,
            "direct_response": _format_property_read_response(target_device, item, value),
        }

    def _read_group_power_property(
        self,
        matches: list[tuple[int, dict[str, Any]]],
        command: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not _looks_like_power_property_query(command):
            return None

        readings: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for _score, raw_device in matches:
            device = dict(raw_device)
            public_device = _public_device(device)
            if not device.get("online", True):
                failures.append({"device": public_device, "reason": "offline"})
                continue
            try:
                spec = self.get_device_spec(str(device["did"]))
                item = _select_power_state_property(spec)
                if item is None:
                    failures.append({"device": public_device, "reason": "unsupported"})
                    continue
                readback = self.send_get_rpc(str(device["did"]), str(item["iid"]))
                readings.append(
                    {
                        "device": public_device,
                        "iid": item["iid"],
                        "property": item,
                        "value": readback.get("value"),
                        "readback": readback,
                    }
                )
            except RuntimeError as exc:
                failures.append({"device": public_device, "reason": str(exc)})

        if not readings:
            return None

        _log_miot_resolver(
            command,
            "resolved",
            len(matches),
            reason="group_power_read",
            success_count=len(readings),
            failure_count=len(failures),
        )
        return {
            "status": "property_read_group",
            "decision": "resolved",
            "property": "power",
            "readings": readings,
            "failures": failures,
            "query": command,
            "direct_response": _format_group_power_read_response(command, readings, failures),
        }

    def send_get_rpc(self, did: str, iid: str) -> dict[str, Any]:
        cmd, siid, piid_or_aiid = split_miot_iid(iid)
        if cmd != "prop":
            raise XiaomiMiotError("Getting MIoT state only supports prop.* iids")
        response = self._api_post(
            "/app/v2/miotspec/prop/get",
            {
                "datasource": 1,
                "params": [
                    {
                        "did": did,
                        "siid": siid,
                        "piid": piid_or_aiid,
                    }
                ],
            },
        )
        result = _first_result(response)
        return {
            "did": did,
            "iid": iid,
            "value": result.get("value"),
            "result": result,
        }

    def send_ctrl_rpc(self, did: str, iid: str, value: Any, verify: bool = True) -> dict[str, Any]:
        cmd, siid, piid_or_aiid = split_miot_iid(iid)
        spec_item = self._find_spec_item(did, iid)
        translated_value = value
        if cmd == "prop":
            if spec_item and not spec_item.get("writeable"):
                raise XiaomiMiotError(f"MIoT property is not writeable: {iid}")
            translated_value = coerce_miot_value(value, spec_item)
            response = self._api_post(
                "/app/v2/miotspec/prop/set",
                {
                    "params": [
                        {
                            "did": did,
                            "siid": siid,
                            "piid": piid_or_aiid,
                            "value": translated_value,
                        }
                    ]
                },
                timeout=15,
            )
            result = _first_result(response)
        else:
            action_inputs = coerce_miot_action_inputs(value)
            response = self._api_post(
                "/app/v2/miotspec/action",
                {
                    "params": {
                        "did": did,
                        "siid": siid,
                        "aiid": piid_or_aiid,
                        "in": action_inputs,
                    }
                },
                timeout=15,
            )
            result = _expect_mapping(
                response.get("result"),
                "MIoT action response missing result",
            )
        ok = bool(result) and result.get("code") in (0, 1)
        if not ok:
            result_text = json.dumps(result, ensure_ascii=False)
            raise XiaomiMiotError(f"Device control failed: {result_text}")
        response_payload: dict[str, Any] = {
            "did": did,
            "iid": iid,
            "status": "ok",
            "result": result,
        }
        if (
            cmd == "prop"
            and verify
            and self.config.control_verify
            and spec_item
            and spec_item.get("readable")
        ):
            readback = self._verify_prop_value(did, iid, translated_value)
            response_payload["status"] = "verified"
            response_payload["readback_value"] = readback.get("value")
            response_payload["readback"] = readback
        return response_payload

    def _verify_prop_value(self, did: str, iid: str, expected_value: Any) -> dict[str, Any]:
        delay = max(0.0, float(self.config.control_verify_delay_seconds))
        last_readback: dict[str, Any] | None = None
        for attempt in range(2):
            if delay > 0:
                time.sleep(delay * (attempt + 1))
            last_readback = self.send_get_rpc(did, iid)
            if miot_values_equal(expected_value, last_readback.get("value")):
                return last_readback
        raise XiaomiMiotError(
            "Device control verification failed: "
            f"target={expected_value!r}, readback={(last_readback or {}).get('value')!r}"
        )

    def _load_devices(self) -> dict[str, dict[str, Any]]:
        homes = self.get_homes()
        home_by_did: dict[str, dict[str, Any]] = {}
        for home in homes.values():
            home_id = str(home.get("home_id") or "")
            home_name = str(home.get("home_name") or "")
            for did in home.get("dids", []) or []:
                home_by_did[str(did)] = {
                    "home_id": home_id,
                    "home_name": home_name,
                    "room_id": home_id,
                    "room_name": home_name,
                }
            for room in home.get("room_list", {}).values():
                for did in room.get("dids", []) or []:
                    home_by_did[str(did)] = {
                        "home_id": home_id,
                        "home_name": home_name,
                        "room_id": str(room.get("room_id") or ""),
                        "room_name": str(room.get("room_name") or ""),
                    }
        dids = sorted(home_by_did)[: max(1, int(self.config.max_devices))]
        devices: dict[str, dict[str, Any]] = {}
        for index in range(0, len(dids), 150):
            devices.update(self._get_device_list_page(dids[index : index + 150]))
        for did, home_info in home_by_did.items():
            if did in devices:
                devices[did].update(home_info)
        return devices

    def _get_dev_room_page(self, max_id: str | None = None) -> dict[str, dict[str, Any]]:
        data: dict[str, Any] = {"start_id": max_id, "limit": 150}
        response = self._api_post("/app/v2/homeroom/get_dev_room_page", data)
        result = _expect_mapping(response.get("result"), "MIoT room page response missing result")
        homes: dict[str, dict[str, Any]] = {}
        for home in result.get("info", []) or []:
            if not isinstance(home, dict) or not home.get("id"):
                continue
            home_id = str(home["id"])
            rooms: dict[str, dict[str, Any]] = {}
            for room in home.get("roomlist", []) or []:
                if not isinstance(room, dict) or not room.get("id"):
                    continue
                room_id = str(room["id"])
                rooms[room_id] = {
                    "room_id": room_id,
                    "room_name": str(room.get("name") or ""),
                    "dids": [str(did) for did in room.get("dids", []) or []],
                }
            homes[home_id] = {
                "home_id": home_id,
                "dids": [str(did) for did in home.get("dids", []) or []],
                "room_list": rooms,
            }
        if result.get("has_more") and result.get("max_id"):
            _merge_home_room_pages(homes, self._get_dev_room_page(str(result["max_id"])))
        return homes

    def _get_device_list_page(
        self,
        dids: list[str],
        start_did: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not dids:
            return {}
        data: dict[str, Any] = {
            "limit": 200,
            "get_split_device": True,
            "dids": dids,
        }
        if start_did:
            data["start_did"] = start_did
        response = self._api_post("/app/v2/home/device_list_page", data)
        result = _expect_mapping(response.get("result"), "MIoT device page response missing result")
        devices: dict[str, dict[str, Any]] = {}
        for raw_device in result.get("list", []) or []:
            if not isinstance(raw_device, dict):
                continue
            did = str(raw_device.get("did") or "")
            name = str(raw_device.get("name") or "")
            model = str(raw_device.get("model") or "")
            if not did or not name or not model:
                continue
            urn = str(raw_device.get("spec_type") or "")
            if not urn:
                urn = self._get_urn_by_model(model) or ""
            if not urn:
                continue
            model_parts = model.split(".")
            extra = raw_device.get("extra") if isinstance(raw_device.get("extra"), dict) else {}
            devices[did] = {
                "did": did,
                "name": name,
                "model": model,
                "urn": urn,
                "online": bool(raw_device.get("isOnline", False)),
                "device_class": model_parts[1] if len(model_parts) > 1 else model,
                "manufacturer": model_parts[0] if model_parts else "",
                "voice_ctrl": raw_device.get("voice_ctrl", 0),
                "local_ip": raw_device.get("local_ip"),
                "ssid": raw_device.get("ssid"),
                "fw_version": extra.get("fw_version") if isinstance(extra, dict) else None,
            }
        if result.get("has_more") and result.get("next_start_did"):
            devices.update(self._get_device_list_page(dids, str(result["next_start_did"])))
        return devices

    def _get_urn_by_model(self, model: str) -> str | None:
        if model in self._urn_by_model_cache:
            return self._urn_by_model_cache[model]
        response = _get_json(
            "https://miot-spec.org/internal/urn-by-model-version",
            params={"model": model, "version": 0},
            timeout=10,
        )
        urn = response.get("urn") if isinstance(response, dict) else None
        self._urn_by_model_cache[model] = str(urn) if urn else None
        return self._urn_by_model_cache[model]

    def _get_spec_by_urn(self, urn: str) -> dict[str, Any]:
        if urn in self._spec_cache:
            return self._spec_cache[urn]
        cached = _load_cached_spec(self.config, urn)
        if cached is not None:
            self._spec_cache[urn] = cached
            return cached
        instance = _get_json(
            "https://miot-spec.org/miot-spec-v2/instance",
            params={"type": urn},
            timeout=self.config.request_timeout_seconds,
        )
        spec = parse_miot_spec_lite(instance, urn)
        _save_cached_spec(self.config, urn, spec)
        self._spec_cache[urn] = spec
        return spec

    def _find_spec_item(self, did: str, iid: str) -> dict[str, Any] | None:
        try:
            spec = self.get_device_spec(did)
        except XiaomiMiotError:
            return None
        items = spec.get("items") if isinstance(spec, dict) else None
        item = items.get(iid) if isinstance(items, dict) else None
        return item if isinstance(item, dict) else None

    def _exchange_token(self, data: dict[str, Any]) -> XiaomiMiotToken:
        url = f"https://{self._host}/app/v2/{PROJECT_CODE}/oauth/get_token"
        text = _http_get_text(
            url,
            params={"data": json.dumps(data, separators=(",", ":"))},
            headers={"content-type": "application/x-www-form-urlencoded"},
            timeout=MIHOME_HTTP_API_TIMEOUT,
        )
        response = json.loads(text)
        if not isinstance(response, dict) or response.get("code") != 0:
            raise XiaomiMiotError("Xiaomi OAuth token exchange failed")
        result = _expect_mapping(
            response.get("result"),
            "Xiaomi OAuth token response missing result",
        )
        if not all(key in result for key in ("access_token", "refresh_token", "expires_in")):
            raise XiaomiMiotError("Xiaomi OAuth token response is missing token fields")
        return XiaomiMiotToken(
            access_token=str(result["access_token"]),
            refresh_token=str(result["refresh_token"]),
            expires_ts=int(time.time() + int(result.get("expires_in", 0)) * TOKEN_EXPIRES_TS_RATIO),
        )

    def _api_post(
        self,
        url_path: str,
        data: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return self._api_post_once(url_path, data, timeout)
        except XiaomiMiotError as exc:
            if "HTTP 401" not in str(exc) or not self.config.auto_refresh:
                raise
            token = self._token or load_xiaomi_miot_token(self.config)
            if not token.refresh_token:
                raise
            self._token = token
            try:
                self.refresh_access_token(token.refresh_token)
            except XiaomiMiotError as refresh_exc:
                raise XiaomiMiotError(
                    "Xiaomi MIoT token refresh failed. Call xiaomi_miot_auth_url, "
                    "finish Xiaomi login, then call xiaomi_miot_exchange_auth_code."
                ) from refresh_exc
            return self._api_post_once(url_path, data, timeout)

    def _api_post_once(
        self,
        url_path: str,
        data: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        token = self._ensure_token()
        self._ensure_crypto()
        assert self._aes_key is not None
        encrypted_body = self._aes_encrypt(data).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{url_path}",
            data=encrypted_body,
            headers={
                "Content-Type": "text/plain",
                "User-Agent": MIHOME_HTTP_USER_AGENT,
                "X-Client-BizId": MIHOME_HTTP_X_CLIENT_BIZID,
                "X-Encrypt-Type": MIHOME_HTTP_X_ENCRYPT_TYPE,
                "X-Client-AppId": OAUTH2_CLIENT_ID,
                "X-Client-Secret": self._client_secret_b64,
                "Host": self._host,
                "Authorization": f"Bearer{token.access_token}",
            },
            method="POST",
        )
        response_text = _http_request_text(
            request,
            timeout=timeout or self.config.request_timeout_seconds,
            error_label=f"MIoT API {url_path}",
        )
        response = self._aes_decrypt(response_text)
        if not isinstance(response, dict) or response.get("code") != 0:
            code = response.get("code") if isinstance(response, dict) else "unknown"
            message = response.get("message") if isinstance(response, dict) else ""
            raise XiaomiMiotError(f"MIoT API {url_path} returned code {code}: {message}")
        return response

    def _ensure_token(self) -> XiaomiMiotToken:
        if self._token is None:
            self._token = load_xiaomi_miot_token(self.config)
        if (
            self.config.auto_refresh
            and self._token.refresh_token
            and self._token.expires_ts
            and time.time() >= self._token.expires_ts - 60
        ):
            try:
                self.refresh_access_token(self._token.refresh_token)
            except XiaomiMiotError as refresh_exc:
                raise XiaomiMiotError(
                    "Xiaomi MIoT token refresh failed. Call xiaomi_miot_auth_url, "
                    "finish Xiaomi login, then call xiaomi_miot_exchange_auth_code."
                ) from refresh_exc
        if not self._token.access_token:
            raise XiaomiMiotError("Xiaomi MIoT token is missing")
        return self._token

    def _ensure_crypto(self) -> None:
        if self._aes_key is not None and self._client_secret_b64:
            return
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
        except ImportError as exc:
            raise XiaomiMiotError(
                "Xiaomi MIoT requires cryptography. Install with: pip install -e \".[miot]\""
            ) from exc

        self._aes_key = os.urandom(16)
        public_key = load_pem_public_key(_normalize_pem(MIHOME_HTTP_API_PUBKEY), default_backend())
        encrypted_key = public_key.encrypt(self._aes_key, asym_padding.PKCS1v15())
        self._client_secret_b64 = base64.b64encode(encrypted_key).decode("utf-8")

    def _aes_encrypt(self, data: dict[str, Any]) -> str:
        assert self._aes_key is not None
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding as sym_padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        cipher = Cipher(
            algorithms.AES(self._aes_key),
            modes.CBC(self._aes_key),
            backend=default_backend(),
        )
        padder = sym_padding.PKCS7(128).padder()
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        padded = padder.update(raw) + padder.finalize()
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded) + encryptor.finalize()
        return base64.b64encode(encrypted).decode("utf-8")

    def _aes_decrypt(self, data: str) -> dict[str, Any]:
        assert self._aes_key is not None
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding as sym_padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        cipher = Cipher(
            algorithms.AES(self._aes_key),
            modes.CBC(self._aes_key),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(base64.b64decode(data)) + decryptor.finalize()
        unpadder = sym_padding.PKCS7(128).unpadder()
        raw = unpadder.update(decrypted) + unpadder.finalize()
        return json.loads(raw.decode("utf-8"))


class XiaomiMiotController:
    def __init__(self, config: XiaomiMiotConfig):
        self.config = config
        self._client: XiaomiMiotClient | None = None

    def auth_url(self, arguments: dict[str, Any]) -> dict[str, Any]:
        skip_confirm = _optional_bool(arguments.get("skip_confirm"), default=False)
        return self._get_client().generate_auth_url(skip_confirm=skip_confirm)

    def exchange_auth_code(self, arguments: dict[str, Any]) -> dict[str, Any]:
        token = self._get_client().exchange_code(
            code=str(arguments.get("code") or ""),
            state=str(arguments.get("state") or "") or None,
        )
        return {"saved": True, "expires_ts": token.expires_ts}

    def get_area_info(self, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        del arguments
        return {"areas": self._get_client().get_area_info()}

    def get_device_classes(self, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        del arguments
        return {"device_classes": self._get_client().get_device_classes()}

    def get_devices(self, arguments: dict[str, Any]) -> dict[str, Any]:
        devices = self._get_client().get_devices(
            area_id=_optional_text(arguments, "area_id"),
            device_class=_optional_text(arguments, "device_class"),
            refresh=_optional_bool(arguments.get("refresh"), default=False),
        )
        return {"devices": list(devices.values())}

    def get_device_spec(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._get_client().get_device_spec(_required_text(arguments, "did"))

    def get_property(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._get_client().send_get_rpc(
            did=_required_text(arguments, "did"),
            iid=_required_text(arguments, "iid"),
        )

    def read_device_property(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._get_client().read_device_property(
            request=str(arguments.get("request") or ""),
            area=str(arguments.get("area") or ""),
            device=str(arguments.get("device") or ""),
            device_class=str(arguments.get("device_class") or ""),
            property_query=str(arguments.get("property") or arguments.get("property_query") or ""),
        )

    def control_device(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._get_client().control_device(
            request=str(arguments.get("request") or ""),
            area=str(arguments.get("area") or ""),
            device=str(arguments.get("device") or ""),
            device_class=str(arguments.get("device_class") or ""),
            action=str(arguments.get("action") or ""),
            property_query=str(arguments.get("property") or arguments.get("property_query") or ""),
            value=arguments.get("value"),
            relative_delta=arguments.get("relative_delta"),
            dry_run=_optional_bool(arguments.get("dry_run"), default=False),
        )

    def control(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "value" not in arguments:
            raise XiaomiMiotError("value is required")
        return self._get_client().send_ctrl_rpc(
            did=_required_text(arguments, "did"),
            iid=_required_text(arguments, "iid"),
            value=arguments["value"],
        )

    def _get_client(self) -> XiaomiMiotClient:
        if self._client is None:
            self._client = XiaomiMiotClient(self.config)
        return self._client


def load_xiaomi_miot_token(config: XiaomiMiotConfig) -> XiaomiMiotToken:
    env_json = os.getenv(config.token_json_env)
    if env_json:
        return _token_from_mapping(json.loads(env_json))

    access_token = os.getenv(config.access_token_env)
    if access_token:
        expires_ts = _safe_int(os.getenv("XIAOMI_MIOT_EXPIRES_TS"), default=0)
        if expires_ts <= 0 and not os.getenv(config.refresh_token_env):
            expires_ts = int(time.time() + 3600)
        return XiaomiMiotToken(
            access_token=access_token,
            refresh_token=os.getenv(config.refresh_token_env, ""),
            expires_ts=expires_ts,
        )

    token_path = _resolve_path(config.token_file)
    if token_path.exists():
        return _token_from_mapping(json.loads(token_path.read_text(encoding="utf-8")))

    client = XiaomiMiotClient(config, token=XiaomiMiotToken(access_token=""))
    auth = client.generate_auth_url()
    raise XiaomiMiotError(
        "Xiaomi MIoT token is not configured. Open the auth_url from "
        "xiaomi_miot_auth_url, finish login, then exchange the redirect code."
        f" auth_url={auth['auth_url']}"
    )


def save_xiaomi_miot_token(config: XiaomiMiotConfig, token: XiaomiMiotToken) -> None:
    token_path = _resolve_path(config.token_file)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(
        json.dumps(token.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_miot_spec_lite(instance: Any, urn: str) -> dict[str, Any]:
    if not isinstance(instance, dict) or not isinstance(instance.get("services"), list):
        raise XiaomiMiotError(f"Invalid MIoT spec instance: {urn}")
    items: dict[str, dict[str, Any]] = {}
    for service in instance.get("services", []):
        if not isinstance(service, dict):
            continue
        service_iid = service.get("iid")
        service_type = str(service.get("type") or "")
        service_name = _urn_name(service_type)
        if service_name == "device-information":
            continue
        service_desc = str(service.get("description") or service_name)
        properties = (
            service.get("properties", []) if isinstance(service.get("properties"), list) else []
        )
        property_by_iid = {
            prop.get("iid"): prop
            for prop in properties
            if isinstance(prop, dict) and prop.get("iid") is not None
        }
        for prop in properties:
            if not isinstance(prop, dict) or prop.get("iid") is None:
                continue
            piid = prop["iid"]
            iid = f"prop.0.{service_iid}.{piid}"
            access = prop.get("access", [])
            if not isinstance(access, list):
                access = []
            value_range = _parse_value_range(prop.get("value-range"))
            value_list = _parse_value_list(prop.get("value-list"))
            prop_desc = str(prop.get("description") or _urn_name(str(prop.get("type") or "")))
            items[iid] = {
                "iid": iid,
                "kind": "property",
                "service": service_desc,
                "name": _urn_name(str(prop.get("type") or "")),
                "description": f"{service_desc} - {prop_desc}",
                "format": str(prop.get("format") or ""),
                "access": access,
                "readable": "read" in access,
                "writeable": "write" in access,
                "unit": prop.get("unit"),
                "value_range": value_range,
                "value_list": value_list,
            }
        actions = service.get("actions", []) if isinstance(service.get("actions"), list) else []
        for action in actions:
            if not isinstance(action, dict) or action.get("iid") is None:
                continue
            aiid = action["iid"]
            iid = f"action.0.{service_iid}.{aiid}"
            action_desc = str(action.get("description") or _urn_name(str(action.get("type") or "")))
            inputs = []
            for input_iid in action.get("in", []) or []:
                prop = property_by_iid.get(input_iid)
                if not isinstance(prop, dict):
                    inputs.append({"iid": input_iid})
                    continue
                inputs.append(
                    {
                        "iid": input_iid,
                        "name": _urn_name(str(prop.get("type") or "")),
                        "description": prop.get("description"),
                        "format": prop.get("format"),
                        "unit": prop.get("unit"),
                        "value_range": _parse_value_range(prop.get("value-range")),
                        "value_list": _parse_value_list(prop.get("value-list")),
                    }
                )
            items[iid] = {
                "iid": iid,
                "kind": "action",
                "service": service_desc,
                "name": _urn_name(str(action.get("type") or "")),
                "description": f"{service_desc} - {action_desc}",
                "format": "action",
                "readable": False,
                "writeable": True,
                "inputs": inputs,
            }
    return {
        "urn": urn,
        "description": instance.get("description") or _urn_name(urn),
        "items": items,
    }


def split_miot_iid(iid: str) -> tuple[str, int, int]:
    parts = str(iid or "").strip().split(".")
    if (
        len(parts) != 4
        or parts[0] not in ("prop", "action")
        or parts[1] != "0"
        or not parts[2].isdigit()
        or not parts[3].isdigit()
    ):
        raise XiaomiMiotError(
            "Invalid MIoT iid. Use xiaomi_miot_get_device_spec and pass prop.0.siid.piid "
            "or action.0.siid.aiid."
        )
    return parts[0], int(parts[2]), int(parts[3])


def coerce_miot_value(value: Any, spec_item: dict[str, Any] | None = None) -> Any:
    value_format = str((spec_item or {}).get("format") or "")
    if value_format.startswith(("int", "uint")):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise XiaomiMiotError("Invalid MIoT integer value") from exc
    if value_format in ("float", "double"):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise XiaomiMiotError("Invalid MIoT float value") from exc
    if value_format == "string":
        return str(value)
    if value_format == "bool":
        return _optional_bool(value, default=False)
    return value


def coerce_miot_action_inputs(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return parsed
        raise XiaomiMiotError("MIoT action value must be a JSON array string")
    raise XiaomiMiotError("MIoT action value must be a list or JSON array string")


def miot_values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, str):
            lowered = actual.strip().lower()
            if lowered in ("true", "1", "yes", "on", "ok"):
                return expected is True
            if lowered in ("false", "0", "no", "off"):
                return expected is False
        return bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(expected) == float(actual)
        except (TypeError, ValueError):
            return False
    return expected == actual


_DEVICE_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "light": (
        "light",
        "灯",
        "灯光",
        "照明",
        "吸顶灯",
        "筒灯",
        "台灯",
        "灯带",
        "主灯",
        "大灯",
        "床头灯",
        "射灯",
    ),
    "switch": ("switch", "开关", "墙壁开关", "插座", "插排", "排插"),
    "curtain": ("curtain", "窗帘", "帘"),
    "aircondition": ("aircondition", "air-conditioner", "空调", "空调机", "冷气"),
    "airfresh": ("airfresh", "新风"),
    "airpurifier": (
        "airpurifier",
        "air-purifier",
        "purifier",
        "\u7a7a\u6c14\u51c0\u5316\u5668",
        "\u51c0\u5316\u5668",
    ),
    "humidifier": ("humidifier", "加湿器"),
    "fan": ("fan", "风扇", "电扇", "吊扇"),
    "wifispeaker": ("wifispeaker", "音箱", "小爱", "speaker"),
    "motion": ("motion", "传感器", "人体", "感应器"),
    "camera": ("camera", "摄像头", "相机"),
    "lock": ("lock", "门锁", "锁"),
}

_AREA_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("客厅", "大厅", "起居室"),
    ("餐厅", "饭厅"),
    ("厨房", "厨", "厨间"),
    ("书房", "办公室", "工作室"),
    ("主卧", "主卧室", "主人房", "卧室"),
    ("次卧", "次卧室", "客卧", "小卧室", "卧室"),
    ("儿童房", "小孩房", "孩子房", "卧室"),
    ("卫生间", "洗手间", "厕所", "浴室"),
    ("阳台", "露台"),
)

_TURN_ON_WORDS = (
    "turn_on",
    "on",
    "open",
    "打开",
    "开启",
    "开灯",
    "开一下",
    "拉开",
    "拉起来",
    "开了",
    "没有开",
    "没开",
    "还没开",
    "亮",
)
_TURN_OFF_WORDS = (
    "turn_off",
    "off",
    "close",
    "关闭",
    "关掉",
    "关上",
    "关灯",
    "拉上",
    "合上",
    "熄灭",
    "关了",
    "没有关",
    "没关",
    "还没关",
)
_SENSITIVE_CLASSES = {"lock", "cooker", "microwave", "fryer"}


def _miot_command_from_arguments(
    *,
    devices: list[dict[str, Any]],
    request: str,
    area: str,
    device: str,
    device_class: str,
    action: str,
    value: Any = None,
    property_query: str = "",
    relative_delta: Any = None,
) -> dict[str, Any]:
    request_text = str(request or "")
    inferred_relative_delta = (
        relative_delta
        if relative_delta is not None
        else _infer_relative_control_delta(request_text)
    )
    inferred_value = value if value is not None else _infer_control_value(request_text)
    inferred_action = _infer_action(action, request_text, inferred_value)
    command = {
        "request": request_text,
        "operation": _infer_miot_operation(request_text, inferred_action),
        "area": str(area or "").strip(),
        "device": str(device or "").strip(),
        "device_class": _infer_device_class(device_class, device, request_text),
        "action": inferred_action,
        "property": str(property_query or "").strip()
        or _infer_control_property_query(request_text, inferred_value, inferred_action),
        "value": inferred_value,
        "relative_delta": _optional_float(inferred_relative_delta),
    }
    if not command["area"] and request_text:
        command["area"] = _infer_area(devices, request_text)
    if not command["device"] and request_text:
        command["device"] = _infer_device_query(request_text, command["area"], command["action"])
    return command


def _miot_read_command_from_arguments(
    *,
    devices: list[dict[str, Any]],
    request: str,
    area: str,
    device: str,
    device_class: str,
    property_query: str,
) -> dict[str, Any]:
    request_text = str(request or "")
    command = {
        "request": request_text,
        "operation": "query",
        "area": str(area or "").strip(),
        "device": str(device or "").strip(),
        "device_class": _infer_device_class(device_class, device, request_text),
        "property": str(property_query or "").strip() or _infer_property_query(request_text),
    }
    if not command["area"] and request_text:
        command["area"] = _infer_area(devices, request_text)
    if not command["device"] and request_text:
        command["device"] = _infer_device_query(request_text, command["area"], "")
    return command


def _resolve_device_matches(
    devices: list[dict[str, Any]],
    command: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    area_query = str(command.get("area") or "")
    device_query = str(command.get("device") or "")
    class_query = str(command.get("device_class") or "")
    matches: list[tuple[int, dict[str, Any]]] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        score = 0
        if area_query:
            area_score = max(
                _area_match_score(area_query, str(device.get("room_name") or "")),
                _area_match_score(area_query, str(device.get("home_name") or "")),
            )
            if area_score <= 0:
                continue
            score += area_score + 80
        if class_query:
            if _device_class_matches(class_query, str(device.get("device_class") or "")):
                score += 90
            else:
                name_class_score = _text_match_score(class_query, str(device.get("name") or ""))
                if name_class_score <= 0:
                    continue
                score += name_class_score
        if device_query:
            device_score = max(
                _text_match_score(device_query, str(device.get("name") or "")),
                _text_match_score(device_query, str(device.get("device_class") or "")),
                _text_match_score(device_query, str(device.get("model") or "")),
            )
            inferred_class = _infer_device_class("", device_query, "")
            if inferred_class and _device_class_matches(
                inferred_class, str(device.get("device_class") or "")
            ):
                device_score = max(device_score, 75)
            if device_score <= 0:
                continue
            score += device_score
        if not area_query and not class_query and not device_query:
            continue
        if device.get("online", True):
            score += 5
        matches.append((score, device))
    return sorted(matches, key=lambda item: item[0], reverse=True)


def _select_control_item(
    spec: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, Any] | None:
    items = spec.get("items") if isinstance(spec, dict) else None
    if not isinstance(items, dict):
        return None
    action = str(command.get("action") or "")
    value = command.get("value")
    if value is None and action in ("turn_on", "turn_off"):
        value = action == "turn_on"

    if isinstance(value, bool):
        selected = _select_bool_property(items)
        if selected is not None:
            return {"iid": selected["iid"], "value": value, "item": selected}

    selected_set_value = _select_set_value_property(items, command)
    if selected_set_value is not None:
        return selected_set_value

    selected_with_value = _select_value_list_property(items, action)
    if selected_with_value is not None:
        return selected_with_value

    selected_action = _select_action_item(items, action)
    if selected_action is not None:
        return selected_action
    return None


def _select_read_property(
    spec: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, Any] | None:
    items = spec.get("items") if isinstance(spec, dict) else None
    if not isinstance(items, dict):
        return None

    query = str(command.get("property") or command.get("request") or "")
    query_norm = _normalize_text(query)
    aliases = _property_aliases_for_query(query_norm)
    candidates: list[tuple[int, dict[str, Any]]] = []
    for item in items.values():
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "property" or not item.get("readable"):
            continue

        item_text = " ".join(
            str(item.get(key) or "")
            for key in ("name", "description", "service", "unit")
        )
        item_norm = _normalize_text(item_text)
        score = max(
            _text_match_score(query, str(item.get("name") or "")),
            _text_match_score(query, str(item.get("description") or "")),
            _text_match_score(query, str(item.get("service") or "")),
        )
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if alias_norm and alias_norm in item_norm:
                score += 90

        if _looks_like_air_quality_query(query_norm):
            if any(
                token in item_norm
                for token in (
                    "airquality",
                    "aqi",
                    "pm25",
                    "pm2.5",
                    "pm2p5",
                    "density",
                    "particulate",
                    "particle",
                    "空气",
                    "质量",
                    "颗粒",
                )
            ):
                score += 120
            if "temperature" in item_norm or "温度" in item_norm:
                score -= 40
            if "humidity" in item_norm or "湿度" in item_norm:
                score -= 40

        if score > 0:
            candidates.append((score, item))

    if not candidates:
        return None
    return sorted(candidates, key=lambda candidate: candidate[0], reverse=True)[0][1]


def _select_bool_property(items: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for item in items.values():
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "property" or not item.get("writeable"):
            continue
        if str(item.get("format") or "") != "bool":
            continue
        text = _normalize_text(
            " ".join(
                str(item.get(key) or "")
                for key in ("name", "description", "service")
            )
        )
        score = 10
        if any(token in text for token in ("on", "power", "switchstatus", "switch", "开关")):
            score += 80
        if item.get("readable"):
            score += 10
        candidates.append((score, item))
    if not candidates:
        return None
    return sorted(candidates, key=lambda candidate: candidate[0], reverse=True)[0][1]


def _select_power_state_property(spec: dict[str, Any]) -> dict[str, Any] | None:
    items = spec.get("items") if isinstance(spec, dict) else None
    if not isinstance(items, dict):
        return None

    candidates: list[tuple[int, dict[str, Any]]] = []
    for item in items.values():
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "property" or not item.get("readable"):
            continue
        if str(item.get("format") or "") != "bool":
            continue
        text = _normalize_text(
            " ".join(
                str(item.get(key) or "")
                for key in ("name", "description", "service")
            )
        )
        if not any(token in text for token in ("on", "power", "switchstatus", "switch", "开关")):
            continue
        score = 100
        if item.get("writeable"):
            score += 10
        candidates.append((score, item))
    if not candidates:
        return None
    return sorted(candidates, key=lambda candidate: candidate[0], reverse=True)[0][1]


def _select_set_value_property(
    items: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, Any] | None:
    action = str(command.get("action") or "")
    value = command.get("value")
    relative_delta = command.get("relative_delta")
    has_relative_delta = isinstance(relative_delta, (int, float)) and not isinstance(
        relative_delta,
        bool,
    )
    if action != "set_value" and (value is None or isinstance(value, bool)):
        return None

    if isinstance(value, str):
        selected_value_list = _select_value_list_property_for_text(items, command, value)
        if selected_value_list is not None:
            return selected_value_list
        return None

    if (not isinstance(value, (int, float)) or isinstance(value, bool)) and not has_relative_delta:
        return None

    property_query = _normalize_text(str(command.get("property") or command.get("request") or ""))
    aliases = _control_property_aliases_for_query(property_query)
    candidates: list[tuple[int, dict[str, Any], Any]] = []
    for item in items.values():
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "property" or not item.get("writeable"):
            continue
        value_range = item.get("value_range") if isinstance(item.get("value_range"), dict) else None
        if value_range is None:
            continue
        coerced_value = _coerce_numeric_control_value(value, item)
        if coerced_value is None and not has_relative_delta:
            continue
        if not has_relative_delta:
            minimum = value_range.get("min")
            maximum = value_range.get("max")
            if isinstance(minimum, (int, float)) and coerced_value < minimum:
                continue
            if isinstance(maximum, (int, float)) and coerced_value > maximum:
                continue

        item_text = _normalize_text(
            " ".join(
                str(item.get(key) or "")
                for key in ("name", "description", "service", "unit")
            )
        )
        score = max(
            _text_match_score(property_query, str(item.get("name") or "")),
            _text_match_score(property_query, str(item.get("description") or "")),
            _text_match_score(property_query, str(item.get("service") or "")),
        )
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if alias_norm and alias_norm in item_text:
                score += 100
        if score <= 0:
            continue
        candidates.append((score, item, coerced_value))

    if not candidates:
        return None
    _score, item, coerced_value = sorted(
        candidates,
        key=lambda candidate: candidate[0],
        reverse=True,
    )[0]
    result = {"iid": item["iid"], "value": coerced_value, "item": item}
    if has_relative_delta:
        result["relative_delta"] = float(relative_delta)
    return result


def _select_value_list_property_for_text(
    items: dict[str, Any],
    command: dict[str, Any],
    value_text: str,
) -> dict[str, Any] | None:
    property_query = _normalize_text(str(command.get("property") or command.get("request") or ""))
    value_query = _normalize_text(value_text)
    aliases = _control_property_aliases_for_query(property_query)
    candidates: list[tuple[int, dict[str, Any], Any]] = []
    for item in items.values():
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "property" or not item.get("writeable"):
            continue
        value_list = item.get("value_list") if isinstance(item.get("value_list"), list) else []
        if not value_list:
            continue

        item_text = _normalize_text(
            " ".join(str(item.get(key) or "") for key in ("name", "description", "service"))
        )
        item_score = 0
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if alias_norm and alias_norm in item_text:
                item_score += 100

        for option in value_list:
            if not isinstance(option, dict):
                continue
            option_text = _normalize_text(str(option.get("description") or ""))
            option_score = _value_list_option_score(value_query, option_text)
            if option_score <= 0:
                continue
            candidates.append((item_score + option_score, item, option.get("value")))

    if not candidates:
        return None
    _score, item, option_value = sorted(
        candidates,
        key=lambda candidate: candidate[0],
        reverse=True,
    )[0]
    return {"iid": item["iid"], "value": option_value, "item": item}


def _value_list_option_score(value_query: str, option_text: str) -> int:
    if not value_query or not option_text:
        return 0
    mode_aliases = {
        "cool": ("cool", "cooling", "制冷", "冷风", "冷气"),
        "heat": ("heat", "heating", "制热", "暖风", "热风"),
        "dry": ("dry", "dehumidify", "除湿", "抽湿"),
        "fan": ("fan", "ventilate", "送风", "通风"),
        "auto": ("auto", "自动"),
        "sleep": ("sleep", "睡眠"),
        "high": ("high", "strong", "强", "高"),
        "medium": ("medium", "normal", "中", "标准"),
        "low": ("low", "weak", "低", "弱"),
    }
    aliases = mode_aliases.get(value_query, (value_query,))
    for alias in aliases:
        alias_norm = _normalize_text(alias)
        if alias_norm and alias_norm in option_text:
            return 120
    return _text_match_score(value_query, option_text)


def _coerce_numeric_control_value(value: int | float, item: dict[str, Any]) -> int | float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value_format = str(item.get("format") or "")
    if value_format.startswith(("int", "uint")):
        return int(round(float(value)))
    if value_format in ("float", "double"):
        return float(value)
    return None


def _apply_relative_numeric_delta(
    current: Any,
    delta: float,
    item: dict[str, Any],
) -> int | float:
    try:
        target = float(current) + delta
    except (TypeError, ValueError) as exc:
        raise XiaomiMiotError(
            "Cannot apply relative MIoT control without readable current value"
        ) from exc

    value_range = item.get("value_range") if isinstance(item.get("value_range"), dict) else {}
    minimum = value_range.get("min")
    maximum = value_range.get("max")
    step = value_range.get("step")
    if isinstance(minimum, (int, float)) and isinstance(step, (int, float)) and step > 0:
        target = float(minimum) + round((target - float(minimum)) / float(step)) * float(step)
    if isinstance(minimum, (int, float)):
        target = max(float(minimum), target)
    if isinstance(maximum, (int, float)):
        target = min(float(maximum), target)

    value_format = str(item.get("format") or "")
    if value_format.startswith(("int", "uint")):
        return int(round(target))
    return int(target) if target.is_integer() else target


def _select_value_list_property(items: dict[str, Any], action: str) -> dict[str, Any] | None:
    if action not in ("turn_on", "turn_off"):
        return None
    action_terms = (
        ("open", "on", "start", "打开", "开启")
        if action == "turn_on"
        else ("close", "off", "stop", "关闭", "停止")
    )
    for item in items.values():
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "property" or not item.get("writeable"):
            continue
        for option in item.get("value_list") or []:
            if not isinstance(option, dict):
                continue
            text = _normalize_text(str(option.get("description") or ""))
            if any(_normalize_text(term) in text for term in action_terms):
                return {"iid": item["iid"], "value": option.get("value"), "item": item}
    return None


def _select_action_item(items: dict[str, Any], action: str) -> dict[str, Any] | None:
    if action not in ("turn_on", "turn_off"):
        return None
    terms = _TURN_ON_WORDS if action == "turn_on" else _TURN_OFF_WORDS
    for item in items.values():
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "action" or not item.get("writeable"):
            continue
        text = _normalize_text(
            " ".join(str(item.get(key) or "") for key in ("name", "description", "service"))
        )
        if any(_normalize_text(term) in text for term in terms):
            return {"iid": item["iid"], "value": [], "item": item}
    return None


def _infer_area(devices: list[dict[str, Any]], request: str) -> str:
    request_norm = _normalize_text(request)
    best_score = 0
    best_areas: set[str] = set()
    for device in devices:
        for key in ("room_name", "home_name"):
            name = str(device.get(key) or "")
            if not name:
                continue
            score = _area_reference_score(name, request_norm)
            if score > best_score:
                best_score = score
                best_areas = {name}
            elif score == best_score and score > 0:
                best_areas.add(name)
    return next(iter(best_areas)) if best_score >= 70 and len(best_areas) == 1 else ""


def _area_match_score(query: str, target: str) -> int:
    query_norm = _normalize_text(query)
    target_norm = _normalize_text(target)
    if not query_norm or not target_norm:
        return 0
    score = _text_match_score(query_norm, target_norm)
    query_aliases = _area_alias_terms(query_norm)
    target_aliases = _area_alias_terms(target_norm)
    if query_norm in target_aliases or target_norm in query_aliases:
        score = max(score, 95)
    if query_aliases & target_aliases:
        score = max(score, 90)
    return score


def _area_reference_score(area: str, request_norm: str) -> int:
    area_norm = _normalize_text(area)
    if not area_norm or not request_norm:
        return 0
    if area_norm in request_norm:
        return 100
    aliases = _area_alias_terms(area_norm)
    if any(alias and alias in request_norm for alias in aliases):
        return 92
    return _text_match_score(area_norm, request_norm)


def _area_alias_terms(value: str) -> set[str]:
    normalized = _normalize_text(value)
    aliases = {normalized} if normalized else set()
    for group in _AREA_ALIAS_GROUPS:
        normalized_group = {_normalize_text(item) for item in group if item}
        if normalized in normalized_group:
            aliases.update(normalized_group)
    return aliases


def _infer_device_query(request: str, area: str, action: str) -> str:
    text = str(request or "")
    for value in (area, *_TURN_ON_WORDS, *_TURN_OFF_WORDS):
        if value:
            text = text.replace(str(value), "")
    for filler in (
        "帮我",
        "请",
        "把",
        "给我",
        "一下",
        "的",
        "设备",
        "米家",
        "查一下",
        "查询",
        "看一下",
        "看看",
        "我们",
        "家里",
        "家中",
        "家里的",
        "家中的",
        "显示的",
        "显示",
        "状态",
        "全部",
        "所有",
        "全都",
        "每个",
        "亮度",
        "温度",
        "模式",
        "调到",
        "调成",
        "设置",
        "设为",
        "开到",
        "百分比",
        "百分之",
        "空气质量",
        "空气",
        "质量",
    ):
        text = text.replace(filler, "")
    return text.strip()


def _infer_property_query(request: str) -> str:
    request_norm = _normalize_text(request)
    if _looks_like_air_quality_query(request_norm):
        return "air quality pm2.5 aqi"
    if "温度" in request_norm or "temperature" in request_norm:
        return "temperature"
    if "湿度" in request_norm or "humidity" in request_norm:
        return "humidity"
    if any(term in request_norm for term in ("开关", "开着", "关着", "power", "switch", "on")):
        return "power on switch"
    return request


def _property_aliases_for_query(query_norm: str) -> tuple[str, ...]:
    if _looks_like_air_quality_query(query_norm):
        return (
            "air quality",
            "air-quality",
            "aqi",
            "pm2.5",
            "pm25",
            "pm2p5",
            "pm10",
            "density",
            "particulate matter",
            "空气质量",
            "空气",
            "颗粒物",
        )
    if "温度" in query_norm or "temperature" in query_norm:
        return ("temperature", "温度")
    if "湿度" in query_norm or "humidity" in query_norm:
        return ("humidity", "湿度")
    if any(term in query_norm for term in ("开关", "开着", "关着", "power", "switch", "on")):
        return ("power", "switch", "on", "开关")
    return ()


def _looks_like_air_quality_query(query_norm: str) -> bool:
    return any(
        term in query_norm
        for term in (
            "空气质量",
            "空气",
            "质量",
            "aqi",
            "pm2.5",
            "pm25",
            "pm2p5",
            "颗粒物",
            "particulate",
            "airquality",
        )
    )


def _unsupported_miot_operation_response(command: dict[str, Any]) -> dict[str, Any] | None:
    operation = str(command.get("operation") or "")
    if operation == "schedule":
        message = (
            "\u6211\u73b0\u5728\u8fd8\u4e0d\u80fd"
            "\u5b9a\u65f6\u63a7\u5236\u7c73\u5bb6\u8bbe\u5907\u3002"
        )
        _log_miot_resolver(command, "unsupported", 0, reason="schedule_unsupported")
        return {
            "status": "unsupported",
            "decision": "unsupported",
            "operation": operation,
            "message": message,
            "direct_response": message,
            "query": command,
        }
    if operation == "scene":
        message = "\u6211\u73b0\u5728\u8fd8\u4e0d\u80fd\u6267\u884c\u7c73\u5bb6\u573a\u666f\u3002"
        _log_miot_resolver(command, "unsupported", 0, reason="scene_unsupported")
        return {
            "status": "unsupported",
            "decision": "unsupported",
            "operation": operation,
            "message": message,
            "direct_response": message,
            "query": command,
        }
    return None


def _infer_miot_operation(request: str, action: str) -> str:
    normalized = _normalize_text(request)
    if action and _looks_like_scheduled_miot_request(normalized):
        return "schedule"
    if _looks_like_scene_miot_request(normalized):
        return "scene"
    if action:
        return "control"
    return ""


def _looks_like_scheduled_miot_request(normalized: str) -> bool:
    if any(
        term in normalized
        for term in (
            "\u5b9a\u65f6",
            "\u9884\u7ea6",
            "\u7a0d\u540e",
            "\u4e00\u4f1a\u513f",
            "\u5f85\u4f1a\u513f",
            "\u8fc7\u4f1a\u513f",
            "\u5230\u70b9",
        )
    ):
        return True
    relative_pattern = (
        r"(?:\d+|[\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d"
        r"\u4e03\u516b\u4e5d\u5341\u534a]+)"
        r"(?:\u79d2|\u5206\u949f|\u5206|\u4e2a\u5c0f\u65f6|\u5c0f\u65f6|\u5929)"
        r"\u540e"
    )
    if re.search(relative_pattern, normalized):
        return True
    clock_pattern = (
        r"(?:\u4eca\u5929|\u660e\u5929|\u540e\u5929|\u4eca\u665a|\u660e\u65e9|"
        r"\u65e9\u4e0a|\u4e0a\u5348|\u4e2d\u5348|\u4e0b\u5348|\u665a\u4e0a)?"
        r"(?:\d+|[\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03"
        r"\u516b\u4e5d\u5341]+)\u70b9"
    )
    return re.search(clock_pattern, normalized) is not None


def _looks_like_scene_miot_request(normalized: str) -> bool:
    return any(term in normalized for term in ("\u573a\u666f", "\u81ea\u52a8\u5316", "scene"))


def _is_group_control_command(command: dict[str, Any]) -> bool:
    text = _normalize_text(
        " ".join(str(command.get(key) or "") for key in ("request", "device"))
    )
    return _has_miot_group_quantifier(text)


def _has_miot_group_quantifier(text: str) -> bool:
    if any(
        term in text
        for term in (
            "\u5168\u90e8",
            "\u6240\u6709",
            "\u5168\u90fd",
            "\u6bcf\u4e2a",
            "all",
        )
    ):
        return True
    if "\u90fd" not in text:
        return False
    action_terms = {
        _normalize_text(word)
        for word in (
            *_TURN_ON_WORDS,
            *_TURN_OFF_WORDS,
            "\u8bbe\u7f6e",
            "\u8bbe\u4e3a",
            "\u8c03\u5230",
            "\u8c03\u6210",
        )
        if _normalize_text(word)
    }
    if any(f"\u90fd{term}" in text for term in action_terms):
        return True
    device_terms: set[str] = {
        "\u8bbe\u5907",
        "\u5bb6\u91cc",
        "\u7c73\u5bb6",
    }
    for aliases in _DEVICE_CLASS_ALIASES.values():
        device_terms.update(_normalize_text(alias) for alias in aliases if alias)
    return any(term and f"{term}\u90fd" in text for term in device_terms)


def _state_filter_for_group_control(command: dict[str, Any]) -> bool | None:
    text = _normalize_text(
        " ".join(str(command.get(key) or "") for key in ("request", "device"))
    )
    action = str(command.get("action") or "")
    if action == "turn_off" and any(
        term in text for term in ("开着", "亮着", "没关", "没有关", "还开", "未关")
    ):
        return True
    if action == "turn_on" and any(
        term in text for term in ("关着", "没开", "没有开", "还关", "未开")
    ):
        return False
    return None


def _target_power_state_for_control(command: dict[str, Any]) -> bool | None:
    action = str(command.get("action") or "")
    command_text = _normalize_text(
        " ".join(str(command.get(key) or "") for key in ("request", "device"))
    )
    if any(term in command_text for term in ("全部", "所有", "全都", "每个", "all")):
        return None
    if action == "turn_off":
        return True
    if action == "turn_on":
        return False
    return None


def _target_power_state_description(command: dict[str, Any]) -> str:
    target_state = _target_power_state_for_control(command)
    if target_state is True:
        return "开着"
    if target_state is False:
        return "关着"
    return "符合状态"


def _looks_like_power_property_query(command: dict[str, Any]) -> bool:
    text = _normalize_text(
        " ".join(str(command.get(key) or "") for key in ("request", "property"))
    )
    return any(
        term in text
        for term in (
            "开关",
            "开着",
            "关着",
            "开没开",
            "关没关",
            "亮着",
            "哪个",
            "power",
            "switch",
            "on",
        )
    )


def _format_group_control_response(
    command: dict[str, Any],
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> str:
    action_text = _group_action_done_text(str(command.get("action") or ""))
    if dry_run:
        action_text = "可处理"
    success_count = len(successes)
    failure_count = len(failures)
    if success_count and not failure_count:
        return f"好的，已{action_text}{success_count}个设备。"
    if success_count and failure_count:
        failed_names = [
            str((failure.get("device") or {}).get("name") or "设备")
            for failure in failures
            if isinstance(failure, dict)
        ]
        return (
            f"已{action_text}{success_count}个设备，"
            f"{failure_count}个失败：{'、'.join(failed_names[:3])}。"
        )
    if failure_count:
        return f"没有成功控制设备，{failure_count}个失败。"
    return "没有需要处理的设备。"


def _group_action_done_text(action: str) -> str:
    if action == "turn_on":
        return "打开"
    if action == "turn_off":
        return "关闭"
    if action == "set_value":
        return "设置"
    return "处理"


def _format_group_power_read_response(
    command: dict[str, Any],
    readings: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    del failures
    text = _normalize_text(str(command.get("request") or ""))
    on_names = [
        str((reading.get("device") or {}).get("name") or "设备")
        for reading in readings
        if miot_values_equal(True, reading.get("value"))
    ]
    off_names = [
        str((reading.get("device") or {}).get("name") or "设备")
        for reading in readings
        if miot_values_equal(False, reading.get("value"))
    ]
    if "哪个" in text or "开着" in text or "亮着" in text:
        if on_names:
            return "开着的是" + "、".join(on_names[:5]) + "。"
        return "没有发现开着的设备。"
    if "关着" in text:
        if off_names:
            return "关着的是" + "、".join(off_names[:5]) + "。"
        return "没有发现关着的设备。"
    parts = []
    if on_names:
        parts.append("开：" + "、".join(on_names[:5]))
    if off_names:
        parts.append("关：" + "、".join(off_names[:5]))
    return "；".join(parts) + "。" if parts else "没有读到设备开关状态。"


def _log_miot_resolver(
    command: dict[str, Any],
    decision: str,
    candidate_count: int,
    *,
    reason: str,
    success_count: int | None = None,
    failure_count: int | None = None,
) -> None:
    params: dict[str, Any] = {
        "decision": decision,
        "reason": reason,
        "candidate_count": candidate_count,
        "action": str(command.get("action") or ""),
        "area": str(command.get("area") or ""),
        "device": str(command.get("device") or ""),
        "device_class": str(command.get("device_class") or ""),
    }
    if success_count is not None:
        params["success_count"] = success_count
    if failure_count is not None:
        params["failure_count"] = failure_count
    log_event("miot", "resolver", log_id="miot.resolver", **params)


def _format_property_read_response(
    device: dict[str, Any],
    item: dict[str, Any],
    value: Any,
) -> str:
    name = str(device.get("name") or "设备")
    description = str(item.get("description") or item.get("name") or "状态")
    unit = str(item.get("unit") or "")
    value_text = _format_property_value(value)
    if unit and unit not in value_text:
        value_text = f"{value_text}{unit}"
    return f"{name}的{description}是{value_text}。"


def _format_property_value(value: Any) -> str:
    if isinstance(value, bool):
        return "开" if value else "关"
    return str(value)


def _infer_control_value(request: str) -> Any:
    text = _normalize_text(request)
    if any(term in text for term in ("一半", "半开", "开半", "开到半")):
        return 50

    percent_match = re.search(r"(?:百分之|百分比)?(\d+(?:\.\d+)?)%", str(request or ""))
    if percent_match:
        return float(percent_match.group(1))
    percent_text_match = re.search(r"百分之\s*(\d+(?:\.\d+)?)", str(request or ""))
    if percent_text_match:
        return float(percent_text_match.group(1))

    number_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:度|℃|%|百分比)?", str(request or ""))
    if number_match and any(
        term in text
        for term in (
            "温度",
            "空调",
            "冷气",
            "度",
            "℃",
            "亮度",
            "调亮",
            "调暗",
            "窗帘",
            "开到",
            "调到",
            "百分",
        )
    ):
        value = float(number_match.group(1))
        return int(value) if value.is_integer() else value

    mode_terms = (
        ("cool", ("制冷", "冷风", "冷气模式")),
        ("heat", ("制热", "暖风", "热风")),
        ("dry", ("除湿", "抽湿")),
        ("fan", ("送风", "通风")),
        ("auto", ("自动",)),
        ("sleep", ("睡眠",)),
        ("high", ("高风", "强风", "大风")),
        ("medium", ("中风", "标准风")),
        ("low", ("低风", "弱风", "小风")),
    )
    for value, terms in mode_terms:
        if any(term in text for term in terms):
            return value
    return None


def _infer_relative_control_delta(request: str) -> float | None:
    text = _normalize_text(request)
    decrease_terms = ("降低", "降", "减少", "减", "调低", "调小", "低一点", "小一点")
    increase_terms = ("升高", "提高", "增加", "调高", "调大", "高一点", "大一点")
    sign = -1 if any(term in text for term in decrease_terms) else 0
    if sign == 0 and any(term in text for term in increase_terms):
        sign = 1
    if sign == 0:
        return None

    number_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:度|℃|%|百分比)?", str(request or ""))
    if number_match is None:
        return None
    return sign * float(number_match.group(1))


def _infer_control_property_query(request: str, value: Any, action: str) -> str:
    text = _normalize_text(request)
    if isinstance(value, str):
        if value in {"cool", "heat", "dry", "fan", "auto", "sleep"}:
            return "mode"
        if value in {"high", "medium", "low"}:
            return "fan level speed"
    if any(term in text for term in ("亮度", "调亮", "调暗", "暗一点", "亮一点")):
        return "brightness"
    if "温度" in text or "度" in text or "℃" in text:
        return "temperature"
    if any(term in text for term in ("窗帘", "帘", "开到", "开一半", "一半", "百分")):
        return "curtain position percentage"
    if "模式" in text:
        return "mode"
    if action == "set_value":
        return request
    return ""


def _control_property_aliases_for_query(query_norm: str) -> tuple[str, ...]:
    if "temperature" in query_norm or "温度" in query_norm or "度" in query_norm:
        return ("target temperature", "temperature", "温度")
    if "brightness" in query_norm or "亮度" in query_norm or "调亮" in query_norm:
        return ("brightness", "bright", "亮度")
    if any(term in query_norm for term in ("curtain", "position", "percentage", "窗帘", "百分")):
        return ("current position", "target position", "position", "percentage", "curtain", "位置")
    if "mode" in query_norm or "模式" in query_norm:
        return ("mode", "模式")
    if "fan" in query_norm or "speed" in query_norm or "风" in query_norm:
        return ("fan level", "fan speed", "speed", "风速", "风量")
    return (query_norm,) if query_norm else ()


def _infer_action(action: str, request: str, value: Any) -> str:
    action_text = _normalize_text(action)
    request_text = _normalize_text(request)
    if value is not None and not isinstance(value, bool):
        return "set_value"
    if isinstance(value, bool):
        return "turn_on" if value else "turn_off"
    if any(term in request_text for term in ("调到", "设置", "设为", "调成", "开到")):
        return "set_value"
    if any(_normalize_text(word) in action_text for word in _TURN_ON_WORDS):
        return "turn_on"
    if any(_normalize_text(word) in action_text for word in _TURN_OFF_WORDS):
        return "turn_off"
    if any(_normalize_text(word) in request_text for word in _TURN_ON_WORDS):
        return "turn_on"
    if any(_normalize_text(word) in request_text for word in _TURN_OFF_WORDS):
        return "turn_off"
    return action.strip()


def _infer_device_class(*values: str) -> str:
    text = _normalize_text(" ".join(str(value or "") for value in values))
    if not text:
        return ""
    for device_class, aliases in _DEVICE_CLASS_ALIASES.items():
        if any(_normalize_text(alias) in text for alias in aliases):
            return device_class
    return ""


def _device_class_matches(query: str, device_class: str) -> bool:
    query_class = _infer_device_class(query)
    query_norm = _normalize_text(query_class or query)
    class_norm = _normalize_text(device_class)
    if query_norm == class_norm:
        return True
    aliases = _DEVICE_CLASS_ALIASES.get(device_class, ())
    return any(_normalize_text(alias) == query_norm for alias in aliases)


def _text_match_score(query: str, target: str) -> int:
    query_norm = _normalize_text(query)
    target_norm = _normalize_text(target)
    if not query_norm or not target_norm:
        return 0
    if query_norm == target_norm:
        return 100
    if query_norm in target_norm:
        return 86
    if target_norm in query_norm:
        return 78
    ratio = difflib.SequenceMatcher(None, query_norm, target_norm).ratio()
    return int(ratio * 70) if ratio >= 0.55 else 0


def _normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[\s,，。.!！?？:：;；'\"“”‘’（）()\\[\\]{}<>《》、_-]+", "", text)
    return _collapse_repeated_cjk(text)


def _collapse_repeated_cjk(value: str) -> str:
    result: list[str] = []
    previous = ""
    for char in value:
        if char == previous and "\u4e00" <= char <= "\u9fff":
            continue
        result.append(char)
        previous = char
    return "".join(result)


def _public_device(device: dict[str, Any]) -> dict[str, Any]:
    return {
        "did": device.get("did"),
        "name": device.get("name"),
        "online": device.get("online"),
        "room_name": device.get("room_name"),
        "home_name": device.get("home_name"),
        "device_class": device.get("device_class"),
        "model": device.get("model"),
    }


def _public_match_candidates(
    matches: list[tuple[int, dict[str, Any]]],
    command: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for score, device in matches:
        public = _public_device(device)
        public["match_score"] = score
        public["hit_fields"] = _match_hit_fields(device, command)
        candidates.append(public)
    return candidates


def _match_hit_fields(device: dict[str, Any], command: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    area_query = str(command.get("area") or "")
    if area_query and max(
        _area_match_score(area_query, str(device.get("room_name") or "")),
        _area_match_score(area_query, str(device.get("home_name") or "")),
    ) > 0:
        fields.append("area")

    class_query = str(command.get("device_class") or "")
    if class_query:
        if _device_class_matches(class_query, str(device.get("device_class") or "")):
            fields.append("device_class")
        elif _text_match_score(class_query, str(device.get("name") or "")) > 0:
            fields.append("name")

    device_query = str(command.get("device") or "")
    if device_query:
        scored_fields = (
            ("name", _text_match_score(device_query, str(device.get("name") or ""))),
            (
                "device_class",
                _text_match_score(device_query, str(device.get("device_class") or "")),
            ),
            ("model", _text_match_score(device_query, str(device.get("model") or ""))),
        )
        best_score = max(score for _field, score in scored_fields)
        fields.extend(field for field, score in scored_fields if score > 0 and score == best_score)
        inferred_class = _infer_device_class("", device_query, "")
        if inferred_class and _device_class_matches(
            inferred_class,
            str(device.get("device_class") or ""),
        ):
            fields.append("device_class")

    return sorted(set(fields))


def _token_from_mapping(data: Any) -> XiaomiMiotToken:
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        data = data["result"]
    if not isinstance(data, dict):
        raise XiaomiMiotError("Invalid Xiaomi MIoT token JSON")
    access_token = str(data.get("access_token") or "")
    if not access_token:
        raise XiaomiMiotError("Xiaomi MIoT token JSON is missing access_token")
    return XiaomiMiotToken(
        access_token=access_token,
        refresh_token=str(data.get("refresh_token") or ""),
        expires_ts=_safe_int(data.get("expires_ts"), default=0),
        user_info=data.get("user_info") if isinstance(data.get("user_info"), dict) else None,
    )


def _load_or_create_uuid(config: XiaomiMiotConfig) -> str:
    env_uuid = os.getenv(config.uuid_env)
    if env_uuid:
        return env_uuid
    uuid_path = _resolve_path(config.uuid_file)
    if uuid_path.exists():
        value = uuid_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = uuid.uuid4().hex
    uuid_path.parent.mkdir(parents=True, exist_ok=True)
    uuid_path.write_text(value, encoding="utf-8")
    return value


def _api_host(cloud_server: str) -> str:
    server = str(cloud_server or "cn").strip()
    return OAUTH2_API_HOST_DEFAULT if server == "cn" else f"{server}.{OAUTH2_API_HOST_DEFAULT}"


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def _cache_path(config: XiaomiMiotConfig, urn: str) -> Path:
    digest = hashlib.sha1(urn.encode("utf-8")).hexdigest()
    return _resolve_path(config.cache_dir) / f"spec_{digest}.json"


def _load_cached_spec(config: XiaomiMiotConfig, urn: str) -> dict[str, Any] | None:
    path = _cache_path(config, urn)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _save_cached_spec(config: XiaomiMiotConfig, urn: str, spec: dict[str, Any]) -> None:
    path = _cache_path(config, urn)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def _merge_home_room_pages(
    target: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]],
) -> None:
    for home_id, home in source.items():
        target.setdefault(
            home_id,
            {
                "home_id": home_id,
                "home_name": home.get("home_name", ""),
                "share_home": home.get("share_home", False),
                "uid": home.get("uid", ""),
                "dids": [],
                "room_list": {},
            },
        )
        target[home_id].setdefault("dids", [])
        target[home_id]["dids"].extend(home.get("dids", []) or [])
        target[home_id].setdefault("room_list", {})
        for room_id, room in (home.get("room_list") or {}).items():
            target[home_id]["room_list"].setdefault(
                room_id,
                {
                    "room_id": room_id,
                    "room_name": room.get("room_name", ""),
                    "dids": [],
                },
            )
            target[home_id]["room_list"][room_id]["dids"].extend(room.get("dids", []) or [])


def _expect_mapping(value: Any, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise XiaomiMiotError(message)
    return value


def _first_result(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result")
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return result[0]
    if isinstance(result, dict):
        return result
    raise XiaomiMiotError("MIoT response is missing result")


def _get_json(url: str, params: dict[str, Any], timeout: float) -> Any:
    return json.loads(_http_get_text(url, params=params, headers={}, timeout=timeout))


def _http_get_text(
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> str:
    full_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    request = urllib.request.Request(full_url, headers=headers, method="GET")
    return _http_request_text(request, timeout=timeout, error_label=url)


def _http_request_text(
    request: urllib.request.Request,
    timeout: float,
    error_label: str,
) -> str:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise XiaomiMiotError(f"{error_label} failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise XiaomiMiotError(f"{error_label} failed: {exc.reason}") from exc


def _normalize_pem(pem: str) -> bytes:
    body = (
        pem.replace("-----BEGIN PUBLIC KEY-----", "")
        .replace("-----END PUBLIC KEY-----", "")
        .strip()
    )
    lines = [body[index : index + 64] for index in range(0, len(body), 64)]
    normalized = "-----BEGIN PUBLIC KEY-----\n" + "\n".join(lines) + "\n-----END PUBLIC KEY-----\n"
    return normalized.encode("utf-8")


def _urn_name(type_value: str) -> str:
    parts = str(type_value or "").split(":")
    return parts[3] if len(parts) > 3 and parts[3] else type_value


def _parse_value_range(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list) and len(value) >= 3:
        return {"min": value[0], "max": value[1], "step": value[2]}
    return None


def _parse_value_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "value": item.get("value"),
                "description": item.get("description"),
            }
        )
    return result


def _required_text(arguments: dict[str, Any], key: str) -> str:
    value = str(arguments.get(key) or "").strip()
    if not value:
        raise XiaomiMiotError(f"{key} is required")
    return value


def _optional_text(arguments: dict[str, Any], key: str) -> str | None:
    value = str(arguments.get(key) or "").strip()
    return value or None


def _optional_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on", "ok"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return bool(value)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
