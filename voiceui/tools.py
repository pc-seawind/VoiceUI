from __future__ import annotations

import array
import html
import io
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from voiceui.http_utils import optional_api_key, post_json, require_api_key
from voiceui.llm import ChatClient, ChatMessage, ToolCall
from voiceui.logs import log_continuous, log_event
from voiceui.models import AssistantConfig, MusicConfig, SearchConfig
from voiceui.system_volume import get_system_output_volume, set_system_output_volume
from voiceui.xiaomi_miot import XiaomiMiotController

ToolHandler = Callable[[dict[str, Any]], Any]


def _weather_place(
    name: str,
    country: str,
    admin1: str,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "country": country,
        "admin1": admin1,
        "latitude": latitude,
        "longitude": longitude,
    }


_WEATHER_CACHE_TTL_SECONDS = 600
_WEATHER_CACHE_PATH = Path(".voiceui/weather_cache.json")
_WEATHER_DISK_CACHE_LOADED = False
_WEATHER_GEOCODE_CACHE: dict[str, dict[str, Any]] = {}
_WEATHER_FORECAST_CACHE: dict[str, dict[str, Any]] = {}
_MIOT_PENDING_CONTEXT_TTL_SECONDS = 90.0

_COMMON_WEATHER_LOCATIONS: dict[str, dict[str, Any]] = {
    "上海": _weather_place("上海", "中国", "上海市", 31.2304, 121.4737),
    "上海市": _weather_place("上海", "中国", "上海市", 31.2304, 121.4737),
    "北京": _weather_place("北京", "中国", "北京市", 39.9042, 116.4074),
    "北京市": _weather_place("北京", "中国", "北京市", 39.9042, 116.4074),
    "北京昌平": _weather_place("北京昌平", "中国", "北京市", 40.2207, 116.2312),
    "北京市昌平区": _weather_place("北京昌平", "中国", "北京市", 40.2207, 116.2312),
    "昌平": _weather_place("北京昌平", "中国", "北京市", 40.2207, 116.2312),
    "昌平区": _weather_place("北京昌平", "中国", "北京市", 40.2207, 116.2312),
    "广州": _weather_place("广州", "中国", "广东", 23.1291, 113.2644),
    "深圳": _weather_place("深圳", "中国", "广东", 22.5431, 114.0579),
    "杭州": _weather_place("杭州", "中国", "浙江", 30.2741, 120.1551),
    "南京": _weather_place("南京", "中国", "江苏", 32.0603, 118.7969),
    "苏州": _weather_place("苏州", "中国", "江苏", 31.2989, 120.5853),
    "成都": _weather_place("成都", "中国", "四川", 30.5728, 104.0668),
    "重庆": _weather_place("重庆", "中国", "重庆市", 29.5630, 106.5516),
    "武汉": _weather_place("武汉", "中国", "湖北", 30.5928, 114.3055),
    "西安": _weather_place("西安", "中国", "陕西", 34.3416, 108.9398),
}

_TOOL_USE_INSTRUCTIONS = (
    "Use the available tools whenever the user asks for live data or side effects. "
    "For web search, current time, current weather, playback volume, mute, music "
    "search, music playback, stopping music, or Xiaomi Home device state/control, "
    "call the matching tool instead of guessing. For natural Xiaomi Home commands "
    "such as turning a room light on or off, prefer xiaomi_miot_control_device. "
    "For natural Xiaomi Home state questions, such as checking an air purifier's "
    "air quality, prefer xiaomi_miot_read_device_property and do not use web_search. "
    "If exactly one low-risk "
    "device matches, execute it without asking a second confirmation; ask only when "
    "the tool reports ambiguity or the target is sensitive. Never say a Xiaomi Home "
    "device was opened, closed, or changed unless a Xiaomi Home control tool returned "
    "success for that request. After web_search returns, answer the user's question "
    "from the returned answer/results instead of merely reading search result titles; "
    "say when the results do not contain a clear answer. Do not say music is playing "
    "unless play_music returned successfully."
)
_ROUTED_TOOL_NAMES = {
    "get_current_time",
    "get_current_weather",
    "web_search",
    "get_system_volume",
    "set_system_volume",
    "search_music",
    "play_music",
    "stop_music",
    "xiaomi_miot_auth_url",
    "xiaomi_miot_exchange_auth_code",
    "xiaomi_miot_control_device",
    "xiaomi_miot_get_area_info",
    "xiaomi_miot_get_device_classes",
    "xiaomi_miot_get_devices",
    "xiaomi_miot_get_device_spec",
    "xiaomi_miot_read_device_property",
    "xiaomi_miot_get_property",
    "xiaomi_miot_control",
}


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class MusicTrack:
    title: str
    artist: str
    playback_url: str
    provider: str
    server: str
    cover_url: str = ""
    lyric_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "artist": self.artist,
            "provider": self.provider,
            "server": self.server,
            "cover_url": self.cover_url,
            "lyric_url": self.lyric_url,
        }


class MusicPlaybackController:
    def __init__(
        self,
        config: MusicConfig,
        fallback_device: str | int | None = None,
        fallback_sample_rate: int | None = None,
        fallback_channels: int | None = None,
    ):
        self.config = config
        self.playback_device = (
            config.playback_device if config.playback_device is not None else fallback_device
        )
        self.playback_sample_rate = (
            config.playback_sample_rate
            if config.playback_sample_rate is not None
            else fallback_sample_rate
        )
        self.playback_channels = (
            config.playback_channels if config.playback_channels is not None else fallback_channels
        )
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._current_track: MusicTrack | None = None
        self._last_error = ""
        self._duck_reasons: set[str] = set()

    def search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _required_text(arguments, "query")
        server = _music_server(arguments, self.config)
        tracks = search_music_tracks(self.config, query, server=server)
        return {
            "provider": self.config.provider,
            "server": server,
            "tracks": [track.to_dict() for track in tracks],
        }

    def play(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = _required_text(arguments, "query")
        server = _music_server(arguments, self.config)
        tracks = search_music_tracks(self.config, query, server=server)
        if not tracks:
            raise RuntimeError(f"Could not find music for query: {query}")

        track = tracks[0]
        if not self.config.playback_enabled:
            return {
                "status": "resolved",
                "playback_enabled": False,
                "track": track.to_dict(),
            }

        self.stop(wait=False)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._download_and_play,
            args=(track, stop_event),
            name="voiceui-music-playback",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
            self._stop_event = stop_event
            self._current_track = track
            self._last_error = ""
        thread.start()
        return {
            "status": "starting",
            "playback_enabled": True,
            "track": track.to_dict(),
        }

    def stop(self, arguments: dict[str, Any] | None = None, wait: bool = False) -> dict[str, Any]:
        del arguments
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            track = self._current_track
        if stop_event is None:
            return {"status": "idle"}

        stop_event.set()
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        still_alive = bool(thread is not None and thread.is_alive())
        if not still_alive:
            with self._lock:
                if self._stop_event is stop_event:
                    self._stop_event = None
                    self._thread = None
                    self._current_track = None
        return {
            "status": "stopping" if still_alive else "stopped",
            "track": track.to_dict() if track is not None else None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            track = self._current_track
            error = self._last_error
        return {
            "status": "playing" if thread is not None and thread.is_alive() else "idle",
            "track": track.to_dict() if track is not None else None,
            "last_error": error,
        }

    def is_active(self) -> bool:
        with self._lock:
            thread = self._thread
            return bool(thread is not None and thread.is_alive())

    def duck(self, reason: str = "assistant") -> None:
        with self._lock:
            was_ducked = bool(self._duck_reasons)
            self._duck_reasons.add(reason)
            should_log = not was_ducked and self._thread is not None and self._thread.is_alive()
            factor = self._ducking_factor()
        if should_log:
            log_event(
                "music",
                "ducked",
                log_id="music.ducked",
                factor=f"{factor:.3f}",
                reason=reason,
            )

    def unduck(self, reason: str = "assistant") -> None:
        with self._lock:
            was_ducked = bool(self._duck_reasons)
            self._duck_reasons.discard(reason)
            is_ducked = bool(self._duck_reasons)
            should_log = (
                was_ducked
                and not is_ducked
                and self._thread is not None
                and self._thread.is_alive()
            )
        if should_log:
            log_event("music", "restored", log_id="music.restored", reason=reason)

    def current_volume_factor(self) -> float:
        with self._lock:
            return self._ducking_factor() if self._duck_reasons else 1.0

    def _ducking_factor(self) -> float:
        return max(0.0, min(1.0, float(self.config.ducking_volume_factor)))

    def _download_and_play(self, track: MusicTrack, stop_event: threading.Event) -> None:
        try:
            if self.config.start_delay_seconds > 0 and stop_event.wait(
                self.config.start_delay_seconds
            ):
                return
            audio_data, audio_format, final_url = _download_audio(
                track.playback_url,
                timeout=self.config.timeout_seconds,
                max_bytes=self.config.max_audio_bytes,
            )
            log_event(
                "music",
                "starting",
                log_id="music.starting",
                title=track.title,
                artist=track.artist,
                format=audio_format,
                url=final_url,
            )
            _play_decoded_audio(
                audio_data,
                audio_format=audio_format,
                device=self.playback_device,
                playback_sample_rate=self.playback_sample_rate,
                playback_channels=self.playback_channels,
                playback_volume=self.config.playback_volume,
                limiter_enabled=self.config.limiter_enabled,
                limiter_threshold=self.config.limiter_threshold,
                dynamic_volume_getter=self.current_volume_factor,
                stop_event=stop_event,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            with self._lock:
                self._last_error = str(exc)
            log_event(
                "music",
                "playback_error",
                log_id="music.playback_error",
                error=exc,
            )
        finally:
            with self._lock:
                if self._stop_event is stop_event:
                    self._stop_event = None
                    self._thread = None
                    self._current_track = None


class VoiceToolRunner:
    def __init__(
        self,
        chat: ChatClient,
        tools: list[ToolDefinition],
        max_iterations: int = 4,
        music_controller: MusicPlaybackController | None = None,
    ):
        self.chat = chat
        self.tools = {tool.name: tool for tool in tools}
        self.tool_payloads = [tool.to_openai_tool() for tool in tools]
        self.max_iterations = max(1, max_iterations)
        self.music_controller = music_controller
        self._last_miot_control: dict[str, Any] | None = None
        self._last_miot_ambiguity: dict[str, Any] | None = None
        self._previous_miot_ambiguity: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.tools)

    def _active_miot_ambiguity(self) -> dict[str, Any] | None:
        if self._last_miot_ambiguity is None:
            return None
        if _miot_context_is_fresh(self._last_miot_ambiguity):
            return self._last_miot_ambiguity
        self._last_miot_ambiguity = None
        return None

    def _active_previous_miot_ambiguity(self) -> dict[str, Any] | None:
        if self._previous_miot_ambiguity is None:
            return None
        if _miot_context_is_fresh(self._previous_miot_ambiguity):
            return self._previous_miot_ambiguity
        self._previous_miot_ambiguity = None
        return None

    def _can_resolve_active_miot_ambiguity(self, text: str) -> bool:
        ambiguity = self._active_miot_ambiguity()
        if not ambiguity:
            return False
        candidates = (
            ambiguity.get("candidates")
            if isinstance(ambiguity.get("candidates"), list)
            else []
        )
        if _select_miot_ambiguity_candidate(text, candidates) is not None:
            return True
        action = _infer_miot_action_from_text(text)
        return bool(
            action
            and (
                _is_miot_followup_reference(text)
                or _is_miot_group_followup_reference(text)
            )
        )

    def _looks_like_last_miot_property_followup(self, text: str) -> bool:
        if not self._last_miot_control or self._active_miot_ambiguity() is not None:
            return False
        if _infer_miot_action_from_text(text) != "set_value":
            return False
        normalized = text.lower().replace(" ", "")
        if _has_explicit_miot_device_text(normalized):
            return False
        if any(term in normalized for term in ("音乐", "歌曲", "播放", "music", "音量")):
            return False
        return any(
            term in normalized
            for term in (
                "调成",
                "调到",
                "设置",
                "设为",
                "调高",
                "调低",
                "亮度",
                "模式",
                "制冷",
                "制热",
                "除湿",
                "睡眠",
                "百分",
            )
        ) or ("度" in normalized and any(char.isdigit() for char in normalized))

    def complete(self, messages: list[ChatMessage]) -> str:
        followup_call = self._build_miot_followup_call(messages)
        if followup_call is not None:
            result = self._execute_tool_call(followup_call)
            return _direct_tool_response([result]) or _format_tool_payload_response(result)

        selected_tool_payloads = self._select_tool_payloads(messages)
        if not selected_tool_payloads:
            llm_started = time.monotonic()
            response = self.chat.complete(messages)
            llm_ms = int((time.monotonic() - llm_started) * 1000)
            log_event(
                "tools",
                "llm_round",
                log_id="tools.llm_round",
                round=1,
                llm_call_ms=llm_ms,
                tools_sent=0,
                tool_calls=0,
            )
            return response.strip()

        working_messages = _with_tool_use_instructions(messages)
        for iteration in range(self.max_iterations):
            llm_started = time.monotonic()
            response = self.chat.complete_with_tools(working_messages, selected_tool_payloads)
            llm_ms = int((time.monotonic() - llm_started) * 1000)
            log_event(
                "tools",
                "llm_round",
                log_id="tools.llm_round",
                round=iteration + 1,
                llm_call_ms=llm_ms,
                tools_sent=len(selected_tool_payloads),
                tool_calls=len(response.tool_calls),
            )
            if not response.tool_calls:
                last_text = _last_user_text(messages)
                if _requires_tool_call_for_text(last_text, set(self.tools)):
                    direct_call = self._build_direct_miot_control_call(last_text)
                    if direct_call is not None:
                        result = self._execute_tool_call(direct_call)
                        return _direct_tool_response([result]) or _format_tool_payload_response(
                            result
                        )
                    log_event(
                        "tools",
                        "missing_required_tool_call",
                        log_id="tools.missing_required_tool_call",
                        requested_tool="xiaomi_miot_control_device",
                    )
                    return "我还没有实际执行到设备控制，不能确认已经完成。请再说一遍具体设备。"
                return response.content.strip()

            working_messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=[_assistant_tool_call_payload(call) for call in response.tool_calls],
                )
            )
            tool_results: list[dict[str, Any]] = []
            for call in response.tool_calls:
                result = self._execute_tool_call(call)
                tool_results.append(result)
                working_messages.append(
                    ChatMessage(
                        role="tool",
                        content=_to_json_text(result),
                        tool_call_id=call.id,
                    )
                )
            direct_response = _direct_tool_response(tool_results)
            if direct_response:
                return direct_response

        return _fallback_tool_response(working_messages)

    def _select_tool_payloads(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        selected_names = _select_tool_names_for_text(_last_user_text(messages), set(self.tools))
        if not selected_names:
            if not any(name in _ROUTED_TOOL_NAMES for name in self.tools):
                return self.tool_payloads
            return []
        payloads = [
            payload
            for payload in self.tool_payloads
            if payload.get("function", {}).get("name") in selected_names
        ]
        return payloads

    def can_handle_miot_control_text(self, text: str) -> bool:
        return (
            "xiaomi_miot_control_device" in self.tools
            and _looks_like_miot_control_text(text)
        )

    def can_handle_miot_text(self, text: str) -> bool:
        if "xiaomi_miot_control_device" in self.tools and (
            _looks_like_miot_control_text(text)
            or self._looks_like_last_miot_property_followup(text)
            or self._can_resolve_active_miot_ambiguity(text)
        ):
            return True
        return "xiaomi_miot_read_device_property" in self.tools and (
            _looks_like_miot_read_text(text)
            or self._can_resolve_active_miot_ambiguity(text)
        )

    def can_handle_miot_followup_text(self, text: str) -> bool:
        if self._looks_like_last_miot_property_followup(
            text
        ) or self._can_resolve_active_miot_ambiguity(text):
            return True
        return (
            "xiaomi_miot_control_device" in self.tools
            and self._last_miot_control is not None
            and self._active_miot_ambiguity() is None
            and _is_miot_followup_reference(text)
        )

    def run_miot_control(
        self,
        arguments: dict[str, Any],
        *,
        remember: bool = True,
    ) -> dict[str, Any]:
        return self._execute_tool_call(
            ToolCall(
                id="local_miot_control",
                name="xiaomi_miot_control_device",
                arguments=arguments,
            ),
            remember=remember,
        )

    def format_tool_response(self, payload: dict[str, Any]) -> str:
        return _format_tool_payload_response(payload)

    def _execute_tool_call(
        self,
        call: ToolCall,
        *,
        remember: bool = True,
    ) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        if tool is None:
            log_event(
                "tool",
                "executed",
                log_id="tool.executed",
                name=call.name,
                latency_ms=0,
                ok=False,
            )
            return {
                "ok": False,
                "error": f"Unknown tool: {call.name}",
            }

        started = time.monotonic()
        try:
            result = tool.handler(call.arguments)
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(
                "tool",
                "executed",
                log_id="tool.executed",
                name=call.name,
                latency_ms=latency_ms,
                ok=True,
            )
            if isinstance(result, dict):
                payload = {"ok": True, **result} if "ok" not in result else dict(result)
            else:
                payload = {"ok": True, "result": result}
            payload.setdefault("latency_ms", latency_ms)
            payload = self._with_tool_direct_response(call.name, payload)
            if remember:
                self._remember_miot_control(call.name, payload)
            return payload
        except Exception as exc:  # pylint: disable=broad-exception-caught
            latency_ms = int((time.monotonic() - started) * 1000)
            log_event(
                "tool",
                "executed",
                log_id="tool.executed",
                name=call.name,
                latency_ms=latency_ms,
                ok=False,
            )
            return {"ok": False, "error": str(exc)}

    def _build_miot_followup_call(self, messages: list[ChatMessage]) -> ToolCall | None:
        request = _last_user_text(messages)
        correction_call = self._build_miot_correction_call(request)
        if correction_call is not None:
            return correction_call
        action = _infer_miot_action_from_text(request)
        ambiguity_call = self._build_miot_ambiguity_followup_call(request, action)
        if ambiguity_call is not None:
            return ambiguity_call
        if "xiaomi_miot_control_device" not in self.tools:
            return None
        if not action or not (
            _is_miot_followup_reference(request)
            or self._looks_like_last_miot_property_followup(request)
        ):
            return None
        if self._active_miot_ambiguity() is not None or not self._last_miot_control:
            return None
        device = self._last_miot_control.get("device")
        if not isinstance(device, dict) or not device.get("name"):
            return None
        arguments: dict[str, Any] = {
            "request": request,
            "device": str(device.get("name") or ""),
            "action": action,
        }
        if device.get("room_name"):
            arguments["area"] = str(device["room_name"])
        if device.get("device_class"):
            arguments["device_class"] = str(device["device_class"])
        log_event(
            "tools",
            "miot_followup",
            log_id="tools.miot_followup",
            action=action,
            device=arguments["device"],
        )
        return ToolCall(
            id="local_miot_followup",
            name="xiaomi_miot_control_device",
            arguments=arguments,
        )

    def _build_direct_miot_control_call(self, request: str) -> ToolCall | None:
        if "xiaomi_miot_control_device" not in self.tools:
            return None
        if not _looks_like_miot_control_text(request):
            return None
        normalized = request.lower().replace(" ", "")
        if not _has_explicit_miot_device_text(normalized):
            return None
        action = _infer_miot_action_from_text(request)
        if not action:
            return None
        log_event(
            "tools",
            "miot_followup",
            log_id="tools.miot_followup",
            action=action,
            device="request",
        )
        return ToolCall(
            id="local_miot_direct",
            name="xiaomi_miot_control_device",
            arguments={
                "request": request,
                "action": action,
            },
        )

    def _build_miot_ambiguity_followup_call(self, request: str, action: str) -> ToolCall | None:
        ambiguity = self._active_miot_ambiguity()
        if not ambiguity:
            return None

        remembered_tool = str(ambiguity.get("tool_name") or "xiaomi_miot_control_device")
        tool_name = (
            "xiaomi_miot_control_device"
            if action and "xiaomi_miot_control_device" in self.tools
            else remembered_tool
        )
        if tool_name not in self.tools:
            return None
        query = ambiguity.get("query") if isinstance(ambiguity.get("query"), dict) else {}
        candidates = (
            ambiguity.get("candidates")
            if isinstance(ambiguity.get("candidates"), list)
            else []
        )
        selected_candidate = _select_miot_ambiguity_candidate(request, candidates)
        effective_action = action or str(query.get("action") or "")
        if tool_name == "xiaomi_miot_control_device" and not effective_action:
            return None
        if not action and selected_candidate is None:
            return None
        if (
            action
            and selected_candidate is None
            and not _is_miot_followup_reference(request)
            and not _is_miot_group_followup_reference(request)
        ):
            return None

        arguments: dict[str, Any] = {
            "request": request,
        }
        if tool_name == "xiaomi_miot_control_device":
            arguments["action"] = effective_action
        if selected_candidate is not None:
            if selected_candidate.get("name"):
                arguments["device"] = str(selected_candidate["name"])
            if selected_candidate.get("room_name"):
                arguments["area"] = str(selected_candidate["room_name"])
            if selected_candidate.get("device_class"):
                arguments["device_class"] = str(selected_candidate["device_class"])
        else:
            for key in ("area", "device", "device_class"):
                value = str(query.get(key) or "").strip()
                if value:
                    arguments[key] = value

        if not arguments.get("device_class"):
            device_class = _common_candidate_value(candidates, "device_class")
            if device_class:
                arguments["device_class"] = device_class
        if not arguments.get("area"):
            area = _common_candidate_value(candidates, "room_name")
            if area:
                arguments["area"] = area
        if not arguments.get("device") and arguments.get("device_class"):
            arguments["device"] = str(arguments["device_class"])
        if tool_name == "xiaomi_miot_read_device_property":
            property_query = str(query.get("property") or query.get("property_query") or "").strip()
            if property_query:
                arguments["property_query"] = property_query

        target = str(arguments.get("device") or arguments.get("device_class") or "ambiguous")
        log_event(
            "tools",
            "miot_followup",
            log_id="tools.miot_followup",
            action=effective_action or "read",
            device=target,
        )
        return ToolCall(
            id="local_miot_ambiguity_followup",
            name=tool_name,
            arguments=arguments,
        )

    def _build_miot_correction_call(self, request: str) -> ToolCall | None:
        if not _looks_like_miot_correction(request):
            return None
        ambiguity = self._active_previous_miot_ambiguity()
        if not ambiguity or not self._last_miot_control:
            return None

        last_device = (
            self._last_miot_control.get("device")
            if isinstance(self._last_miot_control.get("device"), dict)
            else {}
        )
        candidates = (
            ambiguity.get("candidates")
            if isinstance(ambiguity.get("candidates"), list)
            else []
        )
        remaining: list[dict[str, Any]] = []
        last_name = str(last_device.get("name") or "")
        last_did = str(last_device.get("did") or "")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_name = str(candidate.get("name") or "")
            candidate_did = str(candidate.get("did") or "")
            if last_did and candidate_did and candidate_did == last_did:
                continue
            if last_name and candidate_name and candidate_name == last_name:
                continue
            remaining.append(candidate)
        if len(remaining) != 1:
            return None

        query = ambiguity.get("query") if isinstance(ambiguity.get("query"), dict) else {}
        action = str(query.get("action") or self._last_miot_control.get("action") or "")
        if not action:
            return None

        candidate = remaining[0]
        arguments: dict[str, Any] = {
            "request": request,
            "action": action,
        }
        if candidate.get("name"):
            arguments["device"] = str(candidate["name"])
        if candidate.get("room_name"):
            arguments["area"] = str(candidate["room_name"])
        if candidate.get("device_class"):
            arguments["device_class"] = str(candidate["device_class"])
        target = str(arguments.get("device") or arguments.get("device_class") or "correction")
        log_event(
            "tools",
            "miot_followup",
            log_id="tools.miot_followup",
            action=action,
            device=target,
        )
        return ToolCall(
            id="local_miot_correction_followup",
            name="xiaomi_miot_control_device",
            arguments=arguments,
        )

    def _with_tool_direct_response(
        self,
        tool_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name not in {
            "xiaomi_miot_control_device",
            "xiaomi_miot_read_device_property",
            "xiaomi_miot_control",
        }:
            return payload
        if str(payload.get("direct_response") or "").strip():
            return payload
        response = _format_tool_payload_response(payload)
        if response:
            return {**payload, "direct_response": response}
        return payload

    def _remember_miot_control(self, tool_name: str, payload: dict[str, Any]) -> None:
        if tool_name not in {"xiaomi_miot_control_device", "xiaomi_miot_read_device_property"}:
            return
        status = str(payload.get("status") or "")
        if status == "ambiguous":
            if tool_name == "xiaomi_miot_control_device":
                self._last_miot_control = None
            self._last_miot_ambiguity = None
            self._previous_miot_ambiguity = None
            candidates = (
                payload.get("candidates")
                if isinstance(payload.get("candidates"), list)
                else []
            )
            if candidates:
                remembered_candidates = [
                    dict(candidate) for candidate in candidates if isinstance(candidate, dict)
                ]
                query = (
                    dict(payload.get("query") or {})
                    if isinstance(payload.get("query"), dict)
                    else {}
                )
                self._last_miot_ambiguity = {
                    "candidates": remembered_candidates,
                    "query": query,
                    "tool_name": tool_name,
                    "original_request": str(query.get("request") or ""),
                    "created_at": time.monotonic(),
                    "ttl_seconds": _MIOT_PENDING_CONTEXT_TTL_SECONDS,
                }
            return
        if tool_name == "xiaomi_miot_read_device_property":
            self._last_miot_ambiguity = None
            return
        if status not in {"verified", "ok"}:
            self._last_miot_control = None
            self._last_miot_ambiguity = None
            self._previous_miot_ambiguity = None
            return
        if self._last_miot_ambiguity is not None:
            self._previous_miot_ambiguity = dict(self._last_miot_ambiguity)
        self._last_miot_ambiguity = None
        device = payload.get("device") if isinstance(payload.get("device"), dict) else None
        if not device or not device.get("name"):
            return
        self._last_miot_control = {
            "device": dict(device),
            "action": str(payload.get("action") or ""),
        }


def create_tool_runner(
    config: AssistantConfig,
    chat: ChatClient,
) -> VoiceToolRunner | None:
    if not config.tools.enabled:
        return None

    definitions: list[ToolDefinition] = []
    music_controller: MusicPlaybackController | None = None
    if config.tools.allow_time:
        definitions.append(create_current_time_tool())
    if config.tools.allow_weather:
        definitions.append(create_current_weather_tool())
    if config.tools.allow_search:
        definitions.append(create_web_search_tool(config.search))
    if config.tools.allow_volume:
        definitions.append(create_get_system_volume_tool(config.tts.playback_device))
        definitions.append(create_set_system_volume_tool(config.tts.playback_device))
    if config.tools.allow_music and config.music.provider != "disabled":
        music_controller = MusicPlaybackController(
            config.music,
            fallback_device=config.tts.playback_device,
            fallback_sample_rate=config.tts.playback_sample_rate,
            fallback_channels=config.tts.playback_channels,
        )
        definitions.append(create_search_music_tool(music_controller))
        definitions.append(create_play_music_tool(music_controller))
        definitions.append(create_stop_music_tool(music_controller))
    if config.tools.allow_miot and config.xiaomi_miot.enabled:
        miot = XiaomiMiotController(config.xiaomi_miot)
        definitions.append(create_xiaomi_miot_auth_url_tool(miot))
        definitions.append(create_xiaomi_miot_exchange_auth_code_tool(miot))
        definitions.append(create_xiaomi_miot_control_device_tool(miot))
        definitions.append(create_xiaomi_miot_area_info_tool(miot))
        definitions.append(create_xiaomi_miot_device_classes_tool(miot))
        definitions.append(create_xiaomi_miot_devices_tool(miot))
        definitions.append(create_xiaomi_miot_device_spec_tool(miot))
        definitions.append(create_xiaomi_miot_read_device_property_tool(miot))
        definitions.append(create_xiaomi_miot_get_property_tool(miot))
        definitions.append(create_xiaomi_miot_control_tool(miot))

    if not definitions:
        return None
    return VoiceToolRunner(
        chat=chat,
        tools=definitions,
        max_iterations=config.tools.max_iterations,
        music_controller=music_controller,
    )


def create_current_time_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_current_time",
        description="Get the current local time for an IANA timezone.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone name. Use Asia/Shanghai if the user asks for China time."
                    ),
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=get_current_time,
    )


def get_current_time(arguments: dict[str, Any]) -> dict[str, Any]:
    timezone_name = str(arguments.get("timezone") or "").strip()
    if timezone_name:
        try:
            tz = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise RuntimeError(f"Unknown timezone: {timezone_name}") from exc
        now = datetime.now(tz)
    else:
        now = datetime.now().astimezone()
        timezone_name = now.tzname() or "local"

    result = {
        "timezone": timezone_name,
        "datetime": now.isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
    }
    result["direct_response"] = format_current_time_response(result)
    return result


def create_current_weather_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_current_weather",
        description="Get current or tomorrow weather for a city or place using Open-Meteo.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, such as Shanghai or New York.",
                },
                "target_day": {
                    "type": "string",
                    "enum": ["today", "tomorrow"],
                    "description": (
                        "Use today for current weather, or tomorrow for tomorrow forecast."
                    ),
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        handler=get_current_weather,
    )


def get_current_weather(arguments: dict[str, Any]) -> dict[str, Any]:
    location = str(arguments.get("location") or "").strip()
    if not location:
        raise RuntimeError("location is required")

    target_day = _normalize_weather_target_day(arguments.get("target_day"))
    place = _resolve_weather_place(location)
    latitude = place["latitude"]
    longitude = place["longitude"]
    forecast = _get_cached_weather_forecast(
        float(latitude),
        float(longitude),
        require_daily=target_day != "today",
    )
    current = forecast.get("current") if isinstance(forecast, dict) else {}
    units = forecast.get("current_units") if isinstance(forecast, dict) else {}
    daily = _weather_daily_entry(forecast, target_day)
    code_source = daily if daily else current
    code = code_source.get("weather_code") if isinstance(code_source, dict) else None
    result = {
        "location": {
            "name": place.get("name"),
            "country": place.get("country"),
            "admin1": place.get("admin1"),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": forecast.get("timezone"),
        },
        "target_day": target_day,
        "current": current,
        "units": units,
        "daily": daily,
        "daily_units": forecast.get("daily_units") if isinstance(forecast, dict) else {},
        "summary": _weather_code_text(code),
    }
    result["direct_response"] = format_weather_response(result)
    return result


def format_current_time_response(result: dict[str, Any]) -> str:
    timestamp = str(result.get("datetime") or "")
    if "T" in timestamp:
        date_part, time_part = timestamp.split("T", 1)
        time_part = time_part.split("+", 1)[0].split("-", 1)[0]
        return f"现在是{date_part} {time_part}。"
    return f"现在时间是{timestamp}。"


def format_weather_response(result: dict[str, Any]) -> str:
    location = result.get("location") if isinstance(result.get("location"), dict) else {}
    current = result.get("current") if isinstance(result.get("current"), dict) else {}
    units = result.get("units") if isinstance(result.get("units"), dict) else {}
    daily = result.get("daily") if isinstance(result.get("daily"), dict) else {}
    daily_units = result.get("daily_units") if isinstance(result.get("daily_units"), dict) else {}
    name = str(location.get("name") or "当地")
    target_day = str(result.get("target_day") or "today")
    day_label = "明天" if target_day == "tomorrow" else "今天"

    if daily:
        weather_code = daily.get("weather_code")
        summary = _weather_code_zh(weather_code)
        temp_max = daily.get("temperature_2m_max")
        temp_min = daily.get("temperature_2m_min")
        precipitation = daily.get("precipitation_sum")
        precipitation_probability = daily.get("precipitation_probability_max")
        temp_unit = str(
            daily_units.get("temperature_2m_max")
            or daily_units.get("temperature_2m_min")
            or "度"
        )
        parts = [f"{name}{day_label}{summary}"]
        if temp_max is not None:
            parts.append(f"最高{_format_number(temp_max)}{temp_unit}")
        if temp_min is not None:
            parts.append(f"最低{_format_number(temp_min)}{temp_unit}")
        if precipitation_probability is not None:
            parts.append(f"降水概率{_format_number(precipitation_probability)}%")
        response = "，".join(parts) + "。"
        if _is_wet_weather(weather_code, precipitation) or _is_high_precipitation_probability(
            precipitation_probability
        ):
            response += "出门记得带伞。"
        return response

    weather_code = current.get("weather_code")
    summary = _weather_code_zh(weather_code)
    temperature = current.get("temperature_2m")
    apparent = current.get("apparent_temperature")
    precipitation = current.get("precipitation")
    temp_unit = str(units.get("temperature_2m") or "度")

    parts = [f"{name}{day_label}{summary}"]
    if temperature is not None:
        parts.append(f"气温{_format_number(temperature)}{temp_unit}")
    if apparent is not None:
        parts.append(f"体感{_format_number(apparent)}{temp_unit}")
    response = "，".join(parts) + "。"
    if _is_wet_weather(weather_code, precipitation):
        response += "出门记得带伞。"
    return response


def warm_weather_cache(location: str) -> None:
    if not location.strip():
        return
    get_current_weather({"location": location, "target_day": "tomorrow"})


def create_web_search_tool(config: SearchConfig) -> ToolDefinition:
    return ToolDefinition(
        name="web_search",
        description=(
            "Search the web for current or external information using Tavily or Baidu. "
            "Use this when the user asks to search online, find news, look up current "
            "facts, or asks about information that may have changed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in the user's language.",
                },
                "provider": {
                    "type": "string",
                    "enum": ["auto", "tavily", "baidu"],
                    "description": "Optional search provider override.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum number of results to return.",
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year", "d", "w", "m", "y"],
                    "description": "Optional recency filter for Tavily.",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "Optional Tavily topic.",
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional domains to include for Tavily.",
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional domains to exclude for Tavily.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=lambda arguments: search_web(config, arguments),
    )


def search_web(config: SearchConfig, arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise RuntimeError("query is required")
    max_results = _optional_int(arguments, "max_results", default=config.max_results)
    max_results = max(1, min(10, max_results))
    provider = _search_provider(arguments, config)

    if provider == "auto":
        provider = "baidu" if _contains_cjk(query) else "tavily"
    if provider == "tavily":
        result = _search_tavily(config, arguments, query, max_results)
    elif provider == "baidu":
        result = _search_baidu(config, arguments, query, max_results)
    else:
        raise RuntimeError(f"Unsupported search provider: {provider}")
    _log_search_result(result)
    return result


def _resolve_weather_place(location: str) -> dict[str, Any]:
    _load_weather_disk_cache()
    key = _weather_location_key(location)
    common = _COMMON_WEATHER_LOCATIONS.get(key)
    if common is not None:
        return dict(common)
    cached = _WEATHER_GEOCODE_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    search_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {
            "name": location,
            "count": 1,
            "language": "zh",
            "format": "json",
        }
    )
    search = _get_json(search_url, timeout=15)
    results = search.get("results") if isinstance(search, dict) else None
    if not results:
        raise RuntimeError(f"Could not find weather location: {location}")

    place = {
        "name": results[0].get("name"),
        "country": results[0].get("country"),
        "admin1": results[0].get("admin1"),
        "latitude": results[0]["latitude"],
        "longitude": results[0]["longitude"],
    }
    _WEATHER_GEOCODE_CACHE[key] = place
    _save_weather_disk_cache()
    return dict(place)


def _get_cached_weather_forecast(
    latitude: float,
    longitude: float,
    *,
    require_daily: bool = False,
) -> dict[str, Any]:
    _load_weather_disk_cache()
    key = f"{latitude:.4f},{longitude:.4f}"
    cached = _WEATHER_FORECAST_CACHE.get(key)
    now = time.time()
    if (
        cached is not None
        and now - float(cached.get("cached_at", 0)) <= _WEATHER_CACHE_TTL_SECONDS
        and _is_valid_weather_forecast(cached.get("forecast"))
        and (not require_daily or _has_daily_weather(cached.get("forecast")))
    ):
        return dict(cached["forecast"])

    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                ]
            ),
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                ]
            ),
            "forecast_days": 2,
            "timezone": "auto",
        }
    )
    forecast = _get_json(forecast_url, timeout=15)
    if not _is_valid_weather_forecast(forecast):
        raise RuntimeError("Weather provider returned an invalid forecast")
    _WEATHER_FORECAST_CACHE[key] = {
        "cached_at": now,
        "forecast": forecast,
    }
    _save_weather_disk_cache()
    return forecast


def _load_weather_disk_cache() -> None:
    global _WEATHER_DISK_CACHE_LOADED  # noqa: PLW0603
    if _WEATHER_DISK_CACHE_LOADED:
        return
    _WEATHER_DISK_CACHE_LOADED = True
    if not _WEATHER_CACHE_PATH.exists():
        return
    try:
        data = json.loads(_WEATHER_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    geocode = data.get("geocode") if isinstance(data, dict) else None
    forecast = data.get("forecast") if isinstance(data, dict) else None
    if isinstance(geocode, dict):
        _WEATHER_GEOCODE_CACHE.update(
            {str(key): value for key, value in geocode.items() if isinstance(value, dict)}
        )
    if isinstance(forecast, dict):
        now = time.time()
        _WEATHER_FORECAST_CACHE.update(
            {
                str(key): value
                for key, value in forecast.items()
                if isinstance(value, dict)
                and now - float(value.get("cached_at", 0)) <= _WEATHER_CACHE_TTL_SECONDS
                and _is_valid_weather_forecast(value.get("forecast"))
            }
        )


def _save_weather_disk_cache() -> None:
    try:
        _WEATHER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _WEATHER_CACHE_PATH.write_text(
            json.dumps(
                {
                    "geocode": _WEATHER_GEOCODE_CACHE,
                    "forecast": _WEATHER_FORECAST_CACHE,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        return


def _weather_location_key(location: str) -> str:
    return str(location or "").strip().replace(" ", "")


def _is_valid_weather_forecast(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    current = value.get("current")
    return isinstance(current, dict) and "weather_code" in current


def _has_daily_weather(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    daily = value.get("daily")
    if not isinstance(daily, dict):
        return False
    weather_codes = daily.get("weather_code")
    return isinstance(weather_codes, list) and len(weather_codes) >= 2


def _weather_daily_entry(forecast: dict[str, Any], target_day: str) -> dict[str, Any]:
    if target_day == "today":
        return {}
    daily = forecast.get("daily") if isinstance(forecast, dict) else None
    if not isinstance(daily, dict):
        return {}
    index = 1 if target_day == "tomorrow" else 0
    entry: dict[str, Any] = {}
    for key, value in daily.items():
        if isinstance(value, list) and len(value) > index:
            entry[key] = value[index]
    return entry if "weather_code" in entry else {}


def _normalize_weather_target_day(value: object) -> str:
    target_day = str(value or "today").strip().lower()
    if target_day in {"tomorrow", "明天", "明日"}:
        return "tomorrow"
    return "today"


def create_get_system_volume_tool(device: str | int | None = None) -> ToolDefinition:
    return ToolDefinition(
        name="get_system_volume",
        description="Get the current Windows playback volume for the configured speaker.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda _arguments: get_system_output_volume(device=device),
    )


def create_set_system_volume_tool(device: str | int | None = None) -> ToolDefinition:
    return ToolDefinition(
        name="set_system_volume",
        description=(
            "Set, raise, lower, mute, or unmute the Windows playback volume for the "
            "configured speaker. Use volume_percent for absolute requests like 30%, "
            "relative_percent for louder/quieter requests, and muted for mute state."
        ),
        parameters={
            "type": "object",
            "properties": {
                "volume_percent": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Absolute target output volume from 0 to 100.",
                },
                "relative_percent": {
                    "type": "number",
                    "minimum": -100,
                    "maximum": 100,
                    "description": "Relative volume change, such as 10 or -10.",
                },
                "muted": {
                    "type": "boolean",
                    "description": "True to mute playback, false to unmute playback.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda arguments: set_system_output_volume(
            device=device,
            volume_percent=_optional_percent(
                arguments,
                "volume_percent",
                aliases=("volume", "percent", "level"),
            ),
            relative_percent=_optional_percent(arguments, "relative_percent"),
            muted=_optional_bool(arguments, "muted"),
        ),
    )


def create_search_music_tool(music: MusicPlaybackController) -> ToolDefinition:
    return ToolDefinition(
        name="search_music",
        description=(
            "Search the configured music provider for songs. Use before playing if the user "
            "asks to choose among results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Song, artist, album, or playlist search query.",
                },
                "server": {
                    "type": "string",
                    "description": (
                        "Optional Meting source, such as netease, tencent, kugou, or kuwo."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=music.search,
    )


def create_play_music_tool(music: MusicPlaybackController) -> ToolDefinition:
    return ToolDefinition(
        name="play_music",
        description=(
            "Search for a song and start playback through the local speaker. Use this when the "
            "user asks to play music. Do not tell the user music is playing unless this tool "
            "has been called."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Song, artist, album, or playlist the user wants to hear.",
                },
                "server": {
                    "type": "string",
                    "description": (
                        "Optional Meting source, such as netease, tencent, kugou, or kuwo."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=music.play,
    )


def create_stop_music_tool(music: MusicPlaybackController) -> ToolDefinition:
    return ToolDefinition(
        name="stop_music",
        description="Stop the current local music playback.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=lambda arguments: music.stop(arguments, wait=True),
    )


def create_xiaomi_miot_auth_url_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_auth_url",
        description=(
            "Get a Xiaomi Home OAuth login URL when MIoT is not logged in. This only "
            "creates the login URL and does not control devices."
        ),
        parameters={
            "type": "object",
            "properties": {
                "skip_confirm": {
                    "type": "boolean",
                    "description": "Whether Xiaomi should skip the confirmation page if possible.",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=miot.auth_url,
    )


def create_xiaomi_miot_exchange_auth_code_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_exchange_auth_code",
        description=(
            "Exchange a Xiaomi OAuth redirect code for a local MIoT token file. Use only "
            "when the user explicitly provides the redirect code after opening the auth URL."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code parameter from the Xiaomi OAuth redirect URL.",
                },
                "state": {
                    "type": "string",
                    "description": "Optional state parameter from the Xiaomi OAuth redirect URL.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=miot.exchange_auth_code,
    )


def create_xiaomi_miot_control_device_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_control_device",
        description=(
            "Resolve and control a Xiaomi Home device from natural spoken commands. "
            "Use this for requests like turning the study light on/off. It fuzzy matches "
            "area, device name, and device class, selects the MIoT property/action, and "
            "verifies readable property writes. When multiple matched devices remain, "
            "the tool can read current power states to close the only device that is on "
            "or open the only device that is off. If the result status is ambiguous, ask "
            "the user to choose; if one low-risk device matches, do not ask again."
        ),
        parameters={
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "Original user request, e.g. 帮我把书房的灯灯打开.",
                },
                "area": {
                    "type": "string",
                    "description": "Optional room or home name, e.g. 书房.",
                },
                "device": {
                    "type": "string",
                    "description": "Optional device name or kind, e.g. 灯, 吸顶灯, 窗帘.",
                },
                "device_class": {
                    "type": "string",
                    "description": "Optional Xiaomi device class, e.g. light, switch, curtain.",
                },
                "action": {
                    "type": "string",
                    "description": "Desired action, e.g. 打开, 关闭, turn_on, turn_off.",
                },
                "value": {
                    "type": ["string", "number", "boolean", "array", "object"],
                    "description": "Optional explicit target value. Omit for ordinary on/off.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "True to resolve the target without controlling it.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=miot.control_device,
    )


def create_xiaomi_miot_area_info_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_get_area_info",
        description="Get Xiaomi Home homes and rooms that contain devices.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=miot.get_area_info,
    )


def create_xiaomi_miot_device_classes_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_get_device_classes",
        description="Get Xiaomi Home device classes, such as light, curtain, or air-conditioner.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=miot.get_device_classes,
    )


def create_xiaomi_miot_devices_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_get_devices",
        description=(
            "Get Xiaomi Home devices. Use this before reading or controlling a Xiaomi "
            "Home device so the device id is known."
        ),
        parameters={
            "type": "object",
            "properties": {
                "area_id": {
                    "type": "string",
                    "description": "Optional home or room id from xiaomi_miot_get_area_info.",
                },
                "device_class": {
                    "type": "string",
                    "description": "Optional device class from xiaomi_miot_get_device_classes.",
                },
                "refresh": {
                    "type": "boolean",
                    "description": "True to reload the list from Xiaomi Home instead of cache.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=miot.get_devices,
    )


def create_xiaomi_miot_device_spec_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_get_device_spec",
        description=(
            "Get readable/writeable MIoT properties and actions for a Xiaomi Home device. "
            "Use this before xiaomi_miot_get_property or xiaomi_miot_control; do not guess iids."
        ),
        parameters={
            "type": "object",
            "properties": {
                "did": {
                    "type": "string",
                    "description": "Device id from xiaomi_miot_get_devices.",
                }
            },
            "required": ["did"],
            "additionalProperties": False,
        },
        handler=miot.get_device_spec,
    )


def create_xiaomi_miot_get_property_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_get_property",
        description=(
            "Read one Xiaomi Home MIoT property. The iid must come from "
            "xiaomi_miot_get_device_spec and must start with prop.0."
        ),
        parameters={
            "type": "object",
            "properties": {
                "did": {
                    "type": "string",
                    "description": "Device id from xiaomi_miot_get_devices.",
                },
                "iid": {
                    "type": "string",
                    "description": (
                        "Property iid from xiaomi_miot_get_device_spec, e.g. prop.0.2.1."
                    ),
                },
            },
            "required": ["did", "iid"],
            "additionalProperties": False,
        },
        handler=miot.get_property,
    )


def create_xiaomi_miot_read_device_property_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_read_device_property",
        description=(
            "Read a Xiaomi Home device property from a natural spoken request. "
            "Use this for state questions such as checking a room device status, "
            "an air purifier's displayed air quality, PM2.5, temperature, humidity, "
            "or power state. It fuzzy matches the device and readable MIoT property."
        ),
        parameters={
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "Original user request, e.g. 查一下家里净化器的空气质量.",
                },
                "area": {
                    "type": "string",
                    "description": "Optional room or home name.",
                },
                "device": {
                    "type": "string",
                    "description": "Optional device name or type.",
                },
                "device_class": {
                    "type": "string",
                    "description": "Optional device class, e.g. airpurifier, light, sensor.",
                },
                "property": {
                    "type": "string",
                    "description": "Optional property query, e.g. air quality, pm2.5, power.",
                },
                "property_query": {
                    "type": "string",
                    "description": "Alias of property.",
                },
            },
            "required": ["request"],
            "additionalProperties": False,
        },
        handler=miot.read_device_property,
    )


def create_xiaomi_miot_control_tool(miot: XiaomiMiotController) -> ToolDefinition:
    return ToolDefinition(
        name="xiaomi_miot_control",
        description=(
            "Low-level Xiaomi Home MIoT property/action control when the exact did and iid "
            "are already known. For natural spoken room/device commands, prefer "
            "xiaomi_miot_control_device. Otherwise first call "
            "xiaomi_miot_get_devices and xiaomi_miot_get_device_spec, then pass an iid "
            "from that spec."
        ),
        parameters={
            "type": "object",
            "properties": {
                "did": {
                    "type": "string",
                    "description": "Device id from xiaomi_miot_get_devices.",
                },
                "iid": {
                    "type": "string",
                    "description": (
                        "Property or action iid from xiaomi_miot_get_device_spec, such as "
                        "prop.0.2.1 or action.0.2.1."
                    ),
                },
                "value": {
                    "type": ["string", "number", "boolean", "array", "object"],
                    "description": (
                        "Property value, or a list/JSON array string for action input values."
                    )
                },
            },
            "required": ["did", "iid", "value"],
            "additionalProperties": False,
        },
        handler=miot.control,
    )


def search_music_tracks(
    config: MusicConfig,
    query: str,
    server: str | None = None,
) -> list[MusicTrack]:
    if config.provider == "meting":
        return _search_meting_music(config, query, server or config.server)
    raise RuntimeError(f"Unsupported music provider: {config.provider}")


def _assistant_tool_call_payload(call: ToolCall) -> dict[str, Any]:
    if call.raw:
        return call.raw
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": _to_json_text(call.arguments),
        },
    }


def _with_tool_use_instructions(messages: list[ChatMessage]) -> list[ChatMessage]:
    if not messages or messages[0].role != "system":
        return list(messages)
    system = messages[0]
    return [
        ChatMessage(
            role="system",
            content=f"{system.content}\n\n{_TOOL_USE_INSTRUCTIONS}",
            tool_calls=system.tool_calls,
            tool_call_id=system.tool_call_id,
        ),
        *messages[1:],
    ]


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _select_tool_names_for_text(text: str, available_names: set[str]) -> set[str]:
    normalized = text.lower().replace(" ", "")
    selected: set[str] = set()
    miot_read_text = _looks_like_miot_read_text(text)
    miot_control_text = _looks_like_miot_control_text(text)
    if _looks_like_weather_tool_text(normalized):
        selected.add("get_current_weather")
    if any(
        term in normalized
        for term in (
            "搜索",
            "搜一下",
            "查一下",
            "查找",
            "查询",
            "网上",
            "网页",
            "新闻",
            "最新",
            "资料",
            "消息",
            "百度",
            "tavily",
            "websearch",
        )
    ):
        selected.add("web_search")
    if miot_read_text or miot_control_text:
        selected.discard("web_search")
    if any(
        term in normalized
        for term in (
            "几点",
            "几点了",
            "现在时间",
            "当前时间",
            "报一下时间",
            "告诉我时间",
            "time",
        )
    ):
        selected.add("get_current_time")
    if any(term in normalized for term in ("音乐", "歌曲", "播放", "暂停", "歌", "music")):
        selected.update({"search_music", "play_music", "stop_music"})
    if any(term in normalized for term in ("音量", "声音", "静音", "volume", "mute")):
        selected.update({"get_system_volume", "set_system_volume"})
    if any(
        term in normalized
        for term in (
            "米家",
            "灯",
            "开关",
            "窗帘",
            "空调",
            "冷气",
            "空调机",
            "传感器",
            "摄像",
            "门锁",
            "插座",
            "插排",
            "排插",
            "风扇",
            "电扇",
            "吊扇",
            "家里",
            "设备",
            "状态",
            "显示",
            "空气质量",
            "pm2.5",
            "pm25",
            "净化器",
            "加湿器",
            "亮度",
            "模式",
            "制冷",
            "制热",
            "除湿",
            "睡眠",
            "调到",
            "设置",
            "设为",
            "打开",
            "关闭",
            "开了",
            "关了",
            "关掉",
            "关上",
        )
    ) or miot_control_text:
        selected.update(
            {
                "xiaomi_miot_control_device",
                "xiaomi_miot_get_devices",
                "xiaomi_miot_get_device_spec",
                "xiaomi_miot_read_device_property",
                "xiaomi_miot_get_property",
                "xiaomi_miot_control",
            }
        )
    if miot_read_text:
        selected.update(
            {
                "xiaomi_miot_read_device_property",
                "xiaomi_miot_get_devices",
                "xiaomi_miot_get_device_spec",
                "xiaomi_miot_get_property",
                "xiaomi_miot_get_area_info",
            }
        )
    if miot_read_text or miot_control_text:
        selected.discard("get_current_weather")
    return selected & available_names


def _looks_like_weather_tool_text(normalized: str) -> bool:
    if _looks_like_iot_temperature_text(normalized):
        return False
    return any(term in normalized for term in ("天气", "气温", "温度", "下雨", "雨", "weather"))


def _looks_like_iot_temperature_text(normalized: str) -> bool:
    if "温度" not in normalized and "度" not in normalized:
        return False
    device_terms = (
        "空调",
        "冷气",
        "空调机",
        "加湿器",
        "净化器",
        "空气净化器",
        "传感器",
        "设备",
    )
    if any(term in normalized for term in device_terms):
        return True
    return any(
        term in normalized
        for term in ("调成", "调到", "调高", "调低", "设置", "设为")
    )


def _looks_like_miot_read_text(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    if any(term in normalized for term in ("音乐", "歌曲", "播放", "music")):
        return False
    if _looks_like_miot_control_text(text):
        return False
    if not _has_explicit_miot_device_text(normalized) and not any(
        term in normalized for term in ("家里", "米家", "设备")
    ):
        return False
    return any(
        term in normalized
        for term in (
            "查一下",
            "查询",
            "看一下",
            "看看",
            "显示",
            "状态",
            "多少",
            "空气质量",
            "pm2.5",
            "pm25",
            "温度",
            "湿度",
            "开着",
            "关着",
            "开没开",
            "关没关",
            "亮度",
            "模式",
        )
    )


def _looks_like_miot_control_text(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    if not _infer_miot_action_from_text(text):
        return False
    if any(term in normalized for term in ("音乐", "歌曲", "播放", "music", "volume", "音量")):
        return False
    return _has_explicit_miot_device_text(normalized) or _is_miot_followup_reference(text)


def _requires_tool_call_for_text(text: str, available_names: set[str]) -> bool:
    return "xiaomi_miot_control_device" in available_names and _looks_like_miot_control_text(text)


def _infer_miot_action_from_text(text: str) -> str:
    normalized = text.lower().replace(" ", "")
    if any(
        term in normalized
        for term in (
            "turnoff",
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
    ):
        return "turn_off"
    if any(
        term in normalized
        for term in (
            "turnon",
            "on",
            "open",
            "打开",
            "开启",
            "开灯",
            "开一下",
            "拉开",
            "拉起来",
            "开了",
            "亮",
            "没有开",
            "没开",
            "还没开",
        )
    ):
        return "turn_on"
    if any(
        term in normalized
        for term in (
            "调到",
            "调成",
            "设置",
            "设为",
            "调亮",
            "调暗",
            "亮度",
            "温度",
            "模式",
            "制冷",
            "制热",
            "除湿",
            "睡眠",
            "一半",
            "百分",
        )
    ):
        return "set_value"
    return ""


def _is_miot_followup_reference(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    if not _infer_miot_action_from_text(text):
        return False
    if any(term in normalized for term in ("它", "他", "这个", "那个", "刚才", "上一个", "上次")):
        return True
    if any(term in normalized for term in ("没有关", "没关", "还没关", "没有开", "没开", "还没开")):
        return True
    if "再" in normalized and not _has_explicit_miot_device_text(normalized):
        return True
    return False


def _is_miot_group_followup_reference(text: str) -> bool:
    normalized = re.sub(r"[\W_]+", "", str(text or "").lower())
    if not _infer_miot_action_from_text(text):
        return False
    if any(
        term in normalized
        for term in (
            "\u5168\u90e8",
            "\u6240\u6709",
            "\u5168\u90fd",
            "\u6bcf\u4e2a",
        )
    ):
        return True
    if "\u90fd" not in normalized:
        return False
    return any(
        term in normalized
        for term in (
            "\u90fd\u5173",
            "\u90fd\u5173\u95ed",
            "\u90fd\u6253\u5f00",
            "\u90fd\u5f00",
            "\u706f\u90fd",
            "\u7a7a\u8c03\u90fd",
            "\u51b7\u6c14\u90fd",
            "\u7a97\u5e18\u90fd",
            "\u8bbe\u5907\u90fd",
        )
    )


def _looks_like_miot_correction(text: str) -> bool:
    normalized = re.sub(r"[\W_]+", "", str(text or "").lower())
    if any(
        term in normalized
        for term in (
            "\u4e0d\u662f\u8fd9\u4e2a",
            "\u4e0d\u662f\u90a3\u4e2a",
            "\u4e0d\u662f\u5b83",
            "\u4e0d\u662f\u4ed6",
            "\u4e0d\u5bf9",
            "\u9519\u4e86",
        )
    ):
        return True
    if "\u9519" in normalized:
        return True
    return "\u4e0d" in normalized and any(
        term in normalized
        for term in (
            "\u8fd9\u4e2a",
            "\u90a3\u4e2a",
            "\u5b83",
            "\u4ed6",
            "\u5bf9",
        )
    )


def _miot_context_is_fresh(context: dict[str, Any]) -> bool:
    created_at = context.get("created_at")
    if not isinstance(created_at, (int, float)):
        return True
    ttl_seconds = context.get("ttl_seconds")
    ttl = (
        float(ttl_seconds)
        if isinstance(ttl_seconds, (int, float))
        else _MIOT_PENDING_CONTEXT_TTL_SECONDS
    )
    return time.monotonic() - float(created_at) <= max(0.0, ttl)


def _common_candidate_value(candidates: list[Any], key: str) -> str:
    values = {
        str(candidate.get(key) or "").strip()
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get(key) or "").strip()
    }
    return next(iter(values)) if len(values) == 1 else ""


def _select_miot_ambiguity_candidate(text: str, candidates: list[Any]) -> dict[str, Any] | None:
    candidate_dicts = [candidate for candidate in candidates if isinstance(candidate, dict)]
    if not candidate_dicts:
        return None

    ordinal = _miot_ordinal_index(text)
    if ordinal is not None:
        if 0 <= ordinal < len(candidate_dicts):
            return dict(candidate_dicts[ordinal])
        return None

    reference = _normalize_miot_reference(text)
    if not reference:
        return None

    scored: list[tuple[int, dict[str, Any]]] = []
    for candidate in candidate_dicts:
        score = _score_miot_candidate_reference(reference, candidate)
        if score > 0:
            scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return dict(scored[0][1])


def _score_miot_candidate_reference(reference: str, candidate: dict[str, Any]) -> int:
    score = 0
    for key, weight in (("name", 160), ("room_name", 120), ("home_name", 80)):
        value = _normalize_miot_reference(str(candidate.get(key) or ""))
        if value and value in reference:
            score = max(score, weight)
        elif value and reference in value and len(reference) >= 2:
            score = max(score, weight - 20)
    device_class = _normalize_miot_reference(str(candidate.get("device_class") or ""))
    if device_class and device_class in reference:
        score = max(score, 40)
    return score


def _miot_ordinal_index(text: str) -> int | None:
    normalized = _normalize_miot_reference(text)
    ordinal_terms = (
        (0, ("第一个", "第1个", "第一", "第1", "一号", "1号", "选一", "选1")),
        (1, ("第二个", "第2个", "第二", "第2", "二号", "2号", "选二", "选2")),
        (2, ("第三个", "第3个", "第三", "第3", "三号", "3号", "选三", "选3")),
        (3, ("第四个", "第4个", "第四", "第4", "四号", "4号", "选四", "选4")),
        (4, ("第五个", "第5个", "第五", "第5", "五号", "5号", "选五", "选5")),
    )
    for index, terms in ordinal_terms:
        if any(term in normalized for term in terms):
            return index
    return None


def _normalize_miot_reference(text: str) -> str:
    normalized = re.sub(
        r"[\s,，。.!！?？:：;；'\"“”‘’（）()\\[\\]{}<>《》、_-]+",
        "",
        str(text or "").lower(),
    )
    for filler in (
        "就",
        "把",
        "给我",
        "帮我",
        "请",
        "一下",
        "那个",
        "这个",
        "它",
        "他",
        "的",
        "设备",
    ):
        normalized = normalized.replace(filler, "")
    return normalized


def _has_explicit_miot_device_text(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "米家",
            "灯",
            "开关",
            "窗帘",
            "帘",
            "空调",
            "冷气",
            "空调机",
            "传感器",
            "摄像",
            "门锁",
            "插座",
            "插排",
            "排插",
            "风扇",
            "电扇",
            "吊扇",
            "净化器",
            "加湿器",
            "空气净化器",
            "空气质量",
            "pm2.5",
            "pm25",
        )
    )


def _direct_tool_response(results: list[dict[str, Any]]) -> str:
    if not results or any(result.get("ok") is False for result in results):
        return ""
    responses = [
        str(result.get("direct_response") or "").strip()
        for result in results
        if str(result.get("direct_response") or "").strip()
    ]
    if len(responses) != len(results):
        return ""
    return "\n".join(responses)


def _fallback_tool_response(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role != "tool":
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        response = _format_tool_payload_response(payload)
        if response:
            return response
    return "工具调用已经完成，但没有生成最终回复。"


def _format_tool_payload_response(payload: dict[str, Any]) -> str:
    if payload.get("ok") is False:
        return f"操作失败：{payload.get('error') or '工具调用失败'}"
    status = str(payload.get("status") or "")
    if status == "ambiguous":
        candidates = (
            payload.get("candidates")
            if isinstance(payload.get("candidates"), list)
            else []
        )
        names = [
            str(candidate.get("name") or "")
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("name")
        ]
        if names:
            return "找到多个匹配设备：" + "、".join(names[:5]) + "。请指定要控制哪一个。"
        return str(payload.get("message") or "找到多个匹配设备，请再说具体一点。")
    if status in ("not_found", "offline", "unsupported"):
        return str(payload.get("message") or "没有完成设备控制。")
    if status == "property_read":
        direct_response = str(payload.get("direct_response") or "").strip()
        if direct_response:
            return direct_response
        device = payload.get("device") if isinstance(payload.get("device"), dict) else {}
        item = payload.get("property") if isinstance(payload.get("property"), dict) else {}
        name = str(device.get("name") or "设备")
        prop = str(item.get("description") or item.get("name") or "状态")
        value = payload.get("value")
        unit = str(payload.get("unit") or "")
        return f"{name}的{prop}是{value}{unit}。"
    if status == "property_read_group":
        direct_response = str(payload.get("direct_response") or "").strip()
        if direct_response:
            return direct_response
        readings = payload.get("readings") if isinstance(payload.get("readings"), list) else []
        names = [
            str((reading.get("device") or {}).get("name") or "")
            for reading in readings
            if isinstance(reading, dict)
        ]
        if names:
            return "已读取：" + "、".join(name for name in names if name) + "。"
        return "没有读到设备状态。"
    if status in ("group_executed", "group_resolved"):
        direct_response = str(payload.get("direct_response") or "").strip()
        if direct_response:
            return direct_response
        success_count = int(payload.get("success_count") or 0)
        failure_count = int(payload.get("failure_count") or 0)
        if success_count and not failure_count:
            return f"好的，已处理{success_count}个设备。"
        if success_count:
            return f"已处理{success_count}个设备，{failure_count}个失败。"
        return "没有成功控制设备。"
    if status in ("verified", "ok", "resolved"):
        device = payload.get("device") if isinstance(payload.get("device"), dict) else {}
        name = str(device.get("name") or "")
        action = str(payload.get("action") or "")
        action_text = _action_done_text(action)
        if name and action_text:
            return f"好的，{name}已{action_text}。"
        if name:
            return f"好的，{name}已处理。"
        return "好的，已完成。"
    if payload.get("ok") is True:
        return "好的，已完成。"
    return ""


def _action_done_text(action: str) -> str:
    normalized = action.strip().lower()
    if normalized in ("turn_on", "on", "open", "打开", "开启"):
        return "打开"
    if normalized in ("turn_off", "off", "close", "关闭", "关掉"):
        return "关闭"
    if normalized in ("set_value", "set", "设置", "调到", "调成"):
        return "设置"
    return ""


def _get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_text(
    url: str,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> str:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc


def _required_text(arguments: dict[str, Any], key: str) -> str:
    value = str(arguments.get(key) or "").strip()
    if not value:
        raise RuntimeError(f"{key} is required")
    return value


def _optional_percent(
    arguments: dict[str, Any],
    key: str,
    aliases: tuple[str, ...] = (),
) -> float | None:
    for candidate in (key, *aliases):
        if candidate not in arguments or arguments[candidate] is None:
            continue
        value = arguments[candidate]
        if isinstance(value, str) and not value.strip():
            continue
        return float(value)
    return None


def _optional_bool(arguments: dict[str, Any], key: str) -> bool | None:
    if key not in arguments or arguments[key] is None:
        return None
    value = arguments[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on", "mute", "muted"):
            return True
        if lowered in ("false", "0", "no", "off", "unmute", "unmuted"):
            return False
    return bool(value)


def _optional_int(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} must be an integer") from exc


def _search_provider(arguments: dict[str, Any], config: SearchConfig) -> str:
    provider = str(arguments.get("provider") or config.provider or "auto").strip().lower()
    aliases = {
        "百度": "baidu",
        "bd": "baidu",
        "baidu_ai": "baidu",
        "baidu-ai": "baidu",
        "tavily": "tavily",
        "auto": "auto",
        "自动": "auto",
    }
    return aliases.get(provider, provider)


def _search_tavily(
    config: SearchConfig,
    arguments: dict[str, Any],
    query: str,
    max_results: int,
) -> dict[str, Any]:
    api_key = require_api_key(config.tavily_api_key_env)
    payload: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": str(arguments.get("search_depth") or config.tavily_search_depth),
        "topic": str(arguments.get("topic") or config.tavily_topic),
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }
    time_range = str(arguments.get("time_range") or "").strip()
    if time_range:
        payload["time_range"] = time_range
    for key in ("include_domains", "exclude_domains"):
        values = arguments.get(key)
        if isinstance(values, list) and values:
            payload[key] = [str(value) for value in values if str(value).strip()]

    data = post_json(
        config.tavily_endpoint,
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=config.timeout_seconds,
        error_prefix="Tavily search failed",
    )
    result = _tavily_result_from_response(data, query, max_results)
    if time_range and not result["answer"] and not result["results"]:
        retry_payload = dict(payload)
        retry_payload.pop("time_range", None)
        data = post_json(
            config.tavily_endpoint,
            retry_payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=config.timeout_seconds,
            error_prefix="Tavily search failed",
        )
        result = _tavily_result_from_response(data, query, max_results)
        result["retried_without_time_range"] = True
    return result


def _tavily_result_from_response(
    data: Any,
    query: str,
    max_results: int,
) -> dict[str, Any]:
    results = []
    raw_results = data.get("results") if isinstance(data, dict) else None
    if isinstance(raw_results, list):
        for item in raw_results[:max_results]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            results.append(
                {
                    "title": _clean_text(title),
                    "url": url,
                    "content": _clean_text(str(item.get("content") or "")),
                    "score": item.get("score"),
                    "provider": "tavily",
                }
            )
    return {
        "query": query,
        "provider": "tavily",
        "answer": _clean_text(str(data.get("answer") or "")) if isinstance(data, dict) else "",
        "results": results,
    }


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _log_search_result(result: dict[str, Any]) -> None:
    results = result.get("results") if isinstance(result.get("results"), list) else []
    log_event(
        "search",
        "completed",
        log_id="search.completed",
        provider=str(result.get("provider") or ""),
        query=str(result.get("query") or ""),
        results_count=len(results),
        has_answer=bool(str(result.get("answer") or "").strip()),
    )


def _search_baidu(
    config: SearchConfig,
    arguments: dict[str, Any],
    query: str,
    max_results: int,
) -> dict[str, Any]:
    if config.baidu_ai_enabled:
        api_key = optional_api_key(config.baidu_ai_api_key_env)
        if api_key:
            try:
                return _search_baidu_ai(config, query, max_results, api_key)
            except Exception as exc:
                fallback = _search_baidu_ai_fallback(config, arguments, query, max_results, exc)
                if fallback is not None:
                    return fallback
                raise
        elif not config.baidu_ai_fallback_to_html:
            require_api_key(config.baidu_ai_api_key_env)
    return _search_baidu_html(config, query, max_results)


def _search_baidu_ai_fallback(
    config: SearchConfig,
    arguments: dict[str, Any],
    query: str,
    max_results: int,
    error: Exception,
) -> dict[str, Any] | None:
    if optional_api_key(config.tavily_api_key_env):
        log_event(
            "search",
            "baidu_ai_fallback",
            log_id="search.baidu_ai_fallback",
            fallback_provider="tavily",
            reason=error,
        )
        try:
            return _search_tavily(config, arguments, query, max_results)
        except Exception:
            if not config.baidu_ai_fallback_to_html:
                raise

    if config.baidu_ai_fallback_to_html:
        log_event(
            "search",
            "baidu_ai_fallback",
            log_id="search.baidu_ai_fallback",
            fallback_provider="baidu_html",
            reason=error,
        )
        return _search_baidu_html(config, query, max_results)

    return None


def _search_baidu_ai(
    config: SearchConfig,
    query: str,
    max_results: int,
    api_key: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "model": config.baidu_ai_model,
        "search_mode": config.baidu_ai_search_mode,
        "enable_deep_search": bool(config.baidu_ai_deep_search),
        "enable_followup_queries": False,
        "response_format": "text",
        "resource_type_filter": [{"type": "web", "top_k": max_results}],
    }
    bearer = _bearer_token(api_key)
    data = post_json(
        config.baidu_ai_endpoint,
        payload,
        headers={
            "Authorization": bearer,
            "X-Appbuilder-Authorization": bearer,
        },
        timeout=config.timeout_seconds,
        error_prefix="Baidu AI search failed",
    )
    if not isinstance(data, dict):
        raise RuntimeError("Baidu AI search did not return an object")
    error_code = data.get("code")
    if error_code:
        message = str(data.get("message") or "unknown error")
        raise RuntimeError(f"Baidu AI search failed: {error_code}: {message}")

    answer = _baidu_ai_answer(data)
    results = _baidu_ai_results(data, max_results)
    if not answer and not results:
        raise RuntimeError("Baidu AI search returned no answer or references")
    return {
        "query": query,
        "provider": "baidu_ai",
        "answer": answer,
        "results": results,
        "request_id": str(data.get("request_id") or data.get("requestId") or ""),
        "is_safe": data.get("is_safe"),
        "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
    }


def _search_baidu_html(config: SearchConfig, query: str, max_results: int) -> dict[str, Any]:
    url = config.baidu_endpoint + "?" + urllib.parse.urlencode(
        _baidu_query_params(config.baidu_endpoint, query, max_results)
    )
    is_mobile = "m.baidu.com" in config.baidu_endpoint
    user_agent = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
        "AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"
        if is_mobile
        else (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/125 Safari/537.36"
        )
    )
    text = _get_text(
        url,
        timeout=config.timeout_seconds,
        headers={
            "User-Agent": user_agent,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    if "百度安全验证" in text or "mkdjump" in text:
        raise RuntimeError("Baidu search was blocked by security verification")
    results = _parse_baidu_results(text, max_results)
    if not results:
        raise RuntimeError("Baidu search returned no parseable results")
    return {
        "query": query,
        "provider": "baidu",
        "results": results,
    }


def _bearer_token(api_key: str) -> str:
    api_key = api_key.strip()
    if api_key.lower().startswith("bearer "):
        return api_key
    return f"Bearer {api_key}"


def _baidu_ai_answer(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    return _clean_text(str(message.get("content") or ""))


def _baidu_ai_results(data: dict[str, Any], max_results: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    raw_references = data.get("references")
    if not isinstance(raw_references, list):
        return results
    for item in raw_references[:max_results]:
        if not isinstance(item, dict):
            continue
        title = _clean_text(str(item.get("title") or ""))
        url = str(item.get("url") or "").strip()
        if not title or not url:
            continue
        results.append(
            {
                "title": title,
                "url": url,
                "content": _clean_text(str(item.get("content") or "")),
                "date": item.get("date"),
                "ref_id": item.get("id"),
                "web_anchor": _clean_text(str(item.get("web_anchor") or "")),
                "resource_type": item.get("type"),
                "provider": "baidu_ai",
            }
        )
    return results


def _baidu_query_params(endpoint: str, query: str, max_results: int) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.path.endswith("/baidu"):
        return {
            "word": query,
            "tn": "monline_3_dg",
            "ie": "utf-8",
            "rn": max_results,
        }
    if "m.baidu.com" in parsed.netloc:
        return {
            "word": query,
            "pn": 0,
            "rn": max_results,
        }
    return {
        "wd": query,
        "rn": max_results,
        "ie": "utf-8",
    }


def format_search_response(result: dict[str, Any]) -> str:
    answer = str(result.get("answer") or "").strip()
    if answer:
        return answer
    results = result.get("results") if isinstance(result.get("results"), list) else []
    items = [item for item in results if isinstance(item, dict)][:3]
    if not items:
        return "没有查到可用结果。"
    parts = []
    for index, item in enumerate(items, start=1):
        title = _clean_text(str(item.get("title") or ""))
        content = _clean_text(str(item.get("content") or ""))
        if content:
            parts.append(f"{index}. {title}：{content}")
        else:
            parts.append(f"{index}. {title}")
    return "我查到：" + "；".join(parts) + "。"


def _parse_baidu_results(text: str, max_results: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    _append_baidu_escaped_json_results(text, results, seen_urls, max_results)
    if len(results) >= max_results:
        return results

    clean = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)

    h3_pattern = (
        r"(?is)<h3[^>]*>.*?<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>"
        r"(.*?)</a>.*?</h3>"
    )
    for match in re.finditer(h3_pattern, clean):
        _append_baidu_result(results, seen_urls, match.group(2), match.group(1), "", max_results)
        if len(results) >= max_results:
            return results

    for match in re.finditer(r"(?is)<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", clean):
        url = html.unescape(match.group(1)).strip()
        title = _html_to_text(match.group(2))
        if len(title) < 4 or _looks_like_navigation_text(title):
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        _append_baidu_result(results, seen_urls, title, url, "", max_results)
        if len(results) >= max_results:
            break
    return results


def _append_baidu_escaped_json_results(
    text: str,
    results: list[dict[str, Any]],
    seen_urls: set[str],
    max_results: int,
) -> None:
    pattern = re.compile(
        r'\{\\"abstract\\":\\"(?P<abstract>(?:\\\\.|[^\\"])*)\\"'
        r'.*?\\"title\\":\\"(?P<title>(?:\\\\.|[^\\"])*)\\"'
        r'.*?\\"source\\":\{.*?\\"name\\":\\"(?P<source>(?:\\\\.|[^\\"])*)\\"'
        r'.*?\\"linkInfo\\":\{.*?\\"href\\":\\"(?P<href>http[^\\"]*)\\"',
        re.S,
    )
    for match in pattern.finditer(text):
        title = _decode_json_escaped_text(match.group("title"))
        abstract = _decode_json_escaped_text(match.group("abstract"))
        source = _decode_json_escaped_text(match.group("source"))
        href = _decode_json_escaped_text(match.group("href"))
        content = abstract or source
        _append_baidu_result(results, seen_urls, title, href, content, max_results)
        if len(results) >= max_results:
            break


def _append_baidu_result(
    results: list[dict[str, Any]],
    seen_urls: set[str],
    title_html: str,
    url: str,
    content_html: str,
    max_results: int,
) -> None:
    if len(results) >= max_results:
        return
    title = _html_to_text(title_html)
    href = html.unescape(url).strip()
    if not title or not href or href in seen_urls:
        return
    seen_urls.add(href)
    results.append(
        {
            "title": title,
            "url": href,
            "content": _html_to_text(content_html),
            "provider": "baidu",
        }
    )


def _html_to_text(value: str) -> str:
    without_tags = re.sub(r"(?is)<[^>]+>", " ", value)
    return _clean_text(html.unescape(without_tags))


def _decode_json_escaped_text(value: str) -> str:
    try:
        decoded = json.loads(f'"{value}"')
    except json.JSONDecodeError:
        decoded = value
    return _clean_text(str(decoded))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _looks_like_navigation_text(value: str) -> bool:
    normalized = value.strip().lower()
    if re.fullmatch(r"[\d:\s/.-]+", normalized):
        return True
    if "点击即刻体验ai搜索" in normalized:
        return True
    return normalized in {
        "百度首页",
        "登录",
        "设置",
        "更多",
        "更多产品",
        "下一页",
        "上一页",
        "feedback",
        "播放",
        "暂停",
        "抗击肺炎",
        "hao123",
    }


def _music_server(arguments: dict[str, Any], config: MusicConfig) -> str:
    return str(arguments.get("server") or config.server).strip() or "netease"


def _search_meting_music(config: MusicConfig, query: str, server: str) -> list[MusicTrack]:
    url = config.endpoint + "?" + urllib.parse.urlencode(
        {
            "server": server,
            "type": "search",
            "id": query,
        }
    )
    data = _get_json(url, timeout=config.timeout_seconds)
    if not isinstance(data, list):
        raise RuntimeError("Meting music search did not return a list")

    tracks: list[MusicTrack] = []
    for item in data[: max(1, config.max_results)]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        artist = str(item.get("author") or "").strip()
        playback_url = str(item.get("url") or "").strip()
        if not title or not playback_url:
            continue
        tracks.append(
            MusicTrack(
                title=title,
                artist=artist,
                playback_url=playback_url,
                provider=config.provider,
                server=server,
                cover_url=str(item.get("pic") or "").strip(),
                lyric_url=str(item.get("lrc") or "").strip(),
            )
        )
    return tracks


def _download_audio(url: str, timeout: float, max_bytes: int) -> tuple[bytes, str, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"Audio response exceeded limit: {max_bytes} bytes")
            chunks.append(chunk)
        audio_data = b"".join(chunks)
        if not audio_data:
            raise RuntimeError("Music provider returned empty audio data")
        content_type = str(response.headers.get("Content-Type") or "")
        final_url = response.geturl()
    return audio_data, _audio_format_from_response(content_type, final_url, audio_data), final_url


def _audio_format_from_response(content_type: str, url: str, audio_data: bytes) -> str:
    content_type = content_type.lower()
    if "mpeg" in content_type or "mp3" in content_type or audio_data.startswith(b"ID3"):
        return "mp3"
    if "flac" in content_type or audio_data.startswith(b"fLaC"):
        return "flac"
    if "wav" in content_type or audio_data.startswith(b"RIFF"):
        return "wav"

    path = urllib.parse.urlparse(url).path.lower()
    for extension in (".mp3", ".flac", ".wav", ".ogg", ".opus"):
        if path.endswith(extension):
            return extension.lstrip(".")
    return "audio"


def _play_decoded_audio(
    audio_data: bytes,
    audio_format: str,
    device: str | int | None = None,
    playback_sample_rate: int | None = None,
    playback_channels: int | None = None,
    playback_volume: float = 1.0,
    limiter_enabled: bool = True,
    limiter_threshold: float = 0.92,
    dynamic_volume_getter: Callable[[], float] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    del audio_format
    try:
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Music playback requires soundfile. Install with: pip install -e \".[tts]\""
        ) from exc

    from voiceui.tts import _float_audio_to_pcm16, _play_pcm_stream

    data, sample_rate = sf.read(io.BytesIO(audio_data), dtype="float32", always_2d=True)
    if playback_volume != 1.0:
        data = data * float(playback_volume)
    if limiter_enabled:
        data, peak, gain = _limit_float_audio(data, limiter_threshold)
        if gain < 0.999:
            log_continuous(
                "music",
                "limiter",
                log_id="music.limiter",
                peak=f"{peak:.3f}",
                threshold=f"{float(limiter_threshold):.3f}",
                gain=f"{gain:.3f}",
            )
    pcm, channels = _float_audio_to_pcm16(data)
    pcm_chunks = _iter_dynamic_volume_pcm_chunks(
        pcm,
        sample_rate=sample_rate,
        channels=channels,
        dynamic_volume_getter=dynamic_volume_getter,
        stop_event=stop_event,
    )
    _play_pcm_stream(
        pcm_chunks,
        sample_rate=sample_rate,
        source_channels=channels,
        device=device,
        playback_sample_rate=playback_sample_rate,
        playback_channels=playback_channels,
        stop_event=stop_event,
        dump_kind="music_output",
    )


def _iter_dynamic_volume_pcm_chunks(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    dynamic_volume_getter: Callable[[], float] | None = None,
    stop_event: threading.Event | None = None,
):
    from voiceui.tts import _iter_pcm_chunks, _stop_requested

    for chunk in _iter_pcm_chunks(
        pcm,
        sample_rate=sample_rate,
        channels=channels,
    ):
        if _stop_requested(stop_event):
            break
        factor = 1.0
        if dynamic_volume_getter is not None:
            factor = max(0.0, float(dynamic_volume_getter()))
        if factor != 1.0:
            chunk = _scale_pcm16_volume(chunk, factor)
        yield chunk


def _limit_float_audio(data, threshold: float = 0.92):
    import numpy as np  # type: ignore[import-untyped]

    limit = max(0.05, min(1.0, float(threshold)))
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= limit or peak <= 0.0:
        return data, peak, 1.0
    gain = limit / peak
    return data * gain, peak, gain


def _scale_pcm16_volume(pcm: bytes, playback_volume: float) -> bytes:
    volume = float(playback_volume)
    if volume < 0:
        raise ValueError("playback_volume must be >= 0.")
    if not pcm or volume == 1.0:
        return pcm

    playable_len = len(pcm) - (len(pcm) % 2)
    samples = array.array("h")
    samples.frombytes(pcm[:playable_len])
    if sys.byteorder != "little":
        samples.byteswap()
    for index, sample in enumerate(samples):
        samples[index] = max(-32768, min(32767, int(sample * volume)))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes() + pcm[playable_len:]


def _to_json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def _is_wet_weather(code: object, precipitation: object) -> bool:
    try:
        if float(precipitation or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(code, int) and code in {
        51,
        53,
        55,
        56,
        57,
        61,
        63,
        65,
        66,
        67,
        71,
        73,
        75,
        77,
        80,
        81,
        82,
        85,
        86,
        95,
        96,
        99,
    }


def _is_high_precipitation_probability(probability: object) -> bool:
    try:
        return float(probability or 0) >= 50
    except (TypeError, ValueError):
        return False


def _weather_code_zh(code: object) -> str:
    descriptions = {
        0: "晴",
        1: "大致晴朗",
        2: "局部多云",
        3: "阴天",
        45: "有雾",
        48: "有雾凇",
        51: "小毛毛雨",
        53: "中等毛毛雨",
        55: "较强毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        80: "小阵雨",
        81: "中等阵雨",
        82: "强阵雨",
        95: "雷雨",
        96: "雷雨伴小冰雹",
        99: "雷雨伴强冰雹",
    }
    if isinstance(code, int):
        return descriptions.get(code, f"天气代码{code}")
    return "天气情况未知"


def _weather_code_text(code: object) -> str:
    descriptions = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    if isinstance(code, int):
        return descriptions.get(code, f"weather code {code}")
    return "unknown"
