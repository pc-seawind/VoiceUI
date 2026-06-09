from __future__ import annotations

import os
import threading
import time
import unittest
from unittest.mock import patch

import voiceui.tools as tools_module
from voiceui.llm import ChatClient, ChatMessage, ToolCall, ToolChatResponse
from voiceui.models import (
    AssistantConfig,
    MusicConfig,
    SearchConfig,
    ToolsConfig,
    XiaomiMiotConfig,
)
from voiceui.tools import (
    MusicPlaybackController,
    MusicTrack,
    ToolDefinition,
    VoiceToolRunner,
    _iter_dynamic_volume_pcm_chunks,
    _limit_float_audio,
    create_set_system_volume_tool,
    create_tool_runner,
    get_current_time,
    get_current_weather,
    search_music_tracks,
    search_web,
)


class FakeToolChat(ChatClient):
    def __init__(self):
        self.calls: list[tuple[list[ChatMessage], list[dict]]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        return "unused"

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        self.calls.append((list(messages), tools))
        if len(self.calls) == 1:
            return ToolChatResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="demo_tool",
                        arguments={"value": 3},
                        raw={
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "demo_tool",
                                "arguments": "{\"value\":3}",
                            },
                        },
                    )
                ]
            )
        return ToolChatResponse(content="done")


class RepeatingToolChat(ChatClient):
    def __init__(self, tool_name: str = "demo_tool"):
        self.tool_name = tool_name

    def complete(self, messages: list[ChatMessage]) -> str:
        return "unused"

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        return ToolChatResponse(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name=self.tool_name,
                    arguments={},
                    raw={
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": self.tool_name, "arguments": "{}"},
                    },
                )
            ]
        )


class MiotFirstToolThenNoToolChat(ChatClient):
    def __init__(self):
        self.calls: list[tuple[list[ChatMessage], list[dict]]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        return "unused"

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        self.calls.append((list(messages), tools))
        if len(self.calls) == 1:
            return ToolChatResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="xiaomi_miot_control_device",
                        arguments={
                            "request": "打开书房灯",
                            "area": "书房",
                            "device": "灯",
                            "action": "turn_on",
                        },
                        raw={
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "xiaomi_miot_control_device",
                                "arguments": (
                                    '{"request":"打开书房灯","area":"书房",'
                                    '"device":"灯","action":"turn_on"}'
                                ),
                            },
                        },
                    )
                ]
            )
        return ToolChatResponse(content="好的，书房的灯已经关闭了。")


class MiotAmbiguousToolChat(ChatClient):
    def __init__(self):
        self.calls: list[tuple[list[ChatMessage], list[dict]]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        return "unused"

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        self.calls.append((list(messages), tools))
        return ToolChatResponse(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="xiaomi_miot_control_device",
                    arguments={
                        "request": "关闭空调",
                        "device": "空调",
                        "action": "turn_off",
                    },
                    raw={
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "xiaomi_miot_control_device",
                            "arguments": (
                                '{"request":"关闭空调","device":"空调",'
                                '"action":"turn_off"}'
                            ),
                        },
                    },
                )
            ]
        )


class MiotReadAmbiguousToolChat(ChatClient):
    def __init__(self):
        self.calls: list[tuple[list[ChatMessage], list[dict]]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        return "unused"

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        self.calls.append((list(messages), tools))
        return ToolChatResponse(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="xiaomi_miot_read_device_property",
                    arguments={
                        "request": "看一下空调状态",
                        "device": "空调",
                        "property_query": "power",
                    },
                    raw={
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "xiaomi_miot_read_device_property",
                            "arguments": (
                                '{"request":"看一下空调状态","device":"空调",'
                                '"property_query":"power"}'
                            ),
                        },
                    },
                )
            ]
        )


class NoToolChat(ChatClient):
    def __init__(self):
        self.calls: list[tuple[list[ChatMessage], list[dict]]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        return "unused"

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        self.calls.append((list(messages), tools))
        return ToolChatResponse(content="好的，书房的灯已经关闭了。")


class DirectChat(ChatClient):
    def __init__(self):
        self.complete_calls: list[list[ChatMessage]] = []
        self.tool_calls: list[tuple[list[ChatMessage], list[dict]]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.complete_calls.append(list(messages))
        return "plain reply"

    def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
    ) -> ToolChatResponse:
        self.tool_calls.append((list(messages), tools))
        return ToolChatResponse(content="tool reply")


class ToolsTests(unittest.TestCase):
    def test_tool_runner_adds_assistant_and_tool_messages(self) -> None:
        chat = FakeToolChat()
        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="demo_tool",
                    description="Demo",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda args: {"doubled": args["value"] * 2},
                )
            ],
        )

        response = runner.complete([ChatMessage(role="user", content="run")])

        self.assertEqual(response, "done")
        self.assertEqual(len(chat.calls), 2)
        second_messages = chat.calls[1][0]
        self.assertEqual(second_messages[1].role, "assistant")
        self.assertEqual(second_messages[1].tool_calls[0]["id"], "call_1")
        self.assertEqual(second_messages[2].role, "tool")
        self.assertEqual(second_messages[2].tool_call_id, "call_1")
        self.assertIn('"doubled":6', second_messages[2].content)

    def test_tool_runner_fallback_summarizes_last_successful_tool(self) -> None:
        runner = VoiceToolRunner(
            chat=RepeatingToolChat(),
            tools=[
                ToolDefinition(
                    name="demo_tool",
                    description="Demo",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _args: {
                        "status": "verified",
                        "action": "turn_off",
                        "device": {"name": "书房吸顶灯"},
                    },
                )
            ],
            max_iterations=1,
        )

        response = runner.complete([ChatMessage(role="user", content="关灯")])

        self.assertEqual(response, "好的，书房吸顶灯已关闭。")

    def test_tool_runner_fallback_summarizes_ambiguous_tool(self) -> None:
        runner = VoiceToolRunner(
            chat=RepeatingToolChat(),
            tools=[
                ToolDefinition(
                    name="demo_tool",
                    description="Demo",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _args: {
                        "status": "ambiguous",
                        "candidates": [{"name": "书房吸顶灯"}, {"name": "书房台灯"}],
                    },
                )
            ],
            max_iterations=1,
        )

        response = runner.complete([ChatMessage(role="user", content="开灯")])

        self.assertIn("书房吸顶灯", response)
        self.assertIn("书房台灯", response)

    def test_tool_runner_reuses_last_miot_device_for_pronoun_followup(self) -> None:
        chat = MiotFirstToolThenNoToolChat()
        tool_calls: list[dict] = []

        def control_device(arguments: dict) -> dict:
            tool_calls.append(dict(arguments))
            return {
                "status": "verified",
                "action": arguments["action"],
                "device": {
                    "name": "书房吸顶灯",
                    "room_name": "书房",
                    "device_class": "light",
                },
            }

        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=control_device,
                )
            ],
        )

        first = runner.complete([ChatMessage(role="user", content="打开书房灯")])
        second = runner.complete(
            [
                ChatMessage(role="user", content="打开书房灯"),
                ChatMessage(role="assistant", content=first),
                ChatMessage(role="user", content="再把它关了。"),
            ]
        )

        self.assertEqual(first, "好的，书房吸顶灯已打开。")
        self.assertEqual(second, "好的，书房吸顶灯已关闭。")
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(tool_calls[1]["area"], "书房")
        self.assertEqual(tool_calls[1]["device"], "书房吸顶灯")
        self.assertEqual(tool_calls[1]["device_class"], "light")
        self.assertEqual(tool_calls[1]["action"], "turn_off")

    def test_tool_runner_uses_ambiguous_miot_context_for_pronoun_followup(self) -> None:
        chat = MiotAmbiguousToolChat()
        tool_calls: list[dict] = []

        def control_device(arguments: dict) -> dict:
            tool_calls.append(dict(arguments))
            if len(tool_calls) == 1:
                return {
                    "status": "ambiguous",
                    "candidates": [
                        {
                            "name": "书房空调",
                            "room_name": "书房",
                            "device_class": "aircondition",
                        },
                        {
                            "name": "客厅空调",
                            "room_name": "客厅",
                            "device_class": "aircondition",
                        },
                    ],
                    "query": {
                        "device": "空调",
                        "device_class": "aircondition",
                        "action": "turn_off",
                    },
                }
            return {
                "status": "verified",
                "action": arguments["action"],
                "device": {
                    "name": "书房空调",
                    "room_name": "书房",
                    "device_class": "aircondition",
                },
            }

        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=control_device,
                )
            ],
        )
        runner._last_miot_control = {  # pylint: disable=protected-access
            "device": {
                "name": "书房吸顶灯",
                "room_name": "书房",
                "device_class": "light",
            },
            "action": "turn_on",
        }

        first = runner.complete([ChatMessage(role="user", content="关闭空调")])
        second = runner.complete(
            [
                ChatMessage(role="user", content="关闭空调"),
                ChatMessage(role="assistant", content=first),
                ChatMessage(role="user", content="把它关了。"),
            ]
        )

        self.assertIn("书房空调", first)
        self.assertEqual(second, "好的，书房空调已关闭。")
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(tool_calls[1]["request"], "把它关了。")
        self.assertEqual(tool_calls[1]["device"], "空调")
        self.assertEqual(tool_calls[1]["device_class"], "aircondition")
        self.assertEqual(tool_calls[1]["action"], "turn_off")

    def test_tool_runner_uses_ambiguous_miot_context_for_ordinal_selection(self) -> None:
        chat = MiotAmbiguousToolChat()
        tool_calls: list[dict] = []

        def control_device(arguments: dict) -> dict:
            tool_calls.append(dict(arguments))
            if len(tool_calls) == 1:
                return {
                    "status": "ambiguous",
                    "candidates": [
                        {
                            "name": "书房空调",
                            "room_name": "书房",
                            "device_class": "aircondition",
                        },
                        {
                            "name": "客厅空调",
                            "room_name": "客厅",
                            "device_class": "aircondition",
                        },
                    ],
                    "query": {
                        "device": "空调",
                        "device_class": "aircondition",
                        "action": "turn_off",
                    },
                }
            return {
                "status": "verified",
                "action": arguments["action"],
                "device": {
                    "name": arguments["device"],
                    "room_name": arguments["area"],
                    "device_class": arguments["device_class"],
                },
            }

        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=control_device,
                )
            ],
        )

        first = runner.complete([ChatMessage(role="user", content="关闭空调")])
        second = runner.complete(
            [
                ChatMessage(role="user", content="关闭空调"),
                ChatMessage(role="assistant", content=first),
                ChatMessage(role="user", content="第一个。"),
            ]
        )

        self.assertEqual(second, "好的，书房空调已关闭。")
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(tool_calls[1]["request"], "第一个。")
        self.assertEqual(tool_calls[1]["area"], "书房")
        self.assertEqual(tool_calls[1]["device"], "书房空调")
        self.assertEqual(tool_calls[1]["device_class"], "aircondition")
        self.assertEqual(tool_calls[1]["action"], "turn_off")

    def test_tool_runner_ignores_expired_ambiguous_miot_context(self) -> None:
        chat = MiotAmbiguousToolChat()
        tool_calls: list[dict] = []
        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda arguments: tool_calls.append(dict(arguments)),
                )
            ],
        )
        runner._last_miot_ambiguity = {  # pylint: disable=protected-access
            "candidates": [
                {
                    "name": "书房空调",
                    "room_name": "书房",
                    "device_class": "aircondition",
                },
                {
                    "name": "客厅空调",
                    "room_name": "客厅",
                    "device_class": "aircondition",
                },
            ],
            "query": {
                "device": "空调",
                "device_class": "aircondition",
                "action": "turn_off",
            },
            "created_at": time.monotonic() - 10,
            "ttl_seconds": 1,
        }

        response = runner.complete([ChatMessage(role="user", content="第一个。")])

        self.assertEqual(response, "unused")
        self.assertEqual(tool_calls, [])
        self.assertIsNone(runner._last_miot_ambiguity)  # pylint: disable=protected-access

    def test_tool_runner_uses_ambiguous_miot_context_for_room_selection(self) -> None:
        chat = MiotAmbiguousToolChat()
        tool_calls: list[dict] = []

        def control_device(arguments: dict) -> dict:
            tool_calls.append(dict(arguments))
            if len(tool_calls) == 1:
                return {
                    "status": "ambiguous",
                    "candidates": [
                        {
                            "name": "书房空调",
                            "room_name": "书房",
                            "device_class": "aircondition",
                        },
                        {
                            "name": "客厅空调",
                            "room_name": "客厅",
                            "device_class": "aircondition",
                        },
                    ],
                    "query": {
                        "device": "空调",
                        "device_class": "aircondition",
                        "action": "turn_off",
                    },
                }
            return {
                "status": "verified",
                "action": arguments["action"],
                "device": {
                    "name": arguments["device"],
                    "room_name": arguments["area"],
                    "device_class": arguments["device_class"],
                },
            }

        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=control_device,
                )
            ],
        )

        first = runner.complete([ChatMessage(role="user", content="关闭空调")])
        second = runner.complete(
            [
                ChatMessage(role="user", content="关闭空调"),
                ChatMessage(role="assistant", content=first),
                ChatMessage(role="user", content="客厅那个。"),
            ]
        )

        self.assertEqual(second, "好的，客厅空调已关闭。")
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(tool_calls[1]["request"], "客厅那个。")
        self.assertEqual(tool_calls[1]["area"], "客厅")
        self.assertEqual(tool_calls[1]["device"], "客厅空调")
        self.assertEqual(tool_calls[1]["device_class"], "aircondition")
        self.assertEqual(tool_calls[1]["action"], "turn_off")

    def test_tool_runner_uses_ambiguous_read_context_for_room_selection(self) -> None:
        chat = MiotReadAmbiguousToolChat()
        tool_calls: list[dict] = []

        def read_device(arguments: dict) -> dict:
            tool_calls.append(dict(arguments))
            if len(tool_calls) == 1:
                return {
                    "status": "ambiguous",
                    "candidates": [
                        {
                            "name": "书房空调",
                            "room_name": "书房",
                            "device_class": "aircondition",
                        },
                        {
                            "name": "客厅空调",
                            "room_name": "客厅",
                            "device_class": "aircondition",
                        },
                    ],
                    "query": {
                        "device": "空调",
                        "device_class": "aircondition",
                        "property": "power",
                    },
                }
            return {
                "status": "property_read",
                "device": {"name": arguments["device"]},
                "property": {"name": "power", "description": "开关"},
                "value": True,
                "direct_response": "书房空调现在开着。",
            }

        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_read_device_property",
                    description="Read Xiaomi device property",
                    parameters={"type": "object", "properties": {}},
                    handler=read_device,
                )
            ],
        )

        first = runner.complete([ChatMessage(role="user", content="看一下空调状态")])
        second = runner.complete(
            [
                ChatMessage(role="user", content="看一下空调状态"),
                ChatMessage(role="assistant", content=first),
                ChatMessage(role="user", content="书房的空调。"),
            ]
        )

        self.assertIn("书房空调", first)
        self.assertEqual(second, "书房空调现在开着。")
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(tool_calls[1]["request"], "书房的空调。")
        self.assertEqual(tool_calls[1]["area"], "书房")
        self.assertEqual(tool_calls[1]["device"], "书房空调")
        self.assertEqual(tool_calls[1]["device_class"], "aircondition")
        self.assertEqual(tool_calls[1]["property_query"], "power")

    def test_tool_runner_reuses_last_miot_device_for_temperature_followup(self) -> None:
        tool_calls: list[dict] = []

        def control_device(arguments: dict) -> dict:
            tool_calls.append(dict(arguments))
            return {
                "status": "verified",
                "action": arguments["action"],
                "device": {
                    "name": "书房空调",
                    "room_name": "书房",
                    "device_class": "aircondition",
                },
                "target_value": 27,
            }

        runner = VoiceToolRunner(
            chat=NoToolChat(),
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=control_device,
                )
            ],
        )
        runner._last_miot_control = {  # pylint: disable=protected-access
            "device": {
                "name": "书房空调",
                "room_name": "书房",
                "device_class": "aircondition",
            },
            "action": "turn_on",
        }

        response = runner.complete([ChatMessage(role="user", content="温度调成27度。")])

        self.assertEqual(response, "好的，书房空调已设置。")
        self.assertEqual(tool_calls[0]["request"], "温度调成27度。")
        self.assertEqual(tool_calls[0]["area"], "书房")
        self.assertEqual(tool_calls[0]["device"], "书房空调")
        self.assertEqual(tool_calls[0]["device_class"], "aircondition")
        self.assertEqual(tool_calls[0]["action"], "set_value")

    def test_tool_runner_uses_ambiguous_miot_context_for_group_followup(self) -> None:
        chat = MiotAmbiguousToolChat()
        tool_calls: list[dict] = []

        def control_device(arguments: dict) -> dict:
            tool_calls.append(dict(arguments))
            if len(tool_calls) == 1:
                return {
                    "status": "ambiguous",
                    "candidates": [
                        {"name": "书房空调", "room_name": "书房", "device_class": "aircondition"},
                        {"name": "客厅空调", "room_name": "客厅", "device_class": "aircondition"},
                    ],
                    "query": {
                        "device": "空调",
                        "device_class": "aircondition",
                        "action": "turn_off",
                    },
                }
            return {
                "status": "group_executed",
                "action": arguments["action"],
                "success_count": 2,
                "failure_count": 0,
                "direct_response": "好的，已关闭2个设备。",
            }

        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=control_device,
                )
            ],
        )

        first = runner.complete([ChatMessage(role="user", content="关闭空调")])
        second = runner.complete(
            [
                ChatMessage(role="user", content="关闭空调"),
                ChatMessage(role="assistant", content=first),
                ChatMessage(role="user", content="都关了。"),
            ]
        )

        self.assertEqual(second, "好的，已关闭2个设备。")
        self.assertEqual(tool_calls[1]["request"], "都关了。")
        self.assertEqual(tool_calls[1]["device"], "空调")
        self.assertEqual(tool_calls[1]["device_class"], "aircondition")
        self.assertEqual(tool_calls[1]["action"], "turn_off")

    def test_tool_runner_uses_previous_ambiguous_context_for_correction(self) -> None:
        chat = MiotAmbiguousToolChat()
        tool_calls: list[dict] = []

        def control_device(arguments: dict) -> dict:
            tool_calls.append(dict(arguments))
            if len(tool_calls) == 1:
                return {
                    "status": "ambiguous",
                    "candidates": [
                        {
                            "name": "书房空调",
                            "room_name": "书房",
                            "device_class": "aircondition",
                        },
                        {
                            "name": "客厅空调",
                            "room_name": "客厅",
                            "device_class": "aircondition",
                        },
                    ],
                    "query": {
                        "device": "空调",
                        "device_class": "aircondition",
                        "action": "turn_off",
                    },
                }
            return {
                "status": "verified",
                "action": arguments["action"],
                "device": {
                    "name": arguments["device"],
                    "room_name": arguments["area"],
                    "device_class": arguments["device_class"],
                },
            }

        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=control_device,
                )
            ],
        )

        first = runner.complete([ChatMessage(role="user", content="关闭空调")])
        second = runner.complete(
            [
                ChatMessage(role="user", content="关闭空调"),
                ChatMessage(role="assistant", content=first),
                ChatMessage(role="user", content="第一个。"),
            ]
        )
        third = runner.complete(
            [
                ChatMessage(role="user", content="关闭空调"),
                ChatMessage(role="assistant", content=first),
                ChatMessage(role="user", content="第一个。"),
                ChatMessage(role="assistant", content=second),
                ChatMessage(role="user", content="不是这个。"),
            ]
        )

        self.assertEqual(second, "好的，书房空调已关闭。")
        self.assertEqual(third, "好的，客厅空调已关闭。")
        self.assertEqual(tool_calls[2]["request"], "不是这个。")
        self.assertEqual(tool_calls[2]["area"], "客厅")
        self.assertEqual(tool_calls[2]["device"], "客厅空调")
        self.assertEqual(tool_calls[2]["action"], "turn_off")

    def test_tool_runner_does_not_confirm_miot_control_without_tool_call(self) -> None:
        chat = NoToolChat()
        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="xiaomi_miot_control_device",
                    description="Control Xiaomi device",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _arguments: {"status": "verified"},
                )
            ],
        )

        response = runner.complete([ChatMessage(role="user", content="再把它关了。")])

        self.assertEqual(
            response,
            "我还没有实际执行到设备控制，不能确认已经完成。请再说一遍具体设备。",
        )
        self.assertEqual(len(chat.calls), 1)

    def test_tool_runner_uses_plain_llm_when_no_tool_intent_matches(self) -> None:
        chat = DirectChat()
        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="get_current_time",
                    description="Current time",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _arguments: {},
                ),
                ToolDefinition(
                    name="web_search",
                    description="Search web",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _arguments: {},
                ),
            ],
        )

        response = runner.complete([ChatMessage(role="user", content="讲个笑话")])

        self.assertEqual(response, "plain reply")
        self.assertEqual(len(chat.complete_calls), 1)
        self.assertEqual(chat.tool_calls, [])

    def test_tool_runner_sends_only_matching_tool_payloads(self) -> None:
        chat = DirectChat()
        runner = VoiceToolRunner(
            chat=chat,
            tools=[
                ToolDefinition(
                    name="get_current_time",
                    description="Current time",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _arguments: {},
                ),
                ToolDefinition(
                    name="web_search",
                    description="Search web",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _arguments: {},
                ),
            ],
        )

        response = runner.complete([ChatMessage(role="user", content="现在几点")])

        self.assertEqual(response, "tool reply")
        self.assertEqual(chat.complete_calls, [])
        self.assertEqual(len(chat.tool_calls), 1)
        tool_names = [payload["function"]["name"] for payload in chat.tool_calls[0][1]]
        self.assertEqual(tool_names, ["get_current_time"])

    def test_current_time_uses_requested_timezone(self) -> None:
        result = get_current_time({"timezone": "Asia/Shanghai"})

        self.assertEqual(result["timezone"], "Asia/Shanghai")
        self.assertIn("+08:00", result["datetime"])

    def test_current_weather_uses_geocoding_and_forecast(self) -> None:
        responses = [
            {
                "results": [
                    {
                        "name": "Test City",
                        "country": "Test Country",
                        "admin1": "Test Admin",
                        "latitude": 31.23,
                        "longitude": 121.47,
                    }
                ]
            },
            {
                "timezone": "Asia/Shanghai",
                "current": {
                    "temperature_2m": 26.5,
                    "relative_humidity_2m": 70,
                    "weather_code": 2,
                },
                "current_units": {"temperature_2m": "°C"},
            },
        ]

        with patch.dict(tools_module._WEATHER_GEOCODE_CACHE, {}, clear=True):
            with patch.dict(tools_module._WEATHER_FORECAST_CACHE, {}, clear=True):
                with patch("voiceui.tools._load_weather_disk_cache"):
                    with patch("voiceui.tools._save_weather_disk_cache"):
                        with patch("voiceui.tools._get_json", side_effect=responses) as get_json:
                            result = get_current_weather({"location": "Test City"})

        self.assertEqual(get_json.call_count, 2)
        self.assertEqual(result["location"]["name"], "Test City")
        self.assertEqual(result["location"]["timezone"], "Asia/Shanghai")
        self.assertEqual(result["current"]["temperature_2m"], 26.5)
        self.assertEqual(result["summary"], "partly cloudy")

    def test_current_weather_can_format_tomorrow_forecast(self) -> None:
        responses = [
            {
                "results": [
                    {
                        "name": "Test City",
                        "country": "Test Country",
                        "admin1": "Test Admin",
                        "latitude": 31.23,
                        "longitude": 121.47,
                    }
                ]
            },
            {
                "timezone": "Asia/Shanghai",
                "current": {
                    "temperature_2m": 26.5,
                    "relative_humidity_2m": 70,
                    "weather_code": 2,
                },
                "current_units": {"temperature_2m": "°C"},
                "daily": {
                    "time": ["2026-06-03", "2026-06-04"],
                    "weather_code": [2, 61],
                    "temperature_2m_max": [28.0, 30.0],
                    "temperature_2m_min": [20.0, 21.0],
                    "precipitation_sum": [0.0, 2.5],
                    "precipitation_probability_max": [10, 70],
                },
                "daily_units": {
                    "temperature_2m_max": "°C",
                    "temperature_2m_min": "°C",
                },
            },
        ]

        with patch.dict(tools_module._WEATHER_GEOCODE_CACHE, {}, clear=True):
            with patch.dict(tools_module._WEATHER_FORECAST_CACHE, {}, clear=True):
                with patch("voiceui.tools._load_weather_disk_cache"):
                    with patch("voiceui.tools._save_weather_disk_cache"):
                        with patch("voiceui.tools._get_json", side_effect=responses):
                            result = get_current_weather(
                                {"location": "Test City", "target_day": "tomorrow"}
                            )

        self.assertEqual(result["target_day"], "tomorrow")
        self.assertEqual(result["daily"]["weather_code"], 61)
        self.assertIn("明天小雨", result["direct_response"])
        self.assertIn("最高30°C", result["direct_response"])
        self.assertIn("最低21°C", result["direct_response"])
        self.assertIn("降水概率70%", result["direct_response"])
        self.assertIn("带伞", result["direct_response"])

    def test_meting_music_search_uses_configured_provider(self) -> None:
        config = MusicConfig(
            provider="meting",
            endpoint="https://music.example/api",
            server="netease",
            max_results=1,
        )
        response = [
            {
                "title": "Song",
                "author": "Artist",
                "url": "https://music.example/song.mp3",
                "pic": "https://music.example/song.jpg",
                "lrc": "https://music.example/song.lrc",
            }
        ]

        with patch("voiceui.tools._get_json", return_value=response) as get_json:
            tracks = search_music_tracks(config, "song")

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].title, "Song")
        self.assertEqual(tracks[0].artist, "Artist")
        called_url = get_json.call_args.args[0]
        self.assertIn("server=netease", called_url)
        self.assertIn("type=search", called_url)
        self.assertIn("id=song", called_url)

    def test_music_play_resolves_track_when_playback_disabled(self) -> None:
        controller = MusicPlaybackController(
            MusicConfig(provider="meting", playback_enabled=False)
        )
        track = MusicTrack(
            title="Song",
            artist="Artist",
            playback_url="https://music.example/song.mp3",
            provider="meting",
            server="netease",
        )

        with patch("voiceui.tools.search_music_tracks", return_value=[track]) as search:
            result = controller.play({"query": "song"})

        search.assert_called_once()
        self.assertEqual(result["status"], "resolved")
        self.assertFalse(result["playback_enabled"])
        self.assertEqual(result["track"]["title"], "Song")

    def test_music_stop_keeps_active_thread_until_it_finishes(self) -> None:
        controller = MusicPlaybackController(MusicConfig(provider="meting"))
        stop_event = threading.Event()
        track = MusicTrack(
            title="Song",
            artist="Artist",
            playback_url="https://music.example/song.mp3",
            provider="meting",
            server="netease",
        )

        class AliveThread:
            def is_alive(self):
                return True

            def join(self, timeout=None):
                return None

        with controller._lock:
            controller._stop_event = stop_event
            controller._thread = AliveThread()
            controller._current_track = track

        result = controller.stop(wait=False)

        self.assertEqual(result["status"], "stopping")
        self.assertTrue(stop_event.is_set())
        self.assertIs(controller._stop_event, stop_event)
        self.assertIsNotNone(controller._current_track)

    def test_music_duck_factor_is_dynamic(self) -> None:
        controller = MusicPlaybackController(
            MusicConfig(provider="meting", ducking_volume_factor=0.2)
        )

        self.assertEqual(controller.current_volume_factor(), 1.0)
        controller.duck("test")
        self.assertEqual(controller.current_volume_factor(), 0.2)
        controller.unduck("test")
        self.assertEqual(controller.current_volume_factor(), 1.0)

    def test_tool_runner_registers_music_tools_when_enabled(self) -> None:
        config = AssistantConfig(
            tools=ToolsConfig(
                enabled=True,
                allow_time=False,
                allow_weather=False,
                allow_music=True,
            ),
            music=MusicConfig(provider="meting", playback_enabled=False),
        )

        runner = create_tool_runner(config, FakeToolChat())

        self.assertIsNotNone(runner)
        names = [tool["function"]["name"] for tool in runner.tool_payloads]
        self.assertEqual(names, ["search_music", "play_music", "stop_music"])

    def test_tool_runner_registers_volume_tools_when_enabled(self) -> None:
        config = AssistantConfig(
            tools=ToolsConfig(
                enabled=True,
                allow_time=False,
                allow_weather=False,
                allow_volume=True,
                allow_music=False,
            )
        )

        runner = create_tool_runner(config, FakeToolChat())

        self.assertIsNotNone(runner)
        names = [tool["function"]["name"] for tool in runner.tool_payloads]
        self.assertEqual(names, ["get_system_volume", "set_system_volume"])

    def test_tool_runner_registers_xiaomi_miot_tools_when_enabled(self) -> None:
        config = AssistantConfig(
            tools=ToolsConfig(
                enabled=True,
                allow_time=False,
                allow_weather=False,
                allow_volume=False,
                allow_music=False,
                allow_miot=True,
            ),
            xiaomi_miot=XiaomiMiotConfig(enabled=True),
        )

        runner = create_tool_runner(config, FakeToolChat())

        self.assertIsNotNone(runner)
        names = [tool["function"]["name"] for tool in runner.tool_payloads]
        self.assertEqual(
            names,
            [
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
            ],
        )

    def test_tool_selection_routes_home_air_quality_to_miot_not_search(self) -> None:
        available_names = {
            "web_search",
            "xiaomi_miot_read_device_property",
            "xiaomi_miot_get_devices",
            "xiaomi_miot_get_device_spec",
            "xiaomi_miot_get_property",
        }

        selected = tools_module._select_tool_names_for_text(
            "查一下我们家里的空气净化器显示的空气质量",
            available_names,
        )

        self.assertNotIn("web_search", selected)
        self.assertIn("xiaomi_miot_read_device_property", selected)
        self.assertIn("xiaomi_miot_get_devices", selected)

    def test_tool_selection_routes_air_conditioner_alias_to_miot(self) -> None:
        available_names = {
            "xiaomi_miot_control_device",
            "xiaomi_miot_get_devices",
            "xiaomi_miot_get_device_spec",
        }

        selected = tools_module._select_tool_names_for_text("打开客厅冷气", available_names)

        self.assertIn("xiaomi_miot_control_device", selected)
        self.assertIn("xiaomi_miot_get_devices", selected)

    def test_tool_selection_routes_air_conditioner_temperature_to_miot_not_weather(
        self,
    ) -> None:
        available_names = {
            "get_current_weather",
            "xiaomi_miot_control_device",
            "xiaomi_miot_get_devices",
            "xiaomi_miot_get_device_spec",
            "xiaomi_miot_read_device_property",
            "xiaomi_miot_get_property",
            "xiaomi_miot_control",
        }

        selected = tools_module._select_tool_names_for_text(
            "把空调的温度调成27度",
            available_names,
        )

        self.assertNotIn("get_current_weather", selected)
        self.assertIn("xiaomi_miot_control_device", selected)

    def test_tool_selection_keeps_plain_temperature_weather_query(self) -> None:
        selected = tools_module._select_tool_names_for_text(
            "今天温度多少",
            {"get_current_weather"},
        )

        self.assertEqual(selected, {"get_current_weather"})

    def test_tool_runner_registers_web_search_when_enabled(self) -> None:
        config = AssistantConfig(
            tools=ToolsConfig(
                enabled=True,
                allow_time=False,
                allow_weather=False,
                allow_volume=False,
                allow_music=False,
                allow_miot=False,
                allow_search=True,
            )
        )

        runner = create_tool_runner(config, FakeToolChat())

        self.assertIsNotNone(runner)
        names = [tool["function"]["name"] for tool in runner.tool_payloads]
        self.assertEqual(names, ["web_search"])

    def test_baidu_search_parses_html_results(self) -> None:
        html = """
        <html><body>
          <a href="https://example.com/player">01:01 00:00 / 01:01</a>
          <h3><a href="https://example.com/a">第一条结果</a></h3>
          <div>摘要内容 A</div>
          <a href="https://example.com/b">第二条结果</a>
        </body></html>
        """
        config = SearchConfig(provider="baidu", max_results=2, baidu_ai_enabled=False)

        with patch("voiceui.tools._get_text", return_value=html):
            result = search_web(config, {"query": "测试"})

        self.assertEqual(result["provider"], "baidu")
        self.assertEqual(result["results"][0]["title"], "第一条结果")
        self.assertEqual(result["results"][0]["url"], "https://example.com/a")
        self.assertNotIn("direct_response", result)

    def test_baidu_search_prefers_escaped_json_results(self) -> None:
        html = r"""
        {\"abstract\":\"\",\"title\":\"北京到上海多远?距离1200公里左右\",
         \"source\":{\"name\":\"乡村慢生活\"},\"linkInfo\":{\"href\":\"https://example.com/distance\"}}
        <h3><a href="https://example.com/noisy">低质量链接</a></h3>
        """
        config = SearchConfig(provider="baidu", max_results=1, baidu_ai_enabled=False)

        with patch("voiceui.tools._get_text", return_value=html):
            result = search_web(config, {"query": "从北京到上海要多远"})

        self.assertEqual(result["provider"], "baidu")
        self.assertEqual(result["results"][0]["title"], "北京到上海多远?距离1200公里左右")
        self.assertEqual(result["results"][0]["content"], "乡村慢生活")
        self.assertNotIn("direct_response", result)

    def test_baidu_ai_search_posts_structured_payload(self) -> None:
        config = SearchConfig(
            provider="baidu",
            max_results=2,
            baidu_ai_api_key_env="BAIDU_TEST_APPBUILDER_KEY",
        )
        response = {
            "request_id": "req-1",
            "is_safe": True,
            "usage": {"total_tokens": 123},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Beijing to Shanghai is about 1200 km. [ref_1]",
                    }
                }
            ],
            "references": [
                {
                    "id": 1,
                    "title": "Beijing to Shanghai distance",
                    "url": "https://example.com/distance",
                    "content": "About 1200 km",
                    "date": "2026-06-04",
                    "web_anchor": "Example",
                    "type": "web",
                }
            ],
        }

        with patch.dict(os.environ, {"BAIDU_TEST_APPBUILDER_KEY": "bce-test"}):
            with patch("voiceui.tools.post_json", return_value=response) as post:
                result = search_web(config, {"query": "distance from Beijing to Shanghai"})

        post.assert_called_once()
        payload = post.call_args.args[1]
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(payload["messages"][0]["content"], "distance from Beijing to Shanghai")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "deepseek-v3")
        self.assertEqual(payload["search_mode"], "required")
        self.assertEqual(payload["resource_type_filter"], [{"type": "web", "top_k": 2}])
        self.assertEqual(headers["Authorization"], "Bearer bce-test")
        self.assertEqual(headers["X-Appbuilder-Authorization"], "Bearer bce-test")
        self.assertEqual(result["provider"], "baidu_ai")
        self.assertIn("1200", result["answer"])
        self.assertEqual(result["results"][0]["provider"], "baidu_ai")
        self.assertEqual(result["results"][0]["url"], "https://example.com/distance")
        self.assertEqual(result["usage"]["total_tokens"], 123)
        self.assertNotIn("direct_response", result)

    def test_baidu_search_falls_back_to_html_without_ai_key(self) -> None:
        config = SearchConfig(
            provider="baidu",
            max_results=1,
            baidu_ai_api_key_env="VOICEUI_TEST_MISSING_BAIDU_KEY",
        )
        html_result = {
            "query": "test",
            "provider": "baidu",
            "results": [{"title": "Result", "url": "https://example.com"}],
        }

        with patch.dict(os.environ, {"VOICEUI_TEST_MISSING_BAIDU_KEY": ""}):
            with patch("voiceui.tools._search_baidu_ai") as ai_search:
                with patch("voiceui.tools._search_baidu_html", return_value=html_result) as html:
                    result = search_web(config, {"query": "test"})

        ai_search.assert_not_called()
        html.assert_called_once()
        self.assertEqual(result["provider"], "baidu")

    def test_baidu_search_falls_back_to_tavily_when_ai_fails(self) -> None:
        config = SearchConfig(
            provider="baidu",
            max_results=1,
            baidu_ai_api_key_env="VOICEUI_TEST_QIANFAN_KEY",
            tavily_api_key_env="VOICEUI_TEST_TAVILY_KEY",
        )
        tavily_result = {
            "query": "test",
            "provider": "tavily",
            "answer": "fallback answer",
            "results": [{"title": "Result", "url": "https://example.com"}],
        }

        with patch.dict(
            os.environ,
            {
                "VOICEUI_TEST_QIANFAN_KEY": "qianfan-test",
                "VOICEUI_TEST_TAVILY_KEY": "tvly-test",
            },
        ):
            with patch(
                "voiceui.tools._search_baidu_ai",
                side_effect=RuntimeError("account_overdue"),
            ) as ai_search:
                with patch("voiceui.tools._search_tavily", return_value=tavily_result) as tavily:
                    with patch("voiceui.tools._search_baidu_html") as html:
                        result = search_web(config, {"query": "test"})

        ai_search.assert_called_once()
        tavily.assert_called_once()
        html.assert_not_called()
        self.assertEqual(result["provider"], "tavily")
        self.assertEqual(result["answer"], "fallback answer")

    def test_auto_search_routes_chinese_query_to_baidu(self) -> None:
        baidu_result = {
            "query": "OpenAI 新闻",
            "provider": "baidu",
            "results": [{"title": "新闻", "url": "https://example.com", "provider": "baidu"}],
        }

        with patch("voiceui.tools._search_baidu", return_value=baidu_result) as baidu:
            with patch("voiceui.tools._search_tavily") as tavily:
                result = search_web(SearchConfig(provider="auto"), {"query": "OpenAI 新闻"})

        baidu.assert_called_once()
        tavily.assert_not_called()
        self.assertEqual(result["provider"], "baidu")

    def test_auto_search_routes_non_chinese_query_to_tavily(self) -> None:
        tavily_result = {
            "query": "OpenAI news",
            "provider": "tavily",
            "results": [{"title": "News", "url": "https://example.com", "provider": "tavily"}],
        }

        with patch("voiceui.tools._search_baidu") as baidu:
            with patch("voiceui.tools._search_tavily", return_value=tavily_result) as tavily:
                result = search_web(SearchConfig(provider="auto"), {"query": "OpenAI news"})

        baidu.assert_not_called()
        tavily.assert_called_once()
        self.assertEqual(result["provider"], "tavily")

    def test_tavily_search_posts_structured_payload(self) -> None:
        config = SearchConfig(provider="tavily", max_results=3)
        response = {
            "answer": "简短答案",
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com",
                    "content": "Snippet",
                    "score": 0.8,
                }
            ],
        }

        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            with patch("voiceui.tools.post_json", return_value=response) as post:
                result = search_web(config, {"query": "OpenAI", "time_range": "day"})

        post.assert_called_once()
        payload = post.call_args.args[1]
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(payload["query"], "OpenAI")
        self.assertEqual(payload["max_results"], 3)
        self.assertEqual(payload["time_range"], "day")
        self.assertEqual(headers["Authorization"], "Bearer tvly-test")
        self.assertEqual(result["provider"], "tavily")
        self.assertEqual(result["answer"], "简短答案")
        self.assertNotIn("direct_response", result)

    def test_tavily_search_retries_without_time_range_when_empty(self) -> None:
        config = SearchConfig(provider="tavily", max_results=2)
        empty_response = {"answer": "", "results": []}
        retry_response = {
            "answer": "Retry answer",
            "results": [
                {
                    "title": "Retry Result",
                    "url": "https://example.com/retry",
                    "content": "Retry snippet",
                }
            ],
        }

        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}):
            with patch(
                "voiceui.tools.post_json",
                side_effect=[empty_response, retry_response],
            ) as post:
                result = search_web(config, {"query": "OpenAI", "time_range": "week"})

        self.assertEqual(post.call_count, 2)
        first_payload = post.call_args_list[0].args[1]
        second_payload = post.call_args_list[1].args[1]
        self.assertEqual(first_payload["time_range"], "week")
        self.assertNotIn("time_range", second_payload)
        self.assertEqual(result["provider"], "tavily")
        self.assertEqual(result["answer"], "Retry answer")
        self.assertTrue(result["retried_without_time_range"])

    def test_set_system_volume_tool_accepts_aliases(self) -> None:
        tool = create_set_system_volume_tool(device=20)

        with patch("voiceui.tools.set_system_output_volume") as set_volume:
            set_volume.return_value = {"after_percent": 30}
            result = tool.handler({"level": 30, "muted": False})

        set_volume.assert_called_once_with(
            device=20,
            volume_percent=30.0,
            relative_percent=None,
            muted=False,
        )
        self.assertEqual(result["after_percent"], 30)

    def test_music_limiter_reduces_peak_to_threshold(self) -> None:
        import numpy as np  # type: ignore[import-untyped]

        data = np.array([[0.5, -1.2], [0.25, 0.8]], dtype="float32")

        limited, peak, gain = _limit_float_audio(data, threshold=0.92)

        self.assertAlmostEqual(peak, 1.2, places=5)
        self.assertAlmostEqual(gain, 0.92 / 1.2, places=5)
        self.assertLessEqual(float(np.max(np.abs(limited))), 0.92001)

    def test_dynamic_volume_chunks_apply_current_factor(self) -> None:
        factors = iter([1.0, 0.2])
        pcm = (
            (10000).to_bytes(2, "little", signed=True) * 320
            + (10000).to_bytes(2, "little", signed=True) * 320
        )

        chunks = list(
            _iter_dynamic_volume_pcm_chunks(
                pcm,
                sample_rate=16000,
                channels=1,
                dynamic_volume_getter=lambda: next(factors),
            )
        )

        self.assertEqual(
            chunks[0][:2],
            (10000).to_bytes(2, "little", signed=True),
        )
        self.assertEqual(
            chunks[1][:2],
            (2000).to_bytes(2, "little", signed=True),
        )


if __name__ == "__main__":
    unittest.main()
