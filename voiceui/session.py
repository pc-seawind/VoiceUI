from __future__ import annotations

from voiceui.llm import ChatMessage
from voiceui.models import ConversationConfig, LlmConfig


class ConversationSession:
    def __init__(self, llm_config: LlmConfig, conversation_config: ConversationConfig):
        self.llm_config = llm_config
        self.conversation_config = conversation_config
        self.messages: list[ChatMessage] = []
        self.reset()

    def reset(self) -> None:
        self.messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.llm_config.system_prompt)
        ]

    def add_user(self, text: str) -> None:
        self.messages.append(ChatMessage(role="user", content=text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=text))
        self._trim()

    def _trim(self) -> None:
        max_messages = max(2, self.conversation_config.max_turns * 2 + 1)
        if len(self.messages) <= max_messages:
            return
        system = self.messages[0]
        self.messages = [system, *self.messages[-(max_messages - 1) :]]
