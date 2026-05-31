from __future__ import annotations

import unittest

from voiceui.models import ConversationConfig, LlmConfig
from voiceui.session import ConversationSession


class SessionTests(unittest.TestCase):
    def test_session_keeps_system_prompt_while_trimming_history(self) -> None:
        session = ConversationSession(LlmConfig(), ConversationConfig(max_turns=1))

        session.add_user("one")
        session.add_assistant("two")
        session.add_user("three")

        self.assertEqual(session.messages[0].role, "system")
        self.assertLessEqual(len(session.messages), 3)
        self.assertEqual(session.messages[-1].content, "three")

    def test_reset_clears_history_but_keeps_system_prompt(self) -> None:
        session = ConversationSession(
            LlmConfig(system_prompt="system prompt"),
            ConversationConfig(max_turns=3),
        )

        session.add_user("one")
        session.add_assistant("two")
        session.reset()

        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.messages[0].role, "system")
        self.assertEqual(session.messages[0].content, "system prompt")


if __name__ == "__main__":
    unittest.main()
