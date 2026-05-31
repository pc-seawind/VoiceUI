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


if __name__ == "__main__":
    unittest.main()
