from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voiceui.env import load_dotenv


class EnvTests(unittest.TestCase):
    def test_load_dotenv_reads_values_without_overriding_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# local secrets",
                        "MIFY_API_KEY=from-file",
                        "EXISTING=from-file",
                        "export QUOTED='quoted value'",
                        "INLINE=value # comment",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"EXISTING": "from-env"}, clear=True):
                loaded = load_dotenv(path)

                self.assertEqual(loaded, path)
                self.assertEqual(__import__("os").environ["MIFY_API_KEY"], "from-file")
                self.assertEqual(__import__("os").environ["EXISTING"], "from-env")
                self.assertEqual(__import__("os").environ["QUOTED"], "quoted value")
                self.assertEqual(__import__("os").environ["INLINE"], "value")

    def test_env_example_lists_aliyun_nls_keys(self) -> None:
        example = Path(".env.example").read_text(encoding="utf-8")

        self.assertIn("ALIYUN_NLS_APPKEY=", example)
        self.assertIn("ALIYUN_AccessKeyId=", example)
        self.assertIn("ALIYUN_AccessKeySecret=", example)


if __name__ == "__main__":
    unittest.main()
