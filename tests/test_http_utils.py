from __future__ import annotations

import io
import unittest
import urllib.error
from unittest.mock import patch

from voiceui.http_utils import post_json, require_api_key


class HttpUtilsTests(unittest.TestCase):
    def test_require_api_key_reports_missing_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("voiceui.http_utils.load_dotenv", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Missing API key environment variable"):
                    require_api_key("MIFY_API_KEY")

    def test_post_json_includes_http_error_body(self) -> None:
        error = urllib.error.HTTPError(
            url="https://example.test/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"bad model"}'),
        )

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "bad model"):
                post_json(
                    "https://example.test/v1/chat/completions",
                    {"model": "bad"},
                    error_prefix="LLM request failed",
                )


if __name__ == "__main__":
    unittest.main()
