from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from voiceui.env import load_dotenv


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    error_prefix: str = "HTTP request failed",
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = _read_error_body(exc)
        raise RuntimeError(f"{error_prefix}: {url}: HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{error_prefix}: {url}: {exc}") from exc


def require_api_key(env_name: str | None) -> str | None:
    if not env_name:
        return None

    import os

    load_dotenv()
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(
            f"Missing API key environment variable: {env_name}. "
            f"Set it in .env or PowerShell with: $env:{env_name}=\"your-token\""
        )
    return value


def optional_api_key(env_name: str | None) -> str | None:
    if not env_name:
        return None

    load_dotenv()
    return os.environ.get(env_name) or None


def _read_error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    return body or error.reason or "no response body"
