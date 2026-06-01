from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> Path | None:
    env_path = Path(path) if path is not None else _find_dotenv()
    if env_path is None or not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        key_value = _parse_env_line(line)
        if key_value is None:
            continue
        key, value = key_value
        os.environ.setdefault(key, value)
    return env_path


def _find_dotenv() -> Path | None:
    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    else:
        value = _strip_inline_comment(value).strip()
    return key, value


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index]
    return value
