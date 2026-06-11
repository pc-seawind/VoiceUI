from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from voiceui.models import AssistantConfig

T = TypeVar("T")

AUTO_CONFIG = "auto"
WINDOWS_AUTO_CONFIG_PATH = "config.demo.wake.aliyun.yaml"
LINUX_AUTO_CONFIG_PATH = "config.demo.linux.wake.aliyun.yaml"


def load_config(path: str | Path | None = None) -> AssistantConfig:
    config = AssistantConfig()
    resolved_path = resolve_config_path(path)
    if resolved_path is None:
        return config

    raw = _read_mapping(resolved_path)
    merged = _deep_merge(asdict(config), raw)
    return _from_mapping(AssistantConfig, merged)



def resolve_config_path(
    path: str | Path | None,
    *,
    platform: str | None = None,
) -> Path | None:
    if path is None:
        return None
    requested = str(path).strip()
    if requested.casefold() == AUTO_CONFIG:
        return Path(default_config_path_for_platform(platform=platform))
    return Path(path)


def default_config_path_for_platform(*, platform: str | None = None) -> str:
    platform_name = sys.platform if platform is None else platform
    normalized = platform_name.casefold()
    if normalized.startswith(("win32", "cygwin", "msys")):
        return WINDOWS_AUTO_CONFIG_PATH
    if normalized.startswith("linux"):
        return LINUX_AUTO_CONFIG_PATH
    return WINDOWS_AUTO_CONFIG_PATH


def config_to_dict(config: AssistantConfig) -> dict[str, Any]:
    return asdict(config)


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with: pip install -e \".[config]\""
            ) from exc
        data = yaml.safe_load(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _from_mapping(cls: type[T], mapping: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        return mapping  # type: ignore[return-value]

    values: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in mapping:
            continue
        value = mapping[item.name]
        target_type = type_hints.get(item.name, item.type)
        values[item.name] = _from_value(target_type, value)
    return cls(**values)  # type: ignore[misc]


def _from_value(annotation: Any, value: Any) -> Any:
    target_type = _resolve_type(annotation)
    if is_dataclass(target_type) and isinstance(value, dict):
        return _from_mapping(target_type, value)

    origin = get_origin(target_type)
    if origin in {list, tuple} and isinstance(value, list):
        args = get_args(target_type)
        if not args:
            return value
        item_type = args[0]
        return [_from_value(item_type, item) for item in value]

    return value


def _resolve_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin not in {Union, types.UnionType}:
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return args[0] if len(args) == 1 else annotation
