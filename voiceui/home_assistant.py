from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from voiceui.models import HomeAssistantConfig


class HomeAssistantClient:
    def __init__(self, config: HomeAssistantConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def call_service(self, domain: str, service: str, data: dict) -> dict:
        token = os.environ.get(self.config.token_env, "")
        if not token:
            raise RuntimeError(f"Missing Home Assistant token env: {self.config.token_env}")

        url = f"{self.config.url.rstrip('/')}/api/services/{domain}/{service}"
        body = json.dumps(data).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Home Assistant request failed: {url}: {exc}") from exc
