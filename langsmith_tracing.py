import os
import uuid
from datetime import datetime, timezone
from typing import Any

import requests


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LangSmithTracer:
    def __init__(self, api_key: str, enabled: bool, base_url: str | None = None):
        self.api_key = (api_key or os.getenv("LANGSMITH_API_KEY") or "").strip()
        self.enabled = bool(enabled and self.api_key)
        self.base_url = (base_url or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
        self.project_name = (os.getenv("LANGSMITH_PROJECT") or "contato-assessor").strip()

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def start_run(
        self,
        *,
        name: str,
        run_type: str,
        inputs: dict[str, Any],
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        project_name: str | None = None,
    ) -> str | None:
        if not self.enabled:
            return None

        run_id = str(uuid.uuid4())
        payload = {
            "id": run_id,
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "start_time": _iso_now(),
            "session_name": project_name or self.project_name,
            "tags": tags or [],
            "extra": {"metadata": metadata or {}},
        }

        try:
            requests.post(
                f"{self.base_url}/runs",
                json=payload,
                headers=self._headers(),
                timeout=8,
            ).raise_for_status()
            return run_id
        except Exception:
            return None

    def log_event(self, run_id: str | None, event_name: str, details: dict[str, Any] | None = None) -> None:
        if not self.enabled or not run_id:
            return

        payload = {
            "id": str(uuid.uuid4()),
            "name": event_name,
            "run_type": "tool",
            "parent_run_id": run_id,
            "inputs": details or {},
            "start_time": _iso_now(),
            "end_time": _iso_now(),
            "outputs": {"status": "logged"},
        }

        try:
            requests.post(
                f"{self.base_url}/runs",
                json=payload,
                headers=self._headers(),
                timeout=8,
            ).raise_for_status()
        except Exception:
            return

    def end_run(
        self,
        run_id: str | None,
        *,
        status: str,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if not self.enabled or not run_id:
            return

        payload = {
            "end_time": _iso_now(),
            "outputs": outputs or {"status": status},
        }
        if error:
            payload["error"] = error

        try:
            requests.patch(
                f"{self.base_url}/runs/{run_id}",
                json=payload,
                headers=self._headers(),
                timeout=8,
            ).raise_for_status()
        except Exception:
            return
