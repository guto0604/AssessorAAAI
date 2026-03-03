import os
import uuid
from datetime import datetime, timezone
from typing import Any

import requests


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LangSmithTracer:
    def __init__(self, api_key: str, enabled: bool, base_url: str | None = None):
        self.api_key = (os.getenv("LANGSMITH_API_KEY") or api_key or "").strip()
        self.enabled = bool(enabled and self.api_key)
        self.base_url = (base_url or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
        self.project_name = (os.getenv("LANGSMITH_PROJECT") or "poc_datamasters").strip()
        self._dotted_order_by_run_id: dict[str, str] = {}
        self._trace_id_by_run_id: dict[str, str] = {}

    def _build_dotted_order(self, run_id: str, parent_run_id: str | None = None) -> str:
        order_token = f"{_iso_now()}_{run_id}"
        if not parent_run_id:
            return order_token

        parent_dotted_order = self._dotted_order_by_run_id.get(parent_run_id)
        if parent_dotted_order:
            return f"{parent_dotted_order}.{order_token}"

        return order_token

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
            "trace_id": run_id,
            "inputs": inputs,
            "start_time": _iso_now(),
            "session_name": project_name or self.project_name,
            "tags": tags or [],
            "extra": {"metadata": metadata or {}},
        }
        payload["dotted_order"] = self._build_dotted_order(run_id)

        try:
            requests.post(
                f"{self.base_url}/runs",
                json=payload,
                headers=self._headers(),
                timeout=8,
            ).raise_for_status()
            self._dotted_order_by_run_id[run_id] = payload["dotted_order"]
            self._trace_id_by_run_id[run_id] = run_id
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
            "trace_id": self._trace_id_by_run_id.get(run_id, run_id),
            "session_name": self.project_name,
            "inputs": details or {},
            "start_time": _iso_now(),
            "end_time": _iso_now(),
            "outputs": {"status": "logged"},
        }
        payload["dotted_order"] = self._build_dotted_order(payload["id"], run_id)

        try:
            requests.post(
                f"{self.base_url}/runs",
                json=payload,
                headers=self._headers(),
                timeout=8,
            ).raise_for_status()
        except Exception:
            return

    def log_child_run(
        self,
        parent_run_id: str | None,
        *,
        name: str,
        run_type: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        tags: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> str | None:
        if not self.enabled or not parent_run_id:
            return None

        child_id = str(uuid.uuid4())
        payload = {
            "id": child_id,
            "name": name,
            "run_type": run_type,
            "parent_run_id": parent_run_id,
            "trace_id": self._trace_id_by_run_id.get(parent_run_id, parent_run_id),
            "session_name": self.project_name,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "start_time": start_time or _iso_now(),
            "end_time": end_time or _iso_now(),
            "tags": tags or [],
            "extra": {"metadata": metadata or {}},
        }
        payload["dotted_order"] = self._build_dotted_order(child_id, parent_run_id)
        if error:
            payload["error"] = error

        try:
            requests.post(
                f"{self.base_url}/runs",
                json=payload,
                headers=self._headers(),
                timeout=8,
            ).raise_for_status()
            self._dotted_order_by_run_id[child_id] = payload["dotted_order"]
            self._trace_id_by_run_id[child_id] = payload["trace_id"]
            return child_id
        except Exception:
            return None

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
            self._dotted_order_by_run_id.pop(run_id, None)
            self._trace_id_by_run_id.pop(run_id, None)
        except Exception:
            return
