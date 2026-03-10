import json
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LangSmithTracer:
    def __init__(self, api_key: str, enabled: bool, base_url: str | None = None):
        self.api_key = (os.getenv("LANGSMITH_API_KEY") or api_key or "").strip()
        self.enabled = bool(enabled and self.api_key)
        self.base_url = (base_url or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
        self.project_name = (os.getenv("LANGSMITH_PROJECT") or "poc_datamasters").strip()
        self._trace_id_by_run_id: dict[str, str] = {}
        self.last_error: str | None = None

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _send_json(self, method: str, url: str, payload: dict[str, Any]) -> None:
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method=method,
        )
        with urllib.request.urlopen(req, timeout=8):
            return

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

        self.last_error = None

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
        try:
            self._send_json("POST", f"{self.base_url}/runs", payload)
            self._trace_id_by_run_id[run_id] = run_id
            return run_id
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def log_event(self, run_id: str | None, event_name: str, details: dict[str, Any] | None = None) -> None:
        if not self.enabled or not run_id:
            return

        self.last_error = None

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
        try:
            self._send_json("POST", f"{self.base_url}/runs", payload)
        except Exception as exc:
            self.last_error = str(exc)
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

        self.last_error = None

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
        if error:
            payload["error"] = error

        try:
            self._send_json("POST", f"{self.base_url}/runs", payload)
            self._trace_id_by_run_id[child_id] = payload["trace_id"]
            return child_id
        except Exception as exc:
            self.last_error = str(exc)
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

        self.last_error = None

        payload = {
            "end_time": _iso_now(),
            "outputs": outputs or {"status": status},
            "status": status,
        }
        if error:
            payload["error"] = error

        try:
            self._send_json("PATCH", f"{self.base_url}/runs/{run_id}", payload)
            self._trace_id_by_run_id.pop(run_id, None)
        except Exception as exc:
            self.last_error = str(exc)
            return
