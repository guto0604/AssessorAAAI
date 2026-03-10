import os
import uuid
from datetime import datetime, timezone
from typing import Any

from langsmith import Client


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now_utc().isoformat()


class LangSmithTracer:
    def __init__(self, api_key: str, enabled: bool, base_url: str | None = None):
        self.api_key = (os.getenv("LANGSMITH_API_KEY") or api_key or "").strip()
        self.enabled = bool(enabled and self.api_key)
        self.base_url = (base_url or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
        self.project_name = (os.getenv("LANGSMITH_PROJECT") or "poc_datamasters").strip()
        self._client = Client(api_key=self.api_key, api_url=self.base_url) if self.enabled else None
        self._runs: dict[str, dict[str, Any]] = {}
        self.last_error: str | None = None

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
        self._runs[run_id] = {
            "id": run_id,
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "start_time": _now_utc(),
            "project_name": project_name or self.project_name,
            "tags": tags or [],
            "metadata": metadata or {},
            "events": [],
        }
        return run_id

    def log_event(self, run_id: str | None, event_name: str, details: dict[str, Any] | None = None) -> None:
        if not self.enabled or not run_id:
            return
        run_state = self._runs.get(run_id)
        if not run_state:
            return

        run_state["events"].append(
            {
                "name": event_name,
                "details": details or {},
                "time": _iso_now(),
            }
        )

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

        self.log_event(
            parent_run_id,
            event_name=f"child_run:{name}",
            details={
                "run_type": run_type,
                "inputs": inputs or {},
                "outputs": outputs or {},
                "metadata": metadata or {},
                "error": error,
                "tags": tags or [],
                "start_time": start_time,
                "end_time": end_time,
            },
        )
        return None

    def end_run(
        self,
        run_id: str | None,
        *,
        status: str,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        if not self.enabled or not run_id:
            return False

        run_state = self._runs.pop(run_id, None)
        if not run_state or not self._client:
            return False

        final_outputs = outputs or {"status": status}
        if run_state["events"]:
            final_outputs = {
                **final_outputs,
                "events": run_state["events"],
            }

        extra = {"metadata": run_state["metadata"]}
        if error:
            extra["metadata"]["error"] = error

        try:
            self._client.create_run(
                id=run_state["id"],
                trace_id=run_state["id"],
                name=run_state["name"],
                run_type=run_state["run_type"],
                project_name=run_state["project_name"],
                inputs=run_state["inputs"],
                outputs=final_outputs,
                start_time=run_state["start_time"],
                end_time=_now_utc(),
                error=error,
                tags=run_state["tags"],
                extra=extra,
            )
            self.last_error = None
            return True
        except Exception:
            self.last_error = "Falha ao enviar tracing para o LangSmith."
            return False
