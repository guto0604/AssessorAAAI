import os
import uuid
from datetime import datetime, timezone
from typing import Any

from langsmith import Client


def _now_utc() -> datetime:
    """Retorna o timestamp atual em formato padronizado para registros e rastreabilidade.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    """
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    """Retorna o timestamp atual em formato padronizado para registros e rastreabilidade.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    """
    return _now_utc().isoformat()


class LangSmithTracer:
    def __init__(self, api_key: str, enabled: bool, base_url: str | None = None):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            api_key: Valor de entrada necessário para processar 'api_key'.
            enabled: Valor de entrada necessário para processar 'enabled'.
            base_url: Valor de entrada necessário para processar 'base_url'.
        """
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
        """Responsável por processar run no contexto da aplicação de assessoria.

        Args:
            name: Valor de entrada necessário para processar 'name'.
            run_type: Valor de entrada necessário para processar 'run_type'.
            inputs: Valor de entrada necessário para processar 'inputs'.
            tags: Valor de entrada necessário para processar 'tags'.
            metadata: Valor de entrada necessário para processar 'metadata'.
            project_name: Valor de entrada necessário para processar 'project_name'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
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
        """Responsável por processar event no contexto da aplicação de assessoria.

        Args:
            run_id: Identificador usado para referenciar 'run_id'.
            event_name: Valor de entrada necessário para processar 'event_name'.
            details: Valor de entrada necessário para processar 'details'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
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
        """Responsável por processar child run no contexto da aplicação de assessoria.

        Args:
            parent_run_id: Identificador usado para referenciar 'parent_run_id'.
            name: Valor de entrada necessário para processar 'name'.
            run_type: Valor de entrada necessário para processar 'run_type'.
            inputs: Valor de entrada necessário para processar 'inputs'.
            outputs: Valor de entrada necessário para processar 'outputs'.
            metadata: Valor de entrada necessário para processar 'metadata'.
            error: Valor de entrada necessário para processar 'error'.
            tags: Valor de entrada necessário para processar 'tags'.
            start_time: Valor de entrada necessário para processar 'start_time'.
            end_time: Valor de entrada necessário para processar 'end_time'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
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
        """Responsável por processar run no contexto da aplicação de assessoria.

        Args:
            run_id: Identificador usado para referenciar 'run_id'.
            status: Valor de entrada necessário para processar 'status'.
            outputs: Valor de entrada necessário para processar 'outputs'.
            error: Valor de entrada necessário para processar 'error'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
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
                #trace_id=run_state["id"],
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
        except Exception as e:
            self.last_error = f"Falha ao enviar tracing para o LangSmith. \n\n{e}"
            return False

    def submit_feedback(
        self,
        *,
        run_id: str | None,
        score: bool,
        screen_key: str,
        screen_label: str,
        feedback_id: str | None = None,
    ) -> str | None:
        """Cria ou atualiza feedback associado a um run no LangSmith."""
        if not self.enabled or not run_id or not self._client:
            self.last_error = "Tracing LangSmith não está ativo para registrar feedback."
            return None

        score_value = 1 if score else 0
        feedback_value = {
            "sentiment": "like" if score else "dislike",
            "screen_key": screen_key,
            "screen_label": screen_label,
            "recorded_at": _iso_now(),
        }
        comment = f"Feedback da tela: {'like' if score else 'dislike'}"

        try:
            if feedback_id:
                self._client.update_feedback(
                    feedback_id,
                    score=score_value,
                    value=feedback_value,
                    comment=comment,
                )
                self.last_error = None
                return str(feedback_id)

            feedback = self._client.create_feedback(
                run_id=run_id,
                trace_id=run_id,
                key="screen_feedback",
                score=score_value,
                value=feedback_value,
                comment=comment,
                source_info={
                    "source": "streamlit_ui",
                    "component": "screen_feedback",
                    "screen_key": screen_key,
                    "screen_label": screen_label,
                },
            )
            self.last_error = None
            return str(getattr(feedback, "id", "") or "")
        except Exception as e:
            self.last_error = f"Falha ao registrar feedback no LangSmith. \n\n{e}"
            return None
