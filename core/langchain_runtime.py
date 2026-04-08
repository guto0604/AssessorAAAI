import json
import os
from datetime import date, datetime
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from core.openai_client import get_effective_openai_api_key


DEFAULT_PROJECT = "poc_datamasters"


def configure_langsmith() -> None:
    """Configura variáveis para tracing LangSmith de forma idempotente."""
    if not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = DEFAULT_PROJECT

    if os.getenv("LANGSMITH_TRACING") is None and os.getenv("LANGCHAIN_TRACING_V2") is None:
        os.environ["LANGSMITH_TRACING"] = "true"

    api_key = get_effective_openai_api_key()  # no-op just to keep side-effect parity with session usage
    _ = api_key


def get_chat_model(model: str, temperature: float = 1, response_format: dict[str, Any] | None = None) -> ChatOpenAI:
    """Obtém a informação necessária para a etapa atual do processo.

    Args:
        model: Modelo utilizado para executar a etapa 'model'.
        temperature: Valor de entrada necessário para processar 'temperature'.
        response_format: Valor de entrada necessário para processar 'response_format'.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    api_key = get_effective_openai_api_key()
    kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
    if api_key:
        kwargs["api_key"] = api_key
    if response_format is not None:
        kwargs["model_kwargs"] = {"response_format": response_format}
    return ChatOpenAI(**kwargs)


def build_runnable_config(
    *,
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunnableConfig:
    """Monta a estrutura de dados usada nas próximas etapas do fluxo.

    Args:
        run_name: Valor de entrada necessário para processar 'run_name'.
        tags: Valor de entrada necessário para processar 'tags'.
        metadata: Valor de entrada necessário para processar 'metadata'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    """
    configure_langsmith()
    return RunnableConfig(
        run_name=run_name,
        tags=tags or [],
        metadata=metadata or {},
    )


def parse_json_output(text: str) -> Any:
    """Interpreta e normaliza a entrada para manter consistência no processamento.

    Args:
        text: Texto de entrada a ser processado pela função.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    """
    return json.loads(text)




def json_default_serializer(value: Any) -> str:
    """Serializa tipos não padrão para JSON no contexto dos fluxos LangChain."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except TypeError:
            pass
    return str(value)


def json_dumps_safe(payload: Any, *, ensure_ascii: bool = False) -> str:
    """Aplica serialização JSON resiliente para objetos com datas."""
    return json.dumps(payload, ensure_ascii=ensure_ascii, default=json_default_serializer)

str_output_parser = StrOutputParser()
