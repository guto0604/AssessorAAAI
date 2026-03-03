import json
import os
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from openai_client import get_effective_openai_api_key


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
    configure_langsmith()
    return RunnableConfig(
        run_name=run_name,
        tags=tags or [],
        metadata=metadata or {},
    )


def parse_json_output(text: str) -> Any:
    return json.loads(text)


str_output_parser = StrOutputParser()
