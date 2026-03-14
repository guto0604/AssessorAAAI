import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from core.langchain_runtime import build_runnable_config, get_chat_model, parse_json_output, str_output_parser


def _build_api_metrics(response, *, provider: str = "openai") -> dict:
    usage = getattr(response, "usage", {}) or {}
    return {
        "provider": provider,
        "model": getattr(response, "model", None),
        "latency_ms": getattr(response, "elapsed_ms", None),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "response_id": getattr(response, "response_id", None),
    }


def rank_journeys(cliente_info, prompt_assessor, jornadas_df, trace_context: dict | None = None, include_api_metrics: bool = False):
    jornadas_texto = ""
    for _, row in jornadas_df.iterrows():
        jornadas_texto += f"""
ID: {row['Jornada_ID']}
Nome: {row['Nome_Jornada']}
Categoria: {row['Categoria']}
Objetivo: {row['Objetivo_Principal']}
Descrição: {row['Descricao_Resumida']}
---
"""

    system_prompt = """
Você é um modelo responsável por ranquear jornadas comerciais para um assessor de investimentos.

Sua tarefa:
- Analisar as informações do cliente
- Analisar o prompt escrito pelo assessor
- Avaliar a lista de jornadas disponíveis
- Retornar as 5 jornadas mais adequadas

IMPORTANTE:
- Retorne APENAS um JSON válido
- Não escreva texto antes ou depois
- Não use markdown
- Não explique nada fora do JSON
- O JSON deve seguir exatamente o formato abaixo

Formato obrigatório:

{
  "ranking": [
    {
      "jornada_id": "string",
      "nome_jornada": "string",
      "score": float entre 0 e 1
    }
  ]
}
"""

    user_prompt = """
Cliente:
{cliente_info}

Intenção do assessor:
{prompt_assessor}

Jornadas disponíveis:
{jornadas_texto}

Rankeie as 5 jornadas mais adequadas.
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )
    model = get_chat_model(model="gpt-5-mini", temperature=1, response_format={"type": "json_object"})

    config = build_runnable_config(
        run_name="pitch_step_1_rank_journeys",
        tags=["pitch", "step_1", "langchain"],
        metadata={
            "feature": "pitch",
            "step": "step_1",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    messages = prompt.invoke(
        {
            "cliente_info": cliente_info,
            "prompt_assessor": prompt_assessor,
            "jornadas_texto": jornadas_texto,
        },
        config=config,
    )
    response = model.invoke(messages, config=config)
    parsed = parse_json_output(str_output_parser.invoke(response, config=config))

    if include_api_metrics:
        return {
            "result": parsed,
            "api_metrics": _build_api_metrics(response),
        }

    return parsed
