import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from core.langchain_runtime import build_runnable_config, get_chat_model, parse_json_output, str_output_parser


def _build_api_metrics(response, *, provider: str = "openai", prompt: dict | None = None, output: str | None = None) -> dict:
    """Responsável por montar api metrics no contexto da aplicação de assessoria.

    Args:
        response: Valor de entrada necessário para processar 'response'.
        provider: Identificador usado para referenciar 'provider'.
        prompt: Valor de entrada necessário para processar 'prompt'.
        output: Valor de entrada necessário para processar 'output'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    usage = getattr(response, "usage", {}) or {}
    return {
        "provider": provider,
        "model": getattr(response, "model", None),
        "latency_ms": getattr(response, "elapsed_ms", None),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "response_id": getattr(response, "response_id", None),
        "prompt": prompt or {},
        "output": output,
    }


def rank_journeys(cliente_info, prompt_assessor, jornadas_df, trace_context: dict | None = None, include_api_metrics: bool = False):
    """Responsável por priorizar journeys no contexto da aplicação de assessoria.

    Args:
        cliente_info: Dicionário com os dados consolidados do cliente para personalizar a resposta.
        prompt_assessor: Valor de entrada necessário para processar 'prompt_assessor'.
        jornadas_df: Valor de entrada necessário para processar 'jornadas_df'.
        trace_context: Contexto de rastreio da execução para observabilidade.
        include_api_metrics: Indica se a função deve retornar métricas de uso de API junto ao resultado.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
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
            "api_metrics": _build_api_metrics(
                response,
                prompt={"messages": str(messages)},
                output=str_output_parser.invoke(response, config=config),
            ),
        }

    return parsed
