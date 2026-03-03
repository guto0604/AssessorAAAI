import json
from datetime import datetime, timezone
from time import perf_counter
from openai_client import get_openai_client


def _usage_dict(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)

    if input_tokens is None:
        input_tokens = prompt_tokens
    if output_tokens is None:
        output_tokens = completion_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def rank_journeys(cliente_info, prompt_assessor, jornadas_df, trace_context: dict | None = None):
    
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

    user_prompt = f"""
Cliente:
{cliente_info}

Intenção do assessor:
{prompt_assessor}

Jornadas disponíveis:
{jornadas_texto}

Rankeie as 5 jornadas mais adequadas.
"""

    call_start_iso = _iso_now()
    call_start_perf = perf_counter()
    response = get_openai_client().chat.completions.create(
        model="gpt-5-mini",  # modelo rápido para classificação
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1,
        response_format={"type": "json_object"}
    )
    call_end_iso = _iso_now()
    call_duration_s = perf_counter() - call_start_perf

    content = response.choices[0].message.content

    if trace_context:
        tracer = trace_context.get("tracer")
        parent_run_id = trace_context.get("parent_run_id")
        if tracer and parent_run_id:
            tracer.log_child_run(
                parent_run_id,
                name="pitch_step_1_rank_journeys_llm",
                run_type="llm",
                inputs={
                    "model": "gpt-5-mini",
                    "temperature": 1,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                outputs={
                    "raw_response": content,
                    "model_used": getattr(response, "model", "gpt-5-mini"),
                    "openai_latency_seconds": round(call_duration_s, 4),
                    "usage": _usage_dict(response),
                },
                metadata={"step": "step_1"},
                tags=["pitch", "llm", "step_1"],
                start_time=call_start_iso,
                end_time=call_end_iso,
            )

    return json.loads(content)
