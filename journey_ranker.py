import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_runtime import build_runnable_config, get_chat_model, parse_json_output, str_output_parser


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

    chain = prompt | model | str_output_parser | RunnableLambda(parse_json_output)

    config = build_runnable_config(
        run_name="pitch_step_1_rank_journeys",
        tags=["pitch", "step_1", "langchain"],
        metadata={
            "feature": "pitch",
            "step": "step_1",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    return chain.invoke(
        {
            "cliente_info": cliente_info,
            "prompt_assessor": prompt_assessor,
            "jornadas_texto": jornadas_texto,
        },
        config=config,
    )
