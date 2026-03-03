import json
from pathlib import Path
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


def list_kb_files(kb_dir: str = "knowledge_base"):
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return []
    return sorted([str(p.as_posix()) for p in kb_path.rglob("*.txt")])


def select_sources_step4(
    cliente_info: dict,
    prompt_assessor: str,
    jornada_selecionada: dict,
    carteira_summary: dict,
    produtos_df,
    investimentos_cliente_df,
    kb_dir: str = "knowledge_base",
    model: str = "gpt-5-mini",
    trace_context: dict | None = None,
):
    kb_files = list_kb_files(kb_dir)

    produtos_catalogo = produtos_df[[
        "Produto_ID", "Nome_Produto", "Categoria", "Subcategoria", "Risco_Nivel (1-5)", "Suitability_Ideal"
    ]].to_dict(orient="records")

    investimentos_list = investimentos_cliente_df[[
        "Produto", "Categoria", "Valor_Investido"
    ]].to_dict(orient="records")

    system_prompt = """
Você é um agente de seleção de fontes para um copiloto de assessoria de investimentos.

IMPORTANTE: Fluxo em duas etapas dentro desta mesma resposta:
1) Selecionar PRODUTOS candidatos (por Produto_ID) com base em jornada, prompt, perfil e momento (rentabilidade vs CDI, liquidez, gaps), selecione pelo menos 3 produtos.
2) Com base nos PRODUTOS selecionados (categoria/subcategoria/risco), escolher quais documentos .txt da knowledge_base fazem sentido CONSULTAR NO PRÓXIMO PASSO.
   - Você NÃO deve ler conteúdo dos documentos; selecione APENAS PELO NOME do arquivo.
   - Selecione no máximo 5 arquivos da knowledge_base.

Regras:
- Retorne APENAS JSON válido (sem markdown).
- Selecione entre 3 e 5 produtos candidatos (Produto_ID).
- Suitability não precisa ser exato, mas evite escolhas obviamente incompatíveis (ex: cripto para conservador), a menos que o prompt/jornada justifique.
- Sempre inclua as fontes de dados:
  - "investimentos_do_cliente"
  - "base_de_produtos"
  - "metricas_rentabilidade_carteira_e_cdi"

Formato obrigatório:
{
  "data_sources": ["..."],
  "products_selected_ids": ["P1","P2", "..."],
  "kb_files_selected": ["..."], 
  "reasoning_short": "..."
}
"""

    user_payload = {
        "cliente_info": cliente_info,
        "prompt_assessor": prompt_assessor,
        "jornada_selecionada": jornada_selecionada,
        "carteira_summary": carteira_summary,
        "investimentos_atuais": investimentos_list,
        "produtos_catalogo": produtos_catalogo,
        "kb_files_available": kb_files
    }

    call_start_iso = _iso_now()
    call_start_perf = perf_counter()
    resp = get_openai_client().chat.completions.create(
        model=model,
        temperature=1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
    )
    call_end_iso = _iso_now()
    call_duration_s = perf_counter() - call_start_perf

    parsed = json.loads(resp.choices[0].message.content)

    if trace_context:
        tracer = trace_context.get("tracer")
        parent_run_id = trace_context.get("parent_run_id")
        if tracer and parent_run_id:
            tracer.log_child_run(
                parent_run_id,
                name="pitch_step_4_source_selector_llm",
                run_type="llm",
                inputs={
                    "model": model,
                    "temperature": 1,
                    "system_prompt": system_prompt,
                    "user_payload": user_payload,
                    "response_format": {"type": "json_object"},
                },
                outputs={
                    "response": parsed,
                    "model_used": getattr(resp, "model", model),
                    "openai_latency_seconds": round(call_duration_s, 4),
                    "usage": _usage_dict(resp),
                },
                metadata={"step": "step_4"},
                tags=["pitch", "llm", "step_4"],
                start_time=call_start_iso,
                end_time=call_end_iso,
            )

    # saneamento
    parsed["data_sources"] = parsed.get("data_sources", [])
    parsed["products_selected_ids"] = parsed.get("products_selected_ids", [])
    parsed["kb_files_selected"] = parsed.get("kb_files_selected", [])
    parsed["reasoning_short"] = parsed.get("reasoning_short", "")

    # enforce KB limit <= 5, just in case
    parsed["kb_files_selected"] = parsed["kb_files_selected"][:5]

    return parsed
