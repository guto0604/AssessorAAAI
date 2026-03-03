import json

from openai_client import get_openai_client


def _usage_dict(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _read_kb_files(file_paths, max_chars_each=3500):
    """Lê conteúdo dos .txt selecionados (RAG simples)."""
    docs = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if len(txt) > max_chars_each:
                    txt = txt[:max_chars_each] + "\n...[TRUNCADO]"
                docs.append({"path": path, "content": txt})
        except Exception as e:
            docs.append({"path": path, "content": f"[ERRO AO LER: {e}]"})
    return docs


def build_pitch_options_step5(
    cliente_info: dict,
    prompt_assessor: str,
    jornada_selecionada: dict,
    carteira_summary: dict,
    investimentos_cliente_df,
    produtos_selecionados_df,
    kb_files_selected: list[str],
    model: str = "gpt-5.1",
    trace_context: dict | None = None,
):
    """
    Gera opções estruturadas por categoria para o assessor selecionar (Passo 6).
    Retorna JSON com listas priorizadas (ordem importa).
    """
    kb_docs = _read_kb_files(kb_files_selected)

    investimentos_list = investimentos_cliente_df[["Produto", "Categoria", "Valor_Investido"]].to_dict(orient="records")

    produtos_list = []
    if produtos_selecionados_df is not None and not produtos_selecionados_df.empty:
        cols = ["Produto_ID", "Nome_Produto", "Categoria", "Subcategoria", "Risco_Nivel (1-5)", "Suitability_Ideal"]
        produtos_list = produtos_selecionados_df[cols].to_dict(orient="records")

    system_prompt = """
Você é um estrategista de conteúdo comercial para um assessor de investimentos.

Tarefa:
Com base em:
- contexto do cliente e carteira,
- prompt do assessor,
- jornada selecionada (com descrição),
- produtos candidatos (já pré-selecionados),
- documentos internos e research (conteúdo RAG),

Gere opções estruturadas e PRIORITÁRIAS (ordem importa) para o pitch.

Regras:
- Responda APENAS JSON válido (sem markdown).
- NÃO invente números de rentabilidade de produtos que não estejam nos dados. Se não existir, fale de forma qualitativa.
- Diagnóstico deve ser simples e legível.
- Sugira objeções com respostas curtas e úteis (pré-respostas).
- Produtos sugeridos devem referenciar Produto_ID quando possível (dos candidatos).
- Tom e tamanho: selecione 1 sugestão principal e até 2 alternativas.

QUANTIDADE (obrigatório):
- "diagnostico": 3 a 4 itens (curtos)
- "pontos_prioritarios": 3 a 5 itens
- "gatilhos_comerciais": 1 a 3 itens (inclua opções suaves/consultivas e opções mais diretas)
- "objecoes_e_respostas": 1 a 3 itens
- "produtos_sugeridos": sem limite de itens (preferir Produto_ID dos candidatos; se sugerir algo fora, explique e deixe produto_id vazio)
- "tom_sugerido": 1 principal + 3 alternativas
- "tamanho_pitch": 1 principal + 2 alternativas (Pequeno, Médio, Longo)

Formato obrigatório:

{
  "diagnostico": [
    {"id":"d1","texto":"..."}
  ],
  "pontos_prioritarios": [
    {"id":"p1","texto":"..."}
  ],
  "gatilhos_comerciais": [
    {"id":"g1","texto":"..."}
  ],
  "objecoes_e_respostas": [
    {"id":"o1","objecao":"...","resposta":"..."}
  ],
  "produtos_sugeridos": [
    {"id":"s1","produto_id":"P12","texto":"..."}
  ],
  "tom_sugerido": {
    "principal":{"id":"t1","texto":"..."},
    "alternativas":[{"id":"t2","texto":"..."},{"id":"t3","texto":"..."}]
  },
  "tamanho_pitch": {
    "principal":{"id":"l2","texto":"Pequeno|Médio|Longo"},
    "alternativas":[{"id":"l1","texto":"..."},{"id":"l3","texto":"..."}]
  }
}
"""

    user_payload = {
        "cliente_info": cliente_info,
        "prompt_assessor": prompt_assessor,
        "jornada_selecionada": jornada_selecionada,
        "carteira_summary": carteira_summary,
        "investimentos_atuais": investimentos_list,
        "produtos_candidatos": produtos_list,
        "kb_context": kb_docs,
    }

    resp = get_openai_client().chat.completions.create(
        model=model,
        temperature=1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    parsed = json.loads(resp.choices[0].message.content)

    if trace_context:
        tracer = trace_context.get("tracer")
        parent_run_id = trace_context.get("parent_run_id")
        if tracer and parent_run_id:
            tracer.log_child_run(
                parent_run_id,
                name="pitch_step_5_structurer_llm",
                run_type="llm",
                inputs={
                    "model": model,
                    "temperature": 1,
                    "response_format": {"type": "json_object"},
                    "system_prompt": system_prompt,
                    "user_payload": user_payload,
                },
                outputs={
                    "response": parsed,
                    "usage": _usage_dict(resp),
                },
                metadata={"step": "step_5"},
                tags=["pitch", "llm", "step_5"],
            )

    return parsed
