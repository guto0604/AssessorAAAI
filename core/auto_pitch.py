import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from core.langchain_runtime import build_runnable_config, get_chat_model, parse_json_output, str_output_parser
from core.pitch_structurer import read_kb_files_tool
from core.source_selector import list_kb_files


AUTO_PITCH_PRIORITY_MODEL = "gpt-5.4"
AUTO_PITCH_COMMUNICATION_MODEL = "gpt-5.4"
AUTO_PITCH_TEMPERATURE = 1


def _build_api_metrics(response, *, provider: str = "openai", prompt: dict | None = None, output: str | None = None) -> dict:
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_rows(df_like, columns: list[str] | None = None) -> list[dict]:
    if df_like is None:
        return []

    if hasattr(df_like, "to_dict"):
        rows = df_like.to_dict(orient="records")
    else:
        rows = list(df_like)

    if not columns:
        return rows

    filtered = []
    for row in rows:
        filtered.append({column: row.get(column) for column in columns})
    return filtered


def _top_categories(investimentos_rows: list[dict]) -> list[dict]:
    totals: dict[str, float] = {}
    total_investido = 0.0
    for row in investimentos_rows:
        categoria = str(row.get("Categoria") or "Sem categoria")
        valor = _safe_float(row.get("Valor_Investido"))
        totals[categoria] = totals.get(categoria, 0.0) + valor
        total_investido += valor

    ranking = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    top = []
    for categoria, valor in ranking[:4]:
        peso = (valor / total_investido) if total_investido else 0.0
        top.append({"categoria": categoria, "valor": valor, "peso": round(peso, 4)})
    return top


def build_auto_pitch_signal_summary(
    cliente_info: dict,
    carteira_summary: dict,
    investimentos_cliente_df,
) -> dict:
    investimentos_rows = _extract_rows(
        investimentos_cliente_df,
        columns=["Produto", "Categoria", "Valor_Investido"],
    )
    total_investido = sum(_safe_float(row.get("Valor_Investido")) for row in investimentos_rows)
    dinheiro_disponivel = _safe_float(cliente_info.get("Dinheiro_Disponivel_Para_Investir"))
    patrimonio_conosco = _safe_float(cliente_info.get("Patrimonio_Investido_Conosco"))
    patrimonio_outros = _safe_float(
        cliente_info.get("Patrimonio_Investido_Outros", cliente_info.get("Patrimonio_Investido_Fora"))
    )
    spread_vs_cdi = carteira_summary.get("spread_vs_cdi_12m")
    top_categories = _top_categories(investimentos_rows)
    top_category_weight = top_categories[0]["peso"] if top_categories else 0.0
    ultima_interacao_dias = cliente_info.get("Dias_Sem_Contato", cliente_info.get("Ultima_Interacao_Dias"))
    aniversario_proximo_dias = cliente_info.get("Aniversario_Proximo_Dias")
    evento_vida = cliente_info.get("Evento_Vida_Recente")

    return {
        "cliente": {
            "cliente_id": cliente_info.get("Cliente_ID"),
            "nome": cliente_info.get("Nome"),
            "perfil_suitability": cliente_info.get("Perfil_Suitability"),
            "patrimonio_conosco": patrimonio_conosco,
            "patrimonio_outros": patrimonio_outros,
            "dinheiro_disponivel": dinheiro_disponivel,
            "rentabilidade_12_meses": cliente_info.get("Rentabilidade_12_meses"),
            "cdi_12_meses": cliente_info.get("CDI_12_Meses"),
            "spread_vs_cdi_12m": spread_vs_cdi,
            "ultima_interacao_dias": ultima_interacao_dias,
            "aniversario_proximo_dias": aniversario_proximo_dias,
            "evento_vida_recente": evento_vida,
        },
        "portfolio": {
            "total_investido_calculado": round(total_investido, 2),
            "quantidade_posicoes": len(investimentos_rows),
            "top_categorias": top_categories,
            "maior_categoria_peso": round(top_category_weight, 4),
            "caixa_sobre_patrimonio": round((dinheiro_disponivel / patrimonio_conosco), 4) if patrimonio_conosco else None,
            "patrimonio_fora_ratio": round((patrimonio_outros / patrimonio_conosco), 4) if patrimonio_conosco else None,
        },
        "crm_readiness": {
            "tem_follow_up_atrasado": isinstance(ultima_interacao_dias, (int, float)) and ultima_interacao_dias >= 45,
            "tem_evento_relacional": bool(evento_vida),
            "tem_janela_aniversario": isinstance(aniversario_proximo_dias, (int, float)) and aniversario_proximo_dias <= 14,
        },
    }


def build_priority_candidates(signal_summary: dict) -> list[dict]:
    cliente = signal_summary.get("cliente", {})
    portfolio = signal_summary.get("portfolio", {})
    crm = signal_summary.get("crm_readiness", {})

    cash_ratio = portfolio.get("caixa_sobre_patrimonio") or 0.0
    outside_ratio = portfolio.get("patrimonio_fora_ratio") or 0.0
    spread_vs_cdi = cliente.get("spread_vs_cdi_12m")
    top_category_weight = portfolio.get("maior_categoria_peso") or 0.0
    days_since_contact = cliente.get("ultima_interacao_dias")

    candidates = [
        {
            "tipo": "contato_padrão",
            "score_base": 40,
            "gatilhos": ["manter relacionamento ativo", "gerar conversa consultiva mesmo sem campanha específica"],
        },
        {
            "tipo": "parabens",
            "score_base": 20 + (45 if crm.get("tem_janela_aniversario") else 0) + (20 if crm.get("tem_evento_relacional") else 0),
            "gatilhos": ["janela de aniversário ou evento relacional detectado"],
        },
        {
            "tipo": "oferta_produto",
            "score_base": 30 + (30 if cash_ratio >= 0.12 else 0) + (20 if outside_ratio >= 0.35 else 0),
            "gatilhos": ["caixa disponível", "oportunidade de share of wallet", "recomendação orientada por suitability"],
        },
        {
            "tipo": "rebalanceamento",
            "score_base": 25 + (25 if isinstance(spread_vs_cdi, (int, float)) and spread_vs_cdi < -0.01 else 0) + (20 if top_category_weight >= 0.45 else 0),
            "gatilhos": ["concentração relevante", "performance abaixo do CDI", "ajuste de carteira"],
        },
        {
            "tipo": "reativacao",
            "score_base": 25 + (35 if isinstance(days_since_contact, (int, float)) and days_since_contact >= 45 else 0),
            "gatilhos": ["tempo elevado sem contato", "retomar cadência comercial"],
        },
    ]

    ranked = sorted(candidates, key=lambda item: item["score_base"], reverse=True)
    return ranked[:4]


def _sanitize_priorities(parsed: dict) -> dict:
    prioridades = parsed.get("prioridades", [])[:3]
    sanitized = []
    for index, item in enumerate(prioridades, start=1):
        sanitized.append(
            {
                "priority_rank": item.get("priority_rank", index),
                "priority_id": item.get("priority_id", f"p{index}"),
                "categoria": item.get("categoria", "contato_padrão"),
                "titulo": item.get("titulo", f"Prioridade {index}"),
                "objetivo": item.get("objetivo", ""),
                "porque_agora": item.get("porque_agora", ""),
                "sinais_dados": item.get("sinais_dados", []),
                "abordagem_recomendada": item.get("abordagem_recomendada", ""),
                "canal_recomendado": item.get("canal_recomendado", "WhatsApp"),
                "tom": item.get("tom", "consultivo"),
                "products_selected_ids": item.get("products_selected_ids", [])[:3],
                "kb_files_selected": item.get("kb_files_selected", [])[:5],
            }
        )

    return {
        "prioridades": sanitized,
        "resumo_executivo": parsed.get("resumo_executivo", ""),
        "signal_summary": parsed.get("signal_summary") or {},
    }


def _sanitize_communication(parsed: dict) -> dict:
    return {
        "resumo_estrategico": parsed.get("resumo_estrategico", ""),
        "racional_argumentativo": parsed.get("racional_argumentativo", []),
        "provas_evidencias": parsed.get("provas_evidencias", []),
        "mensagem_principal": parsed.get("mensagem_principal", ""),
        "mensagem_follow_up": parsed.get("mensagem_follow_up", ""),
        "cta": parsed.get("cta", ""),
        "observacoes_assessor": parsed.get("observacoes_assessor", []),
    }


def _log_llm_step(trace_context: dict | None, *, name: str, inputs: dict, outputs: dict, api_metrics: dict) -> None:
    tracer = (trace_context or {}).get("tracer")
    parent_run_id = (trace_context or {}).get("parent_run_id")
    if tracer and parent_run_id:
        tracer.log_child_run(
            parent_run_id,
            name=name,
            run_type="llm",
            inputs=inputs,
            outputs=outputs,
            metadata={
                "feature": "auto_pitch",
                "model": api_metrics.get("model"),
                "input_tokens": api_metrics.get("input_tokens"),
                "output_tokens": api_metrics.get("output_tokens"),
                "total_tokens": api_metrics.get("total_tokens"),
                "latency_ms": api_metrics.get("latency_ms"),
            },
            tags=["pitch", "auto-pitch", name],
        )


def generate_auto_pitch_priorities(
    cliente_info: dict,
    carteira_summary: dict,
    investimentos_cliente_df,
    produtos_df,
    prompt_assessor: str = "",
    kb_dir: str = "knowledge_base",
    model: str = AUTO_PITCH_PRIORITY_MODEL,
    trace_context: dict | None = None,
    include_api_metrics: bool = False,
):
    signal_summary = build_auto_pitch_signal_summary(cliente_info, carteira_summary, investimentos_cliente_df)
    candidate_playbooks = build_priority_candidates(signal_summary)
    produtos_catalogo = _extract_rows(
        produtos_df,
        columns=["Produto_ID", "Nome_Produto", "Categoria", "Subcategoria", "Risco_Nivel", "Suitability_Ideal"],
    )[:18]
    kb_files_available = list_kb_files(kb_dir)

    system_prompt = """
Você é um estrategista de auto-pitch para um assessor de investimentos.

Objetivo:
- Remover do assessor a responsabilidade de decidir o que abordar primeiro.
- Combinar sinais de CRM, carteira, suitability, disponibilidade de caixa e contexto opcional do assessor.
- Retornar exatamente 3 prioridades de pitch, ordenadas por prioridade.

Regras:
- Responda APENAS JSON válido.
- Use primeiro os sinais determinísticos em `candidate_playbooks`, mas refine com raciocínio contextual.
- Balanceie relacionamento e oportunidade comercial. Nem toda prioridade deve ser oferta de produto.
- Se houver evento relacional relevante, considere uma prioridade de relacionamento.
- Só recomende produto específico quando houver aderência mínima ao suitability ou ao momento do cliente.
- `products_selected_ids` deve conter até 3 IDs válidos da base de produtos.
- `kb_files_selected` deve conter até 5 arquivos .txt existentes na knowledge base.

Formato obrigatório:
{
  "resumo_executivo": "string",
  "prioridades": [
    {
      "priority_rank": 1,
      "priority_id": "p1",
      "categoria": "contato_padrão|parabens|oferta_produto|rebalanceamento|reativacao",
      "titulo": "string",
      "objetivo": "string",
      "porque_agora": "string",
      "sinais_dados": ["string", "string"],
      "abordagem_recomendada": "string",
      "canal_recomendado": "string",
      "tom": "string",
      "products_selected_ids": ["P1"],
      "kb_files_selected": ["knowledge_base/...txt"]
    }
  ]
}
"""

    user_payload = {
        "prompt_assessor": prompt_assessor,
        "signal_summary": signal_summary,
        "candidate_playbooks": candidate_playbooks,
        "produtos_catalogo": produtos_catalogo,
        "kb_files_available": kb_files_available,
    }

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{user_payload}")])
    llm = get_chat_model(
        model=model,
        temperature=AUTO_PITCH_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    config = build_runnable_config(
        run_name="auto_pitch_priority_planner",
        tags=["pitch", "auto-pitch", "priority-planning", "langchain"],
        metadata={
            "feature": "auto_pitch",
            "step": "priority_planning",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    messages = prompt.invoke({"user_payload": json.dumps(user_payload, ensure_ascii=False)}, config=config)
    response = llm.invoke(messages, config=config)
    raw_output = str_output_parser.invoke(response, config=config)
    parsed = _sanitize_priorities(parse_json_output(raw_output))
    parsed["signal_summary"] = signal_summary
    api_metrics = _build_api_metrics(response, prompt={"messages": str(messages)}, output=raw_output)
    _log_llm_step(
        trace_context,
        name="auto_pitch_priority_planner",
        inputs={"prompt_assessor": prompt_assessor, "signal_summary": signal_summary},
        outputs={"prioridades": parsed.get("prioridades", [])},
        api_metrics=api_metrics,
    )

    if include_api_metrics:
        return {"result": parsed, "api_metrics": api_metrics}
    return parsed


def generate_auto_pitch_communication(
    cliente_info: dict,
    carteira_summary: dict,
    investimentos_cliente_df,
    produtos_df,
    selected_priority: dict,
    model: str = AUTO_PITCH_COMMUNICATION_MODEL,
    trace_context: dict | None = None,
    include_api_metrics: bool = False,
):
    investimentos_rows = _extract_rows(
        investimentos_cliente_df,
        columns=["Produto", "Categoria", "Valor_Investido"],
    )
    selected_product_ids = set(selected_priority.get("products_selected_ids", []))
    produtos_rows = _extract_rows(
        produtos_df,
        columns=["Produto_ID", "Nome_Produto", "Categoria", "Subcategoria", "Risco_Nivel", "Suitability_Ideal"],
    )
    produtos_selecionados = [row for row in produtos_rows if row.get("Produto_ID") in selected_product_ids]
    kb_files = selected_priority.get("kb_files_selected", [])
    kb_docs = read_kb_files_tool.invoke({"file_paths": kb_files, "max_chars_each": 2500}) if kb_files else []

    system_prompt = """
Você é um estrategista comercial para assessoria de investimentos.

Objetivo:
- Transformar a prioridade escolhida pelo assessor em racional argumentativo e comunicação final.
- Entregar material pronto para o assessor usar sem precisar decidir a tese do zero.

Regras:
- Responda APENAS JSON válido.
- Use somente dados do cliente, carteira, prioridade selecionada, produtos selecionados e documentos internos fornecidos.
- Não invente retornos, taxas ou garantias.
- O racional deve ser acionável, em português do Brasil e orientado à conversa.
- `mensagem_principal` deve ficar pronta para WhatsApp ou e-mail curto.

Formato obrigatório:
{
  "resumo_estrategico": "string",
  "racional_argumentativo": ["string", "string", "string"],
  "provas_evidencias": ["string", "string"],
  "mensagem_principal": "string",
  "mensagem_follow_up": "string",
  "cta": "string",
  "observacoes_assessor": ["string", "string"]
}
"""

    user_payload = {
        "cliente_info": cliente_info,
        "carteira_summary": carteira_summary,
        "investimentos_atuais": investimentos_rows,
        "prioridade_selecionada": selected_priority,
        "produtos_selecionados": produtos_selecionados,
        "kb_context": kb_docs,
    }

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{user_payload}")])
    llm = get_chat_model(
        model=model,
        temperature=AUTO_PITCH_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    config = build_runnable_config(
        run_name="auto_pitch_communication_writer",
        tags=["pitch", "auto-pitch", "communication-writer", "langchain"],
        metadata={
            "feature": "auto_pitch",
            "step": "communication_generation",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    messages = prompt.invoke({"user_payload": json.dumps(user_payload, ensure_ascii=False)}, config=config)
    response = llm.invoke(messages, config=config)
    raw_output = str_output_parser.invoke(response, config=config)
    parsed = _sanitize_communication(parse_json_output(raw_output))
    api_metrics = _build_api_metrics(response, prompt={"messages": str(messages)}, output=raw_output)
    _log_llm_step(
        trace_context,
        name="auto_pitch_communication_writer",
        inputs={"selected_priority": selected_priority},
        outputs={"mensagem_principal_preview": parsed.get("mensagem_principal", "")[:240]},
        api_metrics=api_metrics,
    )

    if include_api_metrics:
        return {"result": parsed, "api_metrics": api_metrics}
    return parsed
