import math
from datetime import date, datetime
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from core.langchain_runtime import (
    build_runnable_config,
    get_chat_model,
    json_dumps_safe,
    parse_json_output,
    str_output_parser,
)
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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _coalesce(cliente_info: dict, *keys: str):
    for key in keys:
        value = cliente_info.get(key)
        if not _is_missing(value):
            return value
    return None


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


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    return None


def _compute_days_since_last_contact(cliente_info: dict, today: date) -> int | None:
    for key in ("Dias_Sem_Contato", "Ultima_Interacao_Dias", "Dias_Desde_Ultimo_Contato"):
        value = cliente_info.get(key)
        if isinstance(value, (int, float)):
            return max(int(value), 0)
    last_contact = _parse_date(cliente_info.get("Data_Ultimo_Contato"))
    if last_contact:
        return max((today - last_contact).days, 0)
    return None


def _compute_age(cliente_info: dict, today: date) -> int | None:
    age = _coalesce(cliente_info, "Idade")
    if isinstance(age, (int, float)) and not _is_missing(age):
        return max(int(age), 0)

    birth_date = _parse_date(_coalesce(cliente_info, "Data_Nascimento"))
    if birth_date is None:
        return None

    years = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        years -= 1
    return max(years, 0)


def _compute_days_until_birthday(cliente_info: dict, today: date) -> int | None:
    current = cliente_info.get("Aniversario_Proximo_Dias")
    if isinstance(current, (int, float)):
        return max(int(current), 0)

    birth_date = _parse_date(cliente_info.get("Data_Nascimento"))
    month = cliente_info.get("Mes_Aniversario")
    if birth_date:
        month = birth_date.month
        day = birth_date.day
    else:
        if not isinstance(month, (int, float)):
            return None
        month = int(month)
        day = 1

    if not 1 <= int(month) <= 12:
        return None
    year = today.year
    try:
        next_birthday = date(year, int(month), int(day))
    except ValueError:
        return None
    if next_birthday < today:
        next_birthday = date(year + 1, int(month), int(day))
    return (next_birthday - today).days


def _derive_relationship_status(cliente_info: dict, ultima_interacao_dias: int | None) -> str | None:
    status = _coalesce(cliente_info, "Status_Relacionamento")
    if isinstance(status, str) and status.strip():
        return status.strip()

    score_churn = _coalesce(cliente_info, "Score_Risco_Churn")
    if isinstance(score_churn, (int, float)) and not _is_missing(score_churn):
        if score_churn >= 75:
            return "Em Risco"
        if score_churn >= 45:
            return "Morno"
        return "Ativo"

    if isinstance(ultima_interacao_dias, int):
        if ultima_interacao_dias >= 60:
            return "Em Risco"
        if ultima_interacao_dias >= 30:
            return "Morno"
        return "Ativo"
    return None


def _derive_capture_potential(cliente_info: dict) -> str | None:
    potencial = _coalesce(cliente_info, "Potencial_Captacao")
    if isinstance(potencial, str) and potencial.strip():
        return potencial.strip()

    patrimonio_outros = _safe_float(_coalesce(cliente_info, "Patrimonio_Investido_Outros", "Patrimonio_Investido_Fora"))
    dinheiro_disponivel = _safe_float(_coalesce(cliente_info, "Dinheiro_Disponivel_Para_Investir"))
    if patrimonio_outros >= 750000 or dinheiro_disponivel >= 250000:
        return "Alto"
    if patrimonio_outros >= 200000 or dinheiro_disponivel >= 80000:
        return "Médio"
    return "Baixo"


def _derive_recent_life_event(cliente_info: dict, today: date, aniversario_proximo_dias: int | None) -> str:
    if isinstance(aniversario_proximo_dias, int) and aniversario_proximo_dias <= 21:
        return f"Aniversário em {aniversario_proximo_dias} dias"

    objetivo = str(cliente_info.get("Objetivo_Principal") or "").strip()
    objetivo_data = _parse_date(cliente_info.get("Data_Objetivo_Financeiro"))
    if objetivo and objetivo_data:
        dias_para_objetivo = (objetivo_data - today).days
        if 0 <= dias_para_objetivo <= 120:
            return f"Meta de {objetivo.lower()} prevista para {dias_para_objetivo} dias"

    return ""


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
    today = date.today()
    investimentos_rows = _extract_rows(
        investimentos_cliente_df,
        columns=["Produto", "Categoria", "Valor_Investido"],
    )
    total_investido = sum(_safe_float(row.get("Valor_Investido")) for row in investimentos_rows)
    dinheiro_disponivel = _safe_float(cliente_info.get("Dinheiro_Disponivel_Para_Investir"))
    patrimonio_conosco = _safe_float(cliente_info.get("Patrimonio_Investido_Conosco"))
    patrimonio_outros = _safe_float(_coalesce(cliente_info, "Patrimonio_Investido_Outros", "Patrimonio_Investido_Fora"))
    spread_vs_cdi = carteira_summary.get("spread_vs_cdi_12m")
    top_categories = _top_categories(investimentos_rows)
    top_category_weight = top_categories[0]["peso"] if top_categories else 0.0
    ultima_interacao_dias = _compute_days_since_last_contact(cliente_info, today)
    aniversario_proximo_dias = _compute_days_until_birthday(cliente_info, today)
    evento_vida = _derive_recent_life_event(cliente_info, today, aniversario_proximo_dias)
    idade = _compute_age(cliente_info, today)
    score_risco_churn = _coalesce(cliente_info, "Score_Risco_Churn")
    status_relacionamento = _derive_relationship_status(cliente_info, ultima_interacao_dias)
    potencial_captacao = _derive_capture_potential(cliente_info)

    return {
        "contexto": {"data_atual": today.isoformat()},
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
            "objetivo_principal": _coalesce(cliente_info, "Objetivo_Principal"),
            "data_objetivo_financeiro": _coalesce(cliente_info, "Data_Objetivo_Financeiro"),
            "valor_objetivo_financeiro": _coalesce(cliente_info, "Valor_Objetivo_Financeiro"),
            "score_risco_churn": score_risco_churn,
            "idade": idade,
            "status_relacionamento": status_relacionamento,
            "potencial_captacao": potencial_captacao,
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
            "forca_sinal": 0.45,
            "gatilhos": ["manter relacionamento ativo", "gerar conversa consultiva mesmo sem campanha específica"],
            "papel": "baseline relacional",
        },
        {
            "tipo": "parabens",
            "forca_sinal": 0.2 + (0.45 if crm.get("tem_janela_aniversario") else 0) + (0.2 if crm.get("tem_evento_relacional") else 0),
            "gatilhos": ["janela de aniversário ou evento relacional detectado"],
            "papel": "relacional contextual",
        },
        {
            "tipo": "oferta_produto",
            "forca_sinal": 0.3 + (0.3 if cash_ratio >= 0.12 else 0) + (0.2 if outside_ratio >= 0.35 else 0),
            "gatilhos": ["caixa disponível", "oportunidade de share of wallet", "recomendação orientada por suitability"],
            "papel": "expansão comercial",
        },
        {
            "tipo": "rebalanceamento",
            "forca_sinal": 0.25
            + (0.25 if isinstance(spread_vs_cdi, (int, float)) and spread_vs_cdi < -0.01 else 0)
            + (0.2 if top_category_weight >= 0.45 else 0),
            "gatilhos": ["concentração relevante", "performance abaixo do CDI", "ajuste de carteira"],
            "papel": "otimização de carteira",
        },
        {
            "tipo": "reativacao",
            "forca_sinal": 0.25 + (0.35 if isinstance(days_since_contact, (int, float)) and days_since_contact >= 45 else 0),
            "gatilhos": ["tempo elevado sem contato", "retomar cadência comercial"],
            "papel": "recuperação de cadência",
        },
    ]

    ranked = sorted(candidates, key=lambda item: item["forca_sinal"], reverse=True)
    return ranked


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
- Não use playbooks pré-definidos: descubra as melhores abordagens a partir do contexto completo.
- Considere explicitamente o campo `data_atual` recebido no payload para toda análise temporal.
- Você tem liberdade total para definir as categorias e teses de abordagem.
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
      "categoria": "string",
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
        "data_atual": date.today().isoformat(),
        "prompt_assessor": prompt_assessor,
        "signal_summary": signal_summary,
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

    messages = prompt.invoke(
        {"user_payload": json_dumps_safe(user_payload, ensure_ascii=False)},
        config=config,
    )
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
- Considere explicitamente o campo `data_atual` recebido no payload para toda análise temporal.
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
        "data_atual": date.today().isoformat(),
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

    messages = prompt.invoke(
        {"user_payload": json_dumps_safe(user_payload, ensure_ascii=False)},
        config=config,
    )
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
