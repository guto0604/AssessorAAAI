import json
from datetime import datetime
from typing import Any

from core.tavily_client import search_tavily
from core.openai_client import get_openai_client

TRUSTED_DOMAINS = [
    "valor.globo.com",
    "www.infomoney.com.br",
    "exame.com",
    "bloomberglinea.com.br",
    "www.reuters.com",
    "www.cnnbrasil.com.br",
    "www1.folha.uol.com.br",
    "www.estadao.com.br",
    "g1.globo.com",
    "www.bcb.gov.br",
    "www.cvm.gov.br",
    "www.gov.br",
    "economia.uol.com.br",
    "www.moneytimes.com.br",
    "www.seudinheiro.com",
    "www.neofeed.com.br",
    "www.suno.com.br",
    "www.nordinvestimentos.com.br",
    "www.spacemoney.com.br",
    "pipelinevalor.globo.com",
    "einvestidor.estadao.com.br",
    "agenciabrasil.ebc.com.br",
    "www.canalrural.com.br",
    "www.abrasce.com.br",
    "www.anbima.com.br",
    "ri.b3.com.br",
    "www.brasildefato.com.br",
    "www.poder360.com.br",
    "www.conjur.com.br",
    "www.tecmundo.com.br",
    "epocanegocios.globo.com",
]

SECTOR_COMPANIES = {
    "Bancos": ["Santander Brasil", "Nubank", "Itaú Unibanco", "Bradesco", "Banco do Brasil"],
    "Energia Elétrica": ["Eletrobras", "Engie Brasil", "CPFL Energia", "Energisa", "Neoenergia"],
    "Petróleo e Gás": ["Petrobras", "PRIO", "Raízen", "Ultrapar", "Brava Energia"],
    "Mineração e Siderurgia": ["Vale", "Gerdau", "Usiminas", "CSN", "CBA"],
    "Varejo": ["Magazine Luiza", "Via", "Lojas Renner", "Assaí", "Grupo Mateus"],
    "Saúde": ["Rede D'Or", "Hapvida", "Fleury", "Sabin", "Oncoclínicas"],
    "Telecom": ["Telefônica Brasil", "TIM Brasil", "Claro Brasil", "Oi", "Desktop"],
}

EVENT_TYPES = ["earnings", "regulatório", "macro", "M&A", "corporate", "crédito/risco"]


def _parse_date(value: str | None) -> datetime:
    """Responsável por interpretar date no contexto da aplicação de assessoria.

    Args:
        value: Valor de entrada necessário para processar 'value'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _normalize_result(item: dict[str, Any], source_type: str, sector: str | None = None, company: str | None = None) -> dict[str, Any]:
    """Responsável por processar result no contexto da aplicação de assessoria.

    Args:
        item: Valor de entrada necessário para processar 'item'.
        source_type: Valor de entrada necessário para processar 'source_type'.
        sector: Valor de entrada necessário para processar 'sector'.
        company: Valor de entrada necessário para processar 'company'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    title = (item.get("title") or "Sem título").strip()
    content = item.get("content") or item.get("raw_content") or ""
    summary_seed = (content or title)[:280]
    return {
        "title": title,
        "url": item.get("url") or "",
        "published_at": item.get("published_date") or item.get("publishedDate") or "",
        "source": item.get("source") or item.get("site_name") or item.get("site") or "Fonte não identificada",
        "highlights": [],
        "summary_seed": summary_seed,
        "source_type": source_type,
        "sector": sector,
        "company": company,
    }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responsável por executar uma etapa do fluxo da aplicação de assessoria.

    Args:
        items: Valor de entrada necessário para processar 'items'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("url") or "", item.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _classify_and_summarize(items: list[dict[str, Any]], include_api_metrics: bool = False):
    """Responsável por processar and summarize no contexto da aplicação de assessoria.

    Args:
        items: Valor de entrada necessário para processar 'items'.
        include_api_metrics: Indica se a função deve retornar métricas de uso de API junto ao resultado.

    Returns:
        Resumo objetivo do conteúdo analisado.
    
    """
    if not items:
        return {"items": [], "api_metrics": None} if include_api_metrics else []

    client = get_openai_client()
    payload = [
        {
            "idx": idx,
            "titulo": item["title"],
            "texto_base": item["summary_seed"],
            "setor": item.get("sector"),
            "empresa": item.get("company"),
        }
        for idx, item in enumerate(items)
    ]

    system_prompt = (
        "Você é um analista de mercado brasileiro. "
        "Classifique cada notícia em uma categoria: earnings, regulatório, macro, M&A, corporate, crédito/risco. "
        "Responda em JSON com a chave 'items', contendo lista com: idx, resumo_pt, tipo_evento, tema_relacionado, score_relevancia (0-100)."
    )
    user_prompt = json.dumps(payload, ensure_ascii=False)

    response = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    enriched = {row["idx"]: row for row in parsed.get("items", []) if isinstance(row, dict) and "idx" in row}

    output = []
    for idx, item in enumerate(items):
        meta = enriched.get(idx, {})
        event_type = meta.get("tipo_evento") if meta.get("tipo_evento") in EVENT_TYPES else "corporate"
        try:
            score_value = int(meta.get("score_relevancia"))
        except Exception:
            score_value = 50
        item["event_type"] = event_type
        item["short_summary"] = (meta.get("resumo_pt") or item["summary_seed"])[:340]
        item["related_theme"] = meta.get("tema_relacionado") or item.get("company") or item.get("sector") or "Brasil"
        item["relevance_score"] = max(0, min(100, score_value))
        output.append(item)
    if not include_api_metrics:
        return output

    usage = response.usage or {}
    return {
        "items": output,
        "api_metrics": {
            "provider": "openai",
            "step": "classify_and_summarize",
            "model": response.model,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "prompt": {
                "system": system_prompt,
                "user": user_prompt,
            },
            "output": content,
        },
    }


def _build_macro_queries(sector_name: str, companies: list[str]) -> list[str]:
    """Responsável por montar macro queries no contexto da aplicação de assessoria.

    Args:
        sector_name: Valor de entrada necessário para processar 'sector_name'.
        companies: Valor de entrada necessário para processar 'companies'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    companies_text = " ou ".join(companies)
    return [
        f"Brasil {sector_name} mercado financeiro resultados guidance dividendos",
        f"Brasil {sector_name} regulação CVM Banco Central governo tributação",
        f"Brasil {sector_name} M&A aquisição fusão joint venture",
        f"Brasil {companies_text} notícia investimento bolsa",
    ]


def _build_company_queries(company: str) -> list[str]:
    """Responsável por montar company queries no contexto da aplicação de assessoria.

    Args:
        company: Valor de entrada necessário para processar 'company'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    return [
        f"Brasil {company} resultado guidance receita lucro ebitda",
        f"Brasil {company} M&A aquisição venda ativo parceria",
        f"Brasil {company} dívida captação rating default recuperação judicial",
        f"Brasil {company} regulação CVM ANEEL ANP ANS Anatel CADE",
    ]


def fetch_market_intelligence(days: int = 7, sector: str | None = None, include_api_metrics: bool = False) -> dict[str, Any]:
    """Processa dados de mercado para gerar contexto acionável ao assessor.

    Args:
        days: Valor de entrada necessário para processar 'days'.
        sector: Valor de entrada necessário para processar 'sector'.
        include_api_metrics: Indica se a função deve retornar métricas de uso de API junto ao resultado.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if sector not in SECTOR_COMPANIES:
        sectors_to_fetch = SECTOR_COMPANIES
    else:
        sectors_to_fetch = {sector: SECTOR_COMPANIES[sector]}

    all_news: list[dict[str, Any]] = []
    sectors_payload: list[dict[str, Any]] = []

    for sector_name, companies in sectors_to_fetch.items():
        sector_news: list[dict[str, Any]] = []

        for query in _build_macro_queries(sector_name, companies):
            raw = search_tavily(
                query,
                days=days,
                num_results=8,
                include_domains=TRUSTED_DOMAINS,
                lightweight=True,
            )
            sector_news.extend(_normalize_result(item, source_type="setor", sector=sector_name) for item in raw)

        for company in companies:
            for query in _build_company_queries(company):
                raw_company = search_tavily(
                    query,
                    days=days,
                    num_results=6,
                    include_domains=TRUSTED_DOMAINS,
                    lightweight=True,
                )
                sector_news.extend(_normalize_result(item, source_type="empresa", sector=sector_name, company=company) for item in raw_company)

        sector_news = _dedupe(sector_news)
        all_news.extend(sector_news)
        sectors_payload.append({"sector": sector_name, "companies": companies, "news": sector_news})

    ranked_news = _dedupe(all_news)
    classify_result = _classify_and_summarize(ranked_news, include_api_metrics=include_api_metrics)
    if include_api_metrics:
        ranked_news = classify_result["items"]
        api_calls = [classify_result["api_metrics"]] if classify_result.get("api_metrics") else []
    else:
        ranked_news = classify_result
        api_calls = []
    ranked_news.sort(key=lambda x: (x.get("relevance_score", 0), _parse_date(x.get("published_at"))), reverse=True)

    for sector_block in sectors_payload:
        sector_ranked = [news for news in ranked_news if news.get("sector") == sector_block["sector"]]
        sector_block["news"] = sector_ranked[:35]

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "time_range_days": days,
        "ranked_news": ranked_news[:60],
        "sectors": sectors_payload,
    }
    if include_api_metrics:
        result["api_calls"] = api_calls
    return result
