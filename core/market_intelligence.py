import json
from datetime import datetime
from typing import Any

from core.exa_client import search_exa
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
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _normalize_result(item: dict[str, Any], source_type: str, sector: str | None = None, company: str | None = None) -> dict[str, Any]:
    highlights = item.get("highlights") or []
    text = item.get("text") or ""
    summary_seed = " ".join(highlights[:2]).strip() or text[:280]
    return {
        "title": (item.get("title") or "Sem título").strip(),
        "url": item.get("url") or "",
        "published_at": item.get("publishedDate") or item.get("published_date") or "",
        "source": item.get("author") or item.get("site") or "Fonte não identificada",
        "highlights": highlights,
        "summary_seed": summary_seed,
        "source_type": source_type,
        "sector": sector,
        "company": company,
    }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("url") or "", item.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _classify_and_summarize(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []

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

    response = client.chat.completions.create(
        model="gpt-5-mini",
        #temperature=0.2, # gpt-5 não tem temperatura
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um analista de mercado brasileiro. "
                    "Classifique cada notícia em uma categoria: earnings, regulatório, macro, M&A, corporate, crédito/risco. "
                    "Responda em JSON com a chave 'items', contendo lista com: idx, resumo_pt, tipo_evento, tema_relacionado, score_relevancia (0-100)."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    enriched = {row["idx"]: row for row in parsed.get("items", []) if isinstance(row, dict) and "idx" in row}

    output = []
    for idx, item in enumerate(items):
        meta = enriched.get(idx, {})
        event_type = meta.get("tipo_evento") if meta.get("tipo_evento") in EVENT_TYPES else "corporate"
        score = meta.get("score_relevancia")
        try:
            score_value = int(score)
        except Exception:
            score_value = 50
        item["event_type"] = event_type
        item["short_summary"] = (meta.get("resumo_pt") or item["summary_seed"])[:340]
        item["related_theme"] = meta.get("tema_relacionado") or item.get("company") or item.get("sector") or "Brasil"
        item["relevance_score"] = max(0, min(100, score_value))
        output.append(item)
    return output


def _sector_consolidated_summary(sector: str, companies: list[str], items: list[dict[str, Any]]) -> str:
    if not items:
        return "Sem notícias relevantes no período selecionado."

    client = get_openai_client()
    brief = [
        {
            "titulo": item["title"],
            "tipo": item.get("event_type"),
            "empresa": item.get("company"),
            "resumo": item.get("short_summary"),
        }
        for item in items[:12]
    ]

    response = client.chat.completions.create(
        model="gpt-5-mini",
        # temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "Resuma em português-BR, em até 4 frases objetivas, o que está movimentando o setor para um assessor financeiro.",
            },
            {
                "role": "user",
                "content": json.dumps({"setor": sector, "empresas": companies, "noticias": brief}, ensure_ascii=False),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def fetch_market_intelligence(days: int = 7, sector: str | None = None) -> dict[str, Any]:
    radar_queries = [
        "Brasil mercado financeiro resultados corporativos guidance dividendos M&A CVM Banco Central",
        "Brasil mudanças regulatórias CVM Banco Central fiscal juros inflação governo",
        "Brasil follow-on default recuperação judicial troca de CEO reestruturação empresas listadas",
    ]

    radar_items: list[dict[str, Any]] = []
    for query in radar_queries:
        raw = search_exa(query, days=days, num_results=10, include_domains=TRUSTED_DOMAINS)
        radar_items.extend(_normalize_result(item, source_type="radar") for item in raw)

    sectors_payload: list[dict[str, Any]] = []
    sectors_to_fetch = {sector: SECTOR_COMPANIES[sector]} if sector in SECTOR_COMPANIES else SECTOR_COMPANIES
    for sector_name, companies in sectors_to_fetch.items():
        sector_news: list[dict[str, Any]] = []

        sector_query = f"Brasil setor {sector_name} notícias mercado financeiro empresas listadas"
        raw_sector = search_exa(sector_query, days=days, num_results=8, include_domains=TRUSTED_DOMAINS)
        sector_news.extend(_normalize_result(item, source_type="sector", sector=sector_name) for item in raw_sector)

        for company in companies:
            company_query = f"Brasil {company} resultado guidance M&A dívida regulação"
            raw_company = search_exa(company_query, days=days, num_results=4, include_domains=TRUSTED_DOMAINS)
            sector_news.extend(_normalize_result(item, source_type="company", sector=sector_name, company=company) for item in raw_company)

        sector_news = _dedupe(sector_news)
        sector_news = _classify_and_summarize(sector_news)
        sector_news.sort(key=lambda x: (x.get("relevance_score", 0), _parse_date(x.get("published_at"))), reverse=True)
        sectors_payload.append(
            {
                "sector": sector_name,
                "companies": companies,
                "summary": _sector_consolidated_summary(sector_name, companies, sector_news),
                "news": sector_news[:20],
            }
        )

    radar_items = _dedupe(radar_items)
    radar_items = _classify_and_summarize(radar_items)
    radar_items.sort(key=lambda x: (x.get("relevance_score", 0), _parse_date(x.get("published_at"))), reverse=True)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "time_range_days": days,
        "radar_events": radar_items,
        "sectors": sectors_payload,
    }
