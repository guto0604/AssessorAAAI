from __future__ import annotations

from collections import defaultdict


LIQUIDITY_CATEGORY_HINTS = {
    "Renda Fixa": "alta",
    "Tesouro": "alta",
    "Caixa": "imediata",
    "Previdência": "baixa",
    "Renda Variável": "média",
    "Fundos": "média",
    "Alternativos": "baixa",
    "Multimercado": "média",
}


def _safe_float(value) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _category_liquidity(category: str) -> str:
    for key, label in LIQUIDITY_CATEGORY_HINTS.items():
        if key.lower() in (category or "").lower():
            return label
    return "não informada"


def load_client_master_context(client_id: str) -> dict:
    from core.data_loader import get_cliente_by_id, get_investimentos_by_cliente, load_produtos_full

    client_info = get_cliente_by_id(client_id)
    investimentos_df = get_investimentos_by_cliente(client_id)
    try:
        holdings = investimentos_df.to_dict(orient="records")
    except Exception:
        holdings = []

    total_invested = sum(_safe_float(item.get("Valor_Investido")) for item in holdings)
    by_category = defaultdict(float)
    by_product = []
    for item in holdings:
        amount = _safe_float(item.get("Valor_Investido"))
        by_category[item.get("Categoria") or "Não categorizado"] += amount
        by_product.append(
            {
                "product_name": item.get("Produto", "Produto sem nome"),
                "category": item.get("Categoria", "Não categorizado"),
                "invested_amount": amount,
                "liquidity_hint": _category_liquidity(item.get("Categoria", "")),
            }
        )

    by_category_list = [
        {
            "category": category,
            "invested_amount": amount,
            "allocation_pct": round((amount / total_invested) * 100, 2) if total_invested else 0.0,
            "liquidity_hint": _category_liquidity(category),
        }
        for category, amount in sorted(by_category.items(), key=lambda entry: entry[1], reverse=True)
    ]

    produtos_df = load_produtos_full()
    try:
        produtos_catalog = produtos_df.to_dict(orient="records")
    except Exception:
        produtos_catalog = []

    holdings_lookup = {item.get("product_name") for item in by_product}
    matched_products = [
        product for product in produtos_catalog
        if product.get("Nome_Produto") in holdings_lookup
    ]

    return {
        "client_info": client_info,
        "portfolio": {
            "total_invested": total_invested,
            "holdings": by_product,
            "allocation_by_category": by_category_list,
            "holdings_count": len(by_product),
        },
        "product_reference": matched_products,
        "sources": {
            "client_info": {"source": "informacoes_cliente.parquet", "records": 1},
            "portfolio": {"source": "investimentos_cliente.parquet", "records": len(by_product)},
            "product_reference": {"source": "produtos.parquet", "records": len(matched_products)},
        },
    }


def select_relevant_client_context(master_context: dict, advisor_prompt: str, additional_notes: str = "", tone_focus: str = "") -> dict:
    from core.investment_case_builder.data_relevance_agent import DataRelevanceAgent

    agent = DataRelevanceAgent()
    return agent.run(
        master_context=master_context,
        advisor_prompt=advisor_prompt,
        additional_notes=additional_notes,
        tone_focus=tone_focus,
    ).payload
