from datetime import datetime

import streamlit as st

from core.market_intelligence import EVENT_TYPES, SECTOR_COMPANIES, fetch_market_intelligence
from ui.state import get_tracer

TIME_RANGE_OPTIONS = {
    "24h": 1,
    "7 dias": 7,
    "30 dias": 30,
}


def _format_date(date_str: str) -> str:
    """ format date.

    Args:
        date_str: Descrição do parâmetro `date_str`.

    Returns:
        Valor de retorno da função.
    """
    if not date_str:
        return "-"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return date_str


def _filter_news(items: list[dict], selected_event_type: str, selected_sector: str, selected_company: str, selected_source: str) -> list[dict]:
    """ filter news.

    Args:
        items: Descrição do parâmetro `items`.
        selected_event_type: Descrição do parâmetro `selected_event_type`.
        selected_sector: Descrição do parâmetro `selected_sector`.
        selected_company: Descrição do parâmetro `selected_company`.
        selected_source: Descrição do parâmetro `selected_source`.

    Returns:
        Valor de retorno da função.
    """
    filtered = []
    for item in items:
        if selected_event_type != "Todos" and item.get("event_type") != selected_event_type:
            continue
        if selected_sector != "Todos" and item.get("sector") != selected_sector:
            continue
        if selected_company != "Todas" and item.get("company") != selected_company:
            continue
        if selected_source != "Todas" and item.get("source") != selected_source:
            continue
        filtered.append(item)
    return filtered


def _render_news_card(item: dict):
    """ render news card.

    Args:
        item: Descrição do parâmetro `item`.

    Returns:
        Valor de retorno da função.
    """
    score = item.get("relevance_score", 0)
    header = f"**{item.get('title', 'Sem título')}**"
    if score >= 80:
        st.markdown(f"🔴 {header}")
    elif score >= 60:
        st.markdown(f"🟡 {header}")
    else:
        st.markdown(f"⚪ {header}")

    st.write(item.get("short_summary") or item.get("summary_seed") or "Sem resumo disponível.")
    st.caption(
        f"Tipo: {item.get('event_type', '-')} | Tema: {item.get('related_theme', '-')} | "
        f"Setor: {item.get('sector') or '-'} | Empresa: {item.get('company') or '-'} | "
        f"Data: {_format_date(item.get('published_at'))} | Fonte: {item.get('source')} | Score: {score}"
    )
    url = item.get("url")
    if url:
        st.markdown(f"[🔗 Abrir notícia original]({url})")


def _is_tavily_credits_error(exc: Exception) -> bool:
    """ is tavily credits error.

    Args:
        exc: Descrição do parâmetro `exc`.

    Returns:
        Valor de retorno da função.
    """
    error_msg = str(exc).lower()
    tavily_credit_terms = ["credit", "credits", "insufficient", "quota", "429", "rate limit", "too many requests"]
    return "tavily" in error_msg and any(term in error_msg for term in tavily_credit_terms)


def render_market_intelligence_tab():
    """Render market intelligence tab.

    Returns:
        Valor de retorno da função.
    """
    st.title("📈 Market Intelligence")
    st.caption("Notícias relevantes do mercado brasileiro ranqueadas por impacto e recência.")
    tracer = get_tracer()

    col_period, col_event, col_sector, col_company, col_source = st.columns([1.1, 1, 1, 1, 1])
    with col_period:
        period_label = st.selectbox("Período", list(TIME_RANGE_OPTIONS.keys()), index=1)
    with col_event:
        selected_event_type = st.selectbox("Tipo de evento", ["Todos"] + EVENT_TYPES)
    with col_sector:
        selected_sector = st.selectbox("Setor *", ["Selecione..."] + list(SECTOR_COMPANIES.keys()))
    with col_company:
        sector_companies = SECTOR_COMPANIES.get(selected_sector, [])
        selected_company = st.selectbox("Empresa", ["Todas"] + sector_companies, disabled=not sector_companies)
    with col_source:
        selected_source = st.text_input("Fonte", value="Todas")

    days = TIME_RANGE_OPTIONS[period_label]
    cache_key = f"market_intelligence_{days}_{selected_sector}"

    if selected_sector == "Selecione...":
        st.warning("Selecione um setor para carregar as notícias.")
        return

    if st.button("▶️ Atualizar notícias"):
        market_run_id = tracer.start_run(
            name=f"market_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            run_type="chain",
            inputs={"days": days, "sector": selected_sector},
            tags=["market_intelligence", "streamlit"],
            metadata={"feature": "market_intelligence"},
        )
        with st.spinner("Atualizando Market Intelligence... Isso pode demorar uns mintuos."):
            try:
                result = fetch_market_intelligence(days=days, sector=selected_sector, include_api_metrics=True)
                st.session_state[cache_key] = result

                for api_call in result.get("api_calls", []):
                    tracer.log_child_run(
                        market_run_id,
                        name="market_intelligence_classify_and_summarize",
                        run_type="llm",
                        inputs={
                            "prompt": api_call.get("prompt", {}),
                            "days": days,
                            "sector": selected_sector,
                        },
                        outputs={
                            "output": api_call.get("output"),
                            "status": "success",
                        },
                        metadata=api_call,
                        tags=["market_intelligence", api_call.get("step", "classify_and_summarize")],
                    )

                tracer.end_run(
                    market_run_id,
                    status="success",
                    outputs={
                        "status": "success",
                        "news_count": len(result.get("ranked_news", [])),
                    },
                )
            except Exception as exc:
                tracer.end_run(
                    market_run_id,
                    status="error",
                    error=str(exc),
                    outputs={"status": "error"},
                )
                if _is_tavily_credits_error(exc):
                    st.error("Créditos da Tavily acabaram. Atualize sua chave ou aguarde a renovação para continuar.")
                else:
                    st.error(f"Não foi possível atualizar o Market Intelligence no momento. Tente novamente. {exc}")
                return

    data = st.session_state.get(cache_key)
    if not data:
        st.info("Defina os filtros e clique em 'Atualizar notícias' para carregar os dados.")
        return

    ranked_news = data.get("ranked_news", [])
    source_options = ["Todas"] + sorted({n.get("source") for n in ranked_news if n.get("source")})
    if selected_source not in source_options:
        selected_source = "Todas"

    filtered_news = _filter_news(
        ranked_news,
        selected_event_type=selected_event_type,
        selected_sector=selected_sector,
        selected_company=selected_company,
        selected_source=selected_source,
    )

    st.subheader("Notícias ranqueadas")
    if not filtered_news:
        st.info("Nenhuma notícia encontrada para os filtros selecionados.")
        return

    for item in filtered_news[:35]:
        with st.container(border=True):
            _render_news_card(item)
