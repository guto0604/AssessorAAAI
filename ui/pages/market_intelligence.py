from datetime import datetime

import streamlit as st

from core.market_intelligence import EVENT_TYPES, SECTOR_COMPANIES, fetch_market_intelligence

TIME_RANGE_OPTIONS = {
    "24h": 1,
    "7 dias": 7,
    "30 dias": 30,
}


def _format_date(date_str: str) -> str:
    if not date_str:
        return "-"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return date_str


def _filter_news(items: list[dict], selected_event_type: str, selected_sector: str, selected_company: str, selected_source: str) -> list[dict]:
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


def render_market_intelligence_tab():
    st.title("🧭 Market Intelligence")
    st.caption("Monitoramento executivo de eventos relevantes do mercado brasileiro por impacto e recência.")

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
        st.warning("Selecione um setor para habilitar a execução do EXA.")
        return

    if st.button("▶️ Executar Market Intelligence"):
        with st.spinner("Coletando sinais com EXA e estruturando com OpenAI..."):
            try:
                st.session_state[cache_key] = fetch_market_intelligence(days=days, sector=selected_sector)
            except Exception as exc:
                st.error(f"Erro ao atualizar Market Intelligence: {exc}")
                return

    data = st.session_state.get(cache_key)
    if not data:
        st.info("Defina os filtros e clique em 'Executar Market Intelligence' para carregar os dados.")
        return

    radar_events = data.get("radar_events", [])

    source_options = ["Todas"] + sorted({n.get("source") for n in radar_events if n.get("source")})
    if selected_source not in source_options:
        selected_source = "Todas"

    radar_filtered = _filter_news(
        radar_events,
        selected_event_type=selected_event_type,
        selected_sector=selected_sector,
        selected_company=selected_company,
        selected_source=selected_source,
    )

    st.subheader("Radar de Eventos")
    if not radar_filtered:
        st.info("Nenhum evento encontrado para os filtros selecionados.")
    else:
        for event in radar_filtered[:25]:
            with st.container(border=True):
                _render_news_card(event)

    st.divider()
    st.subheader("Acompanhamento por Setor")

    sectors = data.get("sectors", [])
    sectors = [s for s in sectors if s.get("sector") == selected_sector]

    if not sectors:
        st.info("Nenhum setor disponível para os filtros selecionados.")
        return

    for sector_block in sectors:
        with st.container(border=True):
            st.markdown(f"### {sector_block.get('sector')}")
            st.caption("Empresas monitoradas: " + ", ".join(sector_block.get("companies", [])))
            st.write(sector_block.get("summary") or "Sem resumo consolidado.")

            sector_news = _filter_news(
                sector_block.get("news", []),
                selected_event_type=selected_event_type,
                selected_sector=selected_sector,
                selected_company=selected_company,
                selected_source=selected_source,
            )

            if not sector_news:
                st.info("Sem notícias relevantes neste setor para os filtros aplicados.")
                continue

            for item in sector_news[:10]:
                st.markdown("---")
                _render_news_card(item)
