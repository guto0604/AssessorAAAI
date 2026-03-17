import streamlit as st

from core.data_loader import get_cliente_by_id, load_clientes
from ui.pages.ask_ai import render_ask_ai_tab
from ui.pages.home import render_home_tab
from ui.pages.client_visualization import render_visualizacao_clientes_tab
from ui.pages.meetings import render_meetings_tab
from ui.pages.pitch import render_pitch_tab
from ui.pages.settings import render_settings_tab
from ui.pages.market_intelligence import render_market_intelligence_tab
from ui.pages.talk_to_data import (
    run_duckdb_query,
    sanitize_duckdb_sql,
    validate_read_only_sql,
    render_talk_to_your_data_page,
)
from ui.state import build_cliente_sidebar_table, init_session_state

st.set_page_config(page_title="POC Jornada Comercial", layout="wide")


def main():
    """Responsável por executar uma etapa do fluxo da aplicação de assessoria.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    init_session_state()

    st.title("AssessorAAAI")

    st.sidebar.header("Selecionar Cliente")
    clientes_df = load_clientes()

    cliente_ids = clientes_df["Cliente_ID"].tolist()
    selected_index = 0
    if st.session_state.selected_cliente_id in cliente_ids:
        selected_index = int(cliente_ids.index(st.session_state.selected_cliente_id))

    selected_cliente_id = st.sidebar.selectbox(
        "Cliente",
        cliente_ids,
        index=selected_index,
        key="global_cliente_select",
    )
    st.session_state.selected_cliente_id = selected_cliente_id

    cliente_info = get_cliente_by_id(st.session_state.selected_cliente_id)

    st.sidebar.markdown("### Dados do Cliente")
    dados_cliente_df = build_cliente_sidebar_table(cliente_info)
    st.sidebar.table(dados_cliente_df)

    tab_home, tab_clientes, tab_pitch, tab_meetings, tab_portfolio, tab_ask_ai, tab_settings = st.tabs([
        "🏠 Início",
        "👤 Visualização clientes",
        "🚀 Voz do Assessor (Pitch)",
        "📝 Reuniões",
        "📊 Talk to your Data",
        "🤖 Pergunte à IA",
        "⚙️ Configurações",
    ])

    with tab_home:
        render_home_tab()

    with tab_clientes:
        render_visualizacao_clientes_tab(st.session_state.selected_cliente_id)

    with tab_pitch:
        render_pitch_tab(st.session_state.selected_cliente_id, cliente_info)

    with tab_meetings:
        render_meetings_tab(st.session_state.selected_cliente_id, cliente_info)

    with tab_portfolio:
        render_talk_to_your_data_page()

    #with tab_market:
    #    render_market_intelligence_tab()

    with tab_ask_ai:
        render_ask_ai_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
