import streamlit as st

from core.data_loader import get_cliente_by_id, load_clientes
from ui.feedback import render_screen_feedback
from ui.pages.ask_ai import render_ask_ai_tab
from ui.pages.investment_case_builder import render_investment_case_builder_tab
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
from ui.state import (
    RBAC_AVAILABLE_TABS,
    RLS_SEGMENT_OPTIONS,
    SESSION_RBAC_ENABLED_TABS,
    SESSION_RLS_ALLOWED_SEGMENTS,
    build_cliente_sidebar_table,
    init_session_state,
)

st.set_page_config(page_title="POC Jornada Comercial", layout="wide")


def _segment_label(patrimonio):
    if patrimonio is None:
        return None
    if patrimonio <= 300_000:
        return "Até 300k"
    if patrimonio <= 2_000_000:
        return "300k-2M"
    return "2M+"


def _filter_clientes_by_rls(clientes_df):
    allowed_segments = st.session_state.get(SESSION_RLS_ALLOWED_SEGMENTS, RLS_SEGMENT_OPTIONS)
    if not allowed_segments:
        return clientes_df.iloc[0:0]

    segmented = clientes_df.copy()
    segmented["_segmento_rls"] = segmented["Patrimonio_Investido_Conosco"].apply(_segment_label)
    return segmented[segmented["_segmento_rls"].isin(allowed_segments)].drop(columns=["_segmento_rls"])


def _is_tab_enabled(tab_label: str) -> bool:
    enabled_tabs = st.session_state.get(SESSION_RBAC_ENABLED_TABS, RBAC_AVAILABLE_TABS)
    return tab_label in enabled_tabs


def main():
    """Responsável por executar uma etapa do fluxo da aplicação de assessoria.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    init_session_state()

    st.title("AssessorAAAI")

    st.sidebar.header("Selecionar Cliente")
    clientes_df = _filter_clientes_by_rls(load_clientes())

    if clientes_df.empty:
        st.sidebar.warning("Nenhum cliente disponível para os segmentos RLS selecionados.")
        st.info("Ajuste os segmentos na aba de Configurações para visualizar clientes.")
        return

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

    tab_home, tab_clientes, tab_pitch, tab_meetings, tab_portfolio, tab_case_builder, tab_ask_ai, tab_settings = st.tabs([
        "🏠 Início",
        "👤 Visualização clientes",
        "🚀 Voz do Assessor (Pitch)",
        "📝 Reuniões",
        "📊 Talk to your Data",
        "🧠 Investment Case Builder",
        "🤖 Pergunte à IA",
        "⚙️ Configurações",
    ])

    with tab_home:
        if _is_tab_enabled("🏠 Início"):
            render_home_tab()
        else:
            st.info("Tela desabilitada pelo perfil RBAC")

    with tab_clientes:
        if _is_tab_enabled("👤 Visualização clientes"):
            render_visualizacao_clientes_tab(st.session_state.selected_cliente_id)
        else:
            st.info("Tela desabilitada pelo perfil RBAC")

    with tab_pitch:
        if _is_tab_enabled("🚀 Voz do Assessor (Pitch)"):
            render_pitch_tab(st.session_state.selected_cliente_id, cliente_info)
            render_screen_feedback("pitch", "🚀 Voz do Assessor (Pitch)")
        else:
            st.info("Tela desabilitada pelo perfil RBAC")

    with tab_meetings:
        if _is_tab_enabled("📝 Reuniões"):
            render_meetings_tab(st.session_state.selected_cliente_id, cliente_info)
            render_screen_feedback("meetings", "📝 Reuniões")
        else:
            st.info("Tela desabilitada pelo perfil RBAC")

    with tab_portfolio:
        if _is_tab_enabled("📊 Talk to your Data"):
            render_talk_to_your_data_page()
            render_screen_feedback("talk_to_data", "📊 Talk to your Data")
        else:
            st.info("Tela desabilitada pelo perfil RBAC")

    with tab_case_builder:
        if _is_tab_enabled("🧠 Investment Case Builder"):
            render_investment_case_builder_tab(st.session_state.selected_cliente_id, cliente_info)
        else:
            st.info("Tela desabilitada pelo perfil RBAC")

    with tab_ask_ai:
        if _is_tab_enabled("🤖 Pergunte à IA"):
            render_ask_ai_tab()
            render_screen_feedback("ask_ai", "🤖 Pergunte à IA")
        else:
            st.info("Tela desabilitada pelo perfil RBAC")

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
