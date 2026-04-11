import streamlit as st


def render_home_tab():
    """Renderiza a seção da interface correspondente a este fluxo da aplicação.

    Returns:
        Não retorna valor; atualiza diretamente os componentes da interface.
    """
    st.title("🏠 Início")
    st.markdown("Você está aqui 😄")

    st.markdown("\n👤 **Visualização Clientes (NON AI)**")
    st.markdown(
        "Painel executivo com patrimônio, risco, liquidez, aderência e oportunidades comerciais do cliente."
    )

    st.markdown("\n🚀 **Voz do Assessor**")
    st.markdown(
        "Crie pitches personalizados com IA, mantendo controle humano sobre tese, tom e mensagem final."
    )

    st.markdown("\n📝 **Reuniões**")
    st.markdown(
        "Grave ou envie áudio de reuniões para transcrever, resumir e registrar próximos passos por cliente."
    )

    st.markdown("\n📊 **Talk to your Data**")
    st.markdown(
        "Faça perguntas em linguagem natural e obtenha as respostas baseado em dados."
    )

    st.markdown("\n🤖 **Pergunte à IA**")
    st.markdown(
        "Consulte políticas, procedimentos internos e muito mais com respostas baseadas em nossas fontes."
    )

    st.markdown("\n⚙️ **Configurações (Focado no Admin)**")
    st.markdown(
        "Gerencie chaves, tracing, parâmetros de busca, reindexação vetorial e permissões."
    )
