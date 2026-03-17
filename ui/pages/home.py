import streamlit as st


def render_home_tab():
    """Render home tab.

    Returns:
        Valor de retorno da função.
    """
    st.title("Potencializando assessores com IA")

    st.subheader("🎯 Objetivo da ferramenta")
    st.markdown(
        """
        - Aumentar a qualidade e consistência dos contatos com clientes.
        - Reduzir tempo operacional para montar abordagens e materiais.
        - Transformar dados internos em ações comerciais práticas.
        """
    )

    st.subheader("📚 O que você encontra em cada aba")
    st.markdown(
        """
        - **👤 Visualização clientes**: painel executivo com KPIs, alertas, composição de carteira, aderência de risco e oportunidades comerciais.
        - **🚀 Voz do Assessor (Pitch)**: fluxo de IA guiado para estruturar argumentos, tom e narrativa de contato.
        - **📝 Reuniões**: transcrição e resumo de reuniões, com direcionamento de próximos passos e acompanhamento das interações com o cliente.
        - **📊 Talk to your Data**: pergunte e converse em linguagem natural para explorar e visualizar dados.
        - **🤖 Pergunte à IA**: canal para tirar dúvidas de processos, políticas e documentos internos usando IA.
        - **⚙️ Configurações**: gerenciamento de credenciais e status das integrações da sessão.
        """
    )

    st.info(
        "Para as abas de Visualização Clientes, Voz do Assessor e Reuniões, selecione o cliente na barra lateral antes de iniciar para garantir o contexto correto."
    )
