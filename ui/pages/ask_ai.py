import streamlit as st
def render_ask_ai_tab():
    st.title("🤖 Pergunte à IA")
    st.caption("Visão funcional da próxima etapa (placeholder — implementação ainda não iniciada).")

    st.markdown(
        """
        Esta aba será dedicada a perguntas operacionais e de negócio com respostas baseadas
        em **documentos internos**, políticas e materiais de apoio.
        """
    )

    st.subheader("🔍 Como funcionará")
    st.markdown(
        """
        1. O usuário faz uma pergunta em linguagem natural.
        2. Um pipeline de **RAG** recupera trechos relevantes da base de conhecimento.
        3. A IA responde com contexto operacional e referência do conteúdo encontrado.
        """
    )

    st.subheader("📚 Base de conhecimento (planejado)")
    st.markdown(
        """
        - Upload de novos documentos diretamente pela interface.
        - Indexação dos arquivos com **modelo de embeddings**.
        - Atualização incremental da base para ampliar cobertura de dúvidas.
        """
    )

    st.warning("Tela adicionada como planejamento funcional. O fluxo completo ainda não foi desenvolvido.")


