import streamlit as st

from rag.pipeline import RagService


@st.cache_resource
def get_rag_service() -> RagService:
    """Obtém a informação necessária para a etapa atual do processo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    return RagService()
