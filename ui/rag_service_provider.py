import streamlit as st

from rag.pipeline import RagService


@st.cache_resource
def get_rag_service() -> RagService:
    """Get rag service.

    Returns:
        Valor de retorno da função.
    """
    return RagService()
