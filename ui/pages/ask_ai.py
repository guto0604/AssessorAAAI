import streamlit as st

from rag_pipeline import (
    DuplicateFileError,
    EmptyFileError,
    InvalidFileError,
    RagPipelineError,
    VectorKnowledgeBase,
)


@st.cache_resource(show_spinner=False)
def get_rag_kb() -> VectorKnowledgeBase:
    return VectorKnowledgeBase()


def render_ask_ai_tab():
    st.title("🤖 Pergunte à IA")
    st.caption("Faça perguntas com RAG sobre a knowledge base vetorial local.")

    kb = get_rag_kb()

    col_sync, col_info = st.columns([1, 2])
    with col_sync:
        if st.button("🔄 Sincronizar documentos existentes", use_container_width=True):
            with st.spinner("Indexando documentos já existentes da knowledge base..."):
                indexed, errors = kb.sync_existing_documents()
            if indexed:
                st.success(f"Sincronização concluída. {indexed} chunks indexados.")
            else:
                st.info("Nenhum novo documento foi indexado.")
            for error in errors:
                st.warning(error)
    with col_info:
        st.info("Suporta upload de PDF/TXT com persistência local em knowledge_base + índice vetorial.")

    st.subheader("📤 Upload para a knowledge base")
    target_folder = st.text_input(
        "Pasta de destino dentro de knowledge_base",
        value="uploads",
        help="Exemplo: politicas, research/2026, produtos/novos",
    )
    uploaded_files = st.file_uploader(
        "Selecione arquivos PDF ou TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("📥 Enviar e indexar arquivos", use_container_width=True):
        if not uploaded_files:
            st.warning("Selecione pelo menos um arquivo para upload.")
        else:
            for uploaded_file in uploaded_files:
                try:
                    destination, chunk_count = kb.save_upload_and_index(
                        file_name=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                        target_folder=target_folder,
                    )
                    st.success(f"{destination.name} indexado com sucesso ({chunk_count} chunks).")
                except (InvalidFileError, EmptyFileError, DuplicateFileError) as exc:
                    st.error(f"{uploaded_file.name}: {exc}")
                except Exception as exc:
                    st.error(f"Erro inesperado ao indexar {uploaded_file.name}: {exc}")

    st.divider()
    st.subheader("❓ Faça sua pergunta")
    question = st.text_area("Pergunta", placeholder="Ex.: Qual é o procedimento para consultar liquidez de um produto?")

    if st.button("💬 Perguntar à IA", use_container_width=True):
        try:
            with st.spinner("Buscando trechos relevantes e gerando resposta..."):
                result = kb.ask(question)

            st.markdown("### Resposta")
            st.write(result["answer"])

            st.markdown("### Fontes utilizadas")
            for source in result["sources"]:
                st.markdown(f"- **{source['source']}** (distância: {source['distance']:.4f})")
                st.caption(source["snippet"])
        except RagPipelineError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Erro ao consultar a base vetorial: {exc}")
