from pathlib import Path

import streamlit as st
from datetime import date, datetime

from rag.config import KNOWLEDGE_BASE_DIR, RAG_SEGMENT_OPTIONS, SUPPORTED_EXTENSIONS
from rag.document_loader import InvalidDocumentError
from ui.guardrails import (
    evaluate_input_guardrails,
    guardrail_warning_message,
    handle_guardrail_exception,
)
from ui.rag_service_provider import get_rag_service
from ui.state import (
    SESSION_RAG_SEMANTIC_WEIGHT,
    SESSION_RAG_TOP_K,
    SESSION_RLS_ALLOWED_SEGMENTS,
    get_tracer,
)




def _list_kb_folders() -> list[str]:
    """Responsável por listar kb folders no contexto da aplicação de assessoria.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    folders = [p.relative_to(KNOWLEDGE_BASE_DIR).as_posix() for p in KNOWLEDGE_BASE_DIR.iterdir() if p.is_dir()]
    return sorted(folders)


def render_ask_ai_tab():
    """Renderiza a seção da interface correspondente a este fluxo da aplicação.

    Returns:
        Não retorna valor; atualiza diretamente os componentes da interface.
    """
    st.title("🤖 Pergunte à IA")
    st.caption("Faça perguntas sobre a knowledge base e adicione novos documentos para indexação vetorial.")

    rag = get_rag_service()
    tracer = get_tracer()
    top_k = int(st.session_state.get(SESSION_RAG_TOP_K, 5) or 5)
    semantic_weight = float(st.session_state.get(SESSION_RAG_SEMANTIC_WEIGHT, 0.8) or 0.8)
    bm25_weight = 1.0 - semantic_weight

    st.subheader("📥 Upload para knowledge base")
    folders = _list_kb_folders()
    default_folder = folders[0] if folders else "geral"

    col1, col2 = st.columns([2, 1])
    with col1:
        folder_options = folders + ["+ Nova pasta"]
        folder_choice = st.selectbox("Pasta de destino", options=folder_options, index=0)
    with col2:
        new_folder_name = st.text_input("Nova pasta", value="", placeholder="ex: normativos")

    selected_folder = new_folder_name.strip() if folder_choice == "+ Nova pasta" else folder_choice
    if not selected_folder:
        selected_folder = default_folder

    uploaded_files = st.file_uploader(
        "Envie arquivos PDF ou TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="ask_ai_upload_files",
    )
    selected_segments = st.multiselect(
        "Perfis/RLS com acesso ao documento",
        options=RAG_SEGMENT_OPTIONS,
        default=RAG_SEGMENT_OPTIONS,
        help="Por padrão, novos documentos ficam visíveis para os 3 perfis. Você pode restringir no cadastro.",
        key="ask_ai_upload_segments",
    )
    selected_document_date = st.date_input(
        "Data de referência do documento",
        value=date.today(),
        help="Usada para filtrar documentos durante a consulta.",
        key="ask_ai_upload_document_date",
    )

    if st.button("⬆️ Processar e indexar uploads", key="ask_ai_upload_button"):
        if not uploaded_files:
            st.warning("Selecione ao menos um arquivo para upload.")
        elif not selected_folder:
            st.error("Informe uma pasta de destino válida.")
        elif not selected_segments:
            st.error("Selecione ao menos um perfil para o documento.")
        else:
            added_files = 0
            added_chunks = 0
            for file in uploaded_files:
                suffix = Path(file.name).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    st.error(f"{file.name}: formato inválido. Apenas PDF e TXT.")
                    continue

                try:
                    result = rag.ingest_uploaded_file(
                        selected_folder,
                        file.name,
                        file.getvalue(),
                        allowed_segments=selected_segments,
                        document_date=selected_document_date,
                    )
                    added_files += result.added_files
                    added_chunks += result.added_chunks
                    st.success(f"{file.name}: indexado com sucesso.")
                except InvalidDocumentError as exc:
                    st.error(f"{file.name}: {exc}")
                except Exception as exc:
                    st.error(f"{file.name}: falha ao processar arquivo ({exc}).")

            if added_files:
                st.info(f"Indexação concluída: {added_files} arquivo(s), {added_chunks} chunk(s).")

    #if st.button("🪄 Aplicar metadados padrão onde faltarem", key="ask_ai_backfill_metadata_button"):
    #    with st.spinner("Aplicando metadados padrão nos chunks/documentos sem segmento/data..."):
    #        try:
    #            result = rag.apply_default_metadata_to_all_missing(
    #                default_segments=RAG_SEGMENT_OPTIONS,
    #                default_date=date.today(),
    #            )
    #            st.success(
    #                "Metadados padronizados com sucesso. "
    #                f"Documentos atualizados: {result.updated_documents}. "
    #                f"Chunks atualizados: {result.updated_chunks}. "
    #                f"Data padrão aplicada: {result.default_date_applied}."
    #            )
    #        except Exception as exc:
    #            st.error(f"Falha ao aplicar metadados padrão: {exc}")

    st.divider()

    st.subheader("❓ Pergunta")
    allowed_segments = st.session_state.get(SESSION_RLS_ALLOWED_SEGMENTS, RAG_SEGMENT_OPTIONS)
    #st.caption(
    #    "A consulta respeita os perfis liberados na simulação de RLS: "
    #    + ", ".join(allowed_segments if allowed_segments else RAG_SEGMENT_OPTIONS)
    #)
    enable_date_filter = st.checkbox(
        "Filtrar documentos por data de referência",
        value=False,
        key="ask_ai_enable_date_filter",
    )
    date_filter_start = None
    date_filter_end = None
    date_cols = st.columns(2)
    with date_cols[0]:
        date_filter_start = st.date_input(
            "Data inicial",
            value=date.today(),
            disabled=not enable_date_filter,
            key="ask_ai_date_filter_start",
        )
    with date_cols[1]:
        date_filter_end = st.date_input(
            "Data final",
            value=date.today(),
            disabled=not enable_date_filter,
            key="ask_ai_date_filter_end",
        )
    question = st.text_area("Digite sua pergunta", placeholder="Ex.: Qual foi o resultado das empresas do setor de proteína animal?\nQuais as regras de rebalanceamento?\nComo me preparar para uma reunião com o cliente?")

    if st.button("🚀 Enviar pergunta", key="ask_ai_send_question"):
        question = (question or "").strip()
        if not question:
            st.warning("Digite uma pergunta antes de enviar.")
        elif enable_date_filter and date_filter_start > date_filter_end:
            st.warning("A data inicial não pode ser maior que a data final.")
        else:
            ask_ai_run_id = tracer.start_run(
                name=f"ask_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                run_type="chain",
                inputs={
                    "question": question,
                    "top_k": top_k,
                    "semantic_weight": semantic_weight,
                    "bm25_weight": bm25_weight,
                    "allowed_segments": allowed_segments,
                    "document_date_start": date_filter_start.isoformat() if enable_date_filter else None,
                    "document_date_end": date_filter_end.isoformat() if enable_date_filter else None,
                },
                tags=["ask_ai", "streamlit", "rag"],
                metadata={
                    "feature": "ask_ai",
                },
            )

            try:
                guardrail_result = evaluate_input_guardrails(question, context="ask_ai")
            except Exception as exc:
                guardrail_result = handle_guardrail_exception(question, exc)

            tracer.log_event(
                ask_ai_run_id,
                "input_guardrail_checked",
                {
                    "blocked": guardrail_result.blocked,
                    "violation_type": guardrail_result.violation_type,
                    "reason": guardrail_result.message,
                    "model": guardrail_result.model,
                    "input_tokens": guardrail_result.input_tokens,
                    "output_tokens": guardrail_result.output_tokens,
                    "total_tokens": guardrail_result.total_tokens,
                },
            )

            if guardrail_result.blocked:
                tracer.end_run(
                    ask_ai_run_id,
                    status="blocked",
                    outputs={
                        "status": "blocked",
                        "guardrail": {
                            "violation_type": guardrail_result.violation_type,
                            "reason": guardrail_result.message,
                        },
                    },
                )
                st.warning(guardrail_warning_message(guardrail_result.violation_type, context="ask_ai"))
                return

            with st.spinner("Consultando base vetorial..."):
                try:
                    rag_result = rag.answer_question(
                        question,
                        top_k=top_k,
                        semantic_weight=semantic_weight,
                        bm25_weight=bm25_weight,
                        allowed_segments=allowed_segments,
                        start_date=date_filter_start if enable_date_filter else None,
                        end_date=date_filter_end if enable_date_filter else None,
                        include_api_metrics=True,
                    )
                    answer = rag_result["answer"]
                    sources = rag_result["sources"]
                    tracer.log_event(
                        ask_ai_run_id,
                        "ask_ai_documents_consulted",
                        {
                            "documents": [source.get("source_path") for source in sources],
                            "chunks": [
                                {
                                    "source_path": source.get("source_path"),
                                    "chunk_id": source.get("chunk_id"),
                                }
                                for source in sources
                            ],
                            "total_documents": len({source.get("source_path") for source in sources}),
                            "total_chunks": len(sources),
                        },
                    )

                    for api_call in rag_result.get("api_calls", []):
                        step = api_call.get("step")
                        run_type = "llm" if step in {"chat_completion", "query_parser"} else "embedding"
                        tracer.log_child_run(
                            ask_ai_run_id,
                            name=f"ask_ai_{step}",
                            run_type=run_type,
                            inputs={
                                "question": question,
                                "top_k": top_k,
                                "semantic_weight": semantic_weight,
                                "bm25_weight": bm25_weight,
                                "prompt": api_call.get("prompt", {}),
                            },
                            outputs={
                                "status": "success",
                                "output": api_call.get("output"),
                            },
                            metadata=api_call,
                            tags=["ask_ai", step or "unknown"],
                        )

                    tracer.end_run(
                        ask_ai_run_id,
                        status="success",
                        outputs={
                            "status": "success",
                            "sources_count": len(sources),
                            "api_calls": rag_result.get("api_calls", []),
                        },
                    )

                    st.markdown("### Resposta")
                    st.write(answer)

                    st.markdown("### Fontes utilizadas")
                    if sources:
                        for source in sources:
                            st.markdown(
                                f"- `{source['source_path']}` "
                                f"(chunk {source['chunk_id']}, score={source['score']:.5f}, "
                                f"data={source.get('document_date')}, "
                            )
                    else:
                        st.info("Nenhuma fonte encontrada.")
                except Exception as exc:
                    tracer.end_run(
                        ask_ai_run_id,
                        status="error",
                        error=str(exc),
                        outputs={"status": "error"},
                    )
                    st.error(f"Falha ao responder pergunta: {exc}")
