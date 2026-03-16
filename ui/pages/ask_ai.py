from pathlib import Path

import streamlit as st
from datetime import date, datetime

from rag.config import KNOWLEDGE_BASE_DIR, SUPPORTED_EXTENSIONS
from rag.document_loader import InvalidDocumentError
from ui.guardrails import (
    evaluate_input_guardrails,
    guardrail_warning_message,
    handle_guardrail_exception,
)
from ui.rag_service_provider import get_rag_service
from ui.state import get_tracer




def _list_kb_folders() -> list[str]:
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    folders = [p.relative_to(KNOWLEDGE_BASE_DIR).as_posix() for p in KNOWLEDGE_BASE_DIR.iterdir() if p.is_dir()]
    return sorted(folders)


def render_ask_ai_tab():
    st.title("🤖 Pergunte à IA")
    st.caption("Faça perguntas sobre a knowledge base e adicione novos documentos para indexação vetorial.")

    rag = get_rag_service()
    tracer = get_tracer()

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

    if st.button("⬆️ Processar e indexar uploads", key="ask_ai_upload_button"):
        if not uploaded_files:
            st.warning("Selecione ao menos um arquivo para upload.")
        elif not selected_folder:
            st.error("Informe uma pasta de destino válida.")
        else:
            added_files = 0
            added_chunks = 0
            for file in uploaded_files:
                suffix = Path(file.name).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    st.error(f"{file.name}: formato inválido. Apenas PDF e TXT.")
                    continue

                try:
                    result = rag.ingest_uploaded_file(selected_folder, file.name, file.getvalue())
                    added_files += result.added_files
                    added_chunks += result.added_chunks
                    st.success(f"{file.name}: indexado com sucesso.")
                except InvalidDocumentError as exc:
                    st.error(f"{file.name}: {exc}")
                except Exception as exc:
                    st.error(f"{file.name}: falha ao processar arquivo ({exc}).")

            if added_files:
                st.info(f"Indexação concluída: {added_files} arquivo(s), {added_chunks} chunk(s).")

    st.divider()

    st.subheader("❓ Pergunta")
    question = st.text_area("Digite sua pergunta", placeholder="Ex.: Qual foi o resultado das empresas do setor de proteína animal?\nQuais as regras de rebalanceamento?\nComo me preparar para uma reunião com o cliente?")

    if st.button("🚀 Enviar pergunta", key="ask_ai_send_question"):
        question = (question or "").strip()
        if not question:
            st.warning("Digite uma pergunta antes de enviar.")
        else:
            ask_ai_run_id = tracer.start_run(
                name=f"ask_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                run_type="chain",
                inputs={"question": question, "top_k": 4},
                tags=["ask_ai", "streamlit", "rag"],
                metadata={
                    "feature": "ask_ai",
                },
            )

            try:
                guardrail_result = evaluate_input_guardrails(question)
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
                st.warning(guardrail_warning_message(guardrail_result.violation_type))
                return

            with st.spinner("Consultando base vetorial..."):
                try:
                    rag_result = rag.answer_question(question, top_k=4, include_api_metrics=True)
                    answer = rag_result["answer"]
                    sources = rag_result["sources"]

                    for api_call in rag_result.get("api_calls", []):
                        step = api_call.get("step")
                        run_type = "llm" if step in {"chat_completion", "query_parser"} else "embedding"
                        tracer.log_child_run(
                            ask_ai_run_id,
                            name=f"ask_ai_{step}",
                            run_type=run_type,
                            inputs={"question": question, "top_k": 4},
                            outputs={"status": "success"},
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
                                f"- `{source['source_path']}` (chunk {source['chunk_id']}, score={source['score']:.3f})"
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
