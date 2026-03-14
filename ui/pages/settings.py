import json
import os
from datetime import datetime
import streamlit as st
from core.openai_client import SESSION_OPENAI_KEY, get_effective_openai_api_key
from core.exa_client import SESSION_EXA_KEY, get_effective_exa_api_key
from ui.state import get_tracer, _iso_now, SESSION_LANGSMITH_KEY, SESSION_LANGSMITH_TRACING_ENABLED, SESSION_TRACING_HEALTH_STATUS
from ui.rag_service_provider import get_rag_service
def render_settings_tab():
    st.title("Configurações")
    st.caption("Preencha apenas as credenciais essenciais da sessão (quando necessário).")

    env_openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    env_langsmith_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    env_exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    current_openai_key = (st.session_state.get(SESSION_OPENAI_KEY, "") or "").strip()
    current_langsmith_key = (st.session_state.get(SESSION_LANGSMITH_KEY, "") or "").strip()
    current_exa_key = (st.session_state.get(SESSION_EXA_KEY, "") or "").strip()

    effective_openai_key = (get_effective_openai_api_key() or "").strip()
    effective_langsmith_key = env_langsmith_key or current_langsmith_key
    effective_exa_key = (get_effective_exa_api_key() or "").strip()

    st.subheader("Integrações")

    openai_is_configured = bool(effective_openai_key)
    langsmith_is_configured = bool(effective_langsmith_key)
    exa_is_configured = bool(effective_exa_key)

    tracing_health_status = st.session_state.get(SESSION_TRACING_HEALTH_STATUS)
    tracing_is_ok = tracing_health_status == "ok"

    st.markdown("### Status")
    st.code(
        json.dumps(
            {
                "openai_api_key_configurada": openai_is_configured,
                "langsmith_api_key_configurada": langsmith_is_configured,
                "exa_api_key_configurada": exa_is_configured,
                "tracing_langsmith_ok": tracing_is_ok,
            },
            ensure_ascii=False,
            indent=2,
        ),
        language="json",
    )

    openai_api_key_input = st.text_input(
        "OPENAI_API_KEY (sessão)",
        value="" if env_openai_key else current_openai_key,
        type="password",
        key="settings_openai_api_key_input",
        disabled=bool(env_openai_key),
    )

    langsmith_api_key_input = st.text_input(
        "LANGSMITH_API_KEY (sessão)",
        value="" if env_langsmith_key else current_langsmith_key,
        type="password",
        key="settings_langsmith_api_key_input",
        disabled=bool(env_langsmith_key),
    )

    exa_api_key_input = st.text_input(
        "EXA_API_KEY (sessão)",
        value="" if env_exa_key else current_exa_key,
        type="password",
        key="settings_exa_api_key_input",
        disabled=bool(env_exa_key),
    )

    if st.button("💾 Salvar configurações", key="settings_save_keys"):
        if not env_openai_key:
            st.session_state[SESSION_OPENAI_KEY] = openai_api_key_input.strip()
        if not env_langsmith_key:
            st.session_state[SESSION_LANGSMITH_KEY] = langsmith_api_key_input.strip()
        if not env_exa_key:
            st.session_state[SESSION_EXA_KEY] = exa_api_key_input.strip()
        st.session_state[SESSION_LANGSMITH_TRACING_ENABLED] = True
        st.session_state[SESSION_TRACING_HEALTH_STATUS] = None
        st.success("Configurações salvas na sessão.")

    if st.button("🩺 Testar tracing LangSmith", key="settings_test_tracing"):
        tracer = get_tracer()
        if not tracer.enabled:
            st.session_state[SESSION_TRACING_HEALTH_STATUS] = "error"
            st.error("Tracing inativo: informe a LANGSMITH_API_KEY para validar.")
        else:
            healthcheck_run_id = tracer.start_run(
                name=f"settings_healthcheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                run_type="tool",
                inputs={"source": "settings_tab_healthcheck"},
                tags=["settings", "healthcheck"],
                metadata={"checked_at": _iso_now()},
            )

            if healthcheck_run_id:
                sent_ok = tracer.end_run(
                    healthcheck_run_id,
                    status="success",
                    outputs={"status": "ok", "message": "healthcheck_passed"},
                )
                if sent_ok:
                    st.session_state[SESSION_TRACING_HEALTH_STATUS] = "ok"
                    st.success("Tracing validado com sucesso no LangSmith.")
                else:
                    st.session_state[SESSION_TRACING_HEALTH_STATUS] = "error"
                    st.error(
                        "Run de teste criada, mas não foi enviada ao LangSmith. "
                        f"{tracer.last_error or ''}".strip()
                    )
            else:
                st.session_state[SESSION_TRACING_HEALTH_STATUS] = "error"
                st.error("Falha ao criar run de teste no LangSmith.")



    st.divider()
    st.subheader("Knowledge Base Vetorial")
    st.caption("Reconstrói o índice FAISS local usando todos os arquivos PDF/TXT da pasta knowledge_base.")

    if st.button("🔁 Reindexar base vetorial", key="settings_reindex_vectorstore"):
        with st.spinner("Reindexando documentos da knowledge base..."):
            try:
                rag = get_rag_service()
                result = rag.reindex_all_documents()
                st.success(
                    f"Reindexação concluída: {result.added_files} arquivo(s), {result.added_chunks} chunk(s)."
                )
                if result.skipped_files:
                    st.warning("Arquivos ignorados: " + ", ".join(result.skipped_files))
            except Exception as exc:
                st.error(f"Falha na reindexação: {exc}")
