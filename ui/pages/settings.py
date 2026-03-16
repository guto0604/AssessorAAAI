import json
import os
from datetime import datetime
import streamlit as st
from core.openai_client import SESSION_OPENAI_KEY, get_effective_openai_api_key
from core.tavily_client import SESSION_TAVILY_KEY, get_effective_tavily_api_key
from ui.state import (
    SESSION_LANGSMITH_KEY,
    SESSION_LANGSMITH_TRACING_ENABLED,
    SESSION_RAG_CROSS_ENCODER_ENABLED,
    SESSION_RAG_SEMANTIC_WEIGHT,
    SESSION_RAG_TOP_K,
    SESSION_RAG_TOP_N,
    SESSION_TRACING_HEALTH_STATUS,
    _iso_now,
    get_tracer,
)
from ui.rag_service_provider import get_rag_service
def render_settings_tab():
    st.title("Configurações")
    st.caption("Preencha apenas as credenciais essenciais da sessão (quando necessário).")

    env_openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    env_langsmith_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    env_tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    current_openai_key = (st.session_state.get(SESSION_OPENAI_KEY, "") or "").strip()
    current_langsmith_key = (st.session_state.get(SESSION_LANGSMITH_KEY, "") or "").strip()
    current_tavily_key = (st.session_state.get(SESSION_TAVILY_KEY, "") or "").strip()

    effective_openai_key = (get_effective_openai_api_key() or "").strip()
    effective_langsmith_key = env_langsmith_key or current_langsmith_key
    effective_tavily_key = (get_effective_tavily_api_key() or "").strip()

    st.subheader("Integrações")

    openai_is_configured = bool(effective_openai_key)
    langsmith_is_configured = bool(effective_langsmith_key)
    tavily_is_configured = bool(effective_tavily_key)

    tracing_health_status = st.session_state.get(SESSION_TRACING_HEALTH_STATUS)
    tracing_is_ok = tracing_health_status == "ok"

    st.markdown("### Status")
    st.code(
        json.dumps(
            {
                "openai_api_key_configurada": openai_is_configured,
                "langsmith_api_key_configurada": langsmith_is_configured,
                "tavily_api_key_configurada": tavily_is_configured,
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

    tavily_api_key_input = st.text_input(
        "TAVILY_API_KEY (sessão)",
        value="" if env_tavily_key else current_tavily_key,
        type="password",
        key="settings_tavily_api_key_input",
        disabled=bool(env_tavily_key),
    )

    if st.button("💾 Salvar configurações", key="settings_save_keys"):
        if not env_openai_key:
            st.session_state[SESSION_OPENAI_KEY] = openai_api_key_input.strip()
        if not env_langsmith_key:
            st.session_state[SESSION_LANGSMITH_KEY] = langsmith_api_key_input.strip()
        if not env_tavily_key:
            st.session_state[SESSION_TAVILY_KEY] = tavily_api_key_input.strip()
        st.session_state[SESSION_LANGSMITH_TRACING_ENABLED] = True
        st.session_state[SESSION_TRACING_HEALTH_STATUS] = None
        st.success("Configurações salvas na sessão.")

    st.divider()
    st.subheader("Busca híbrida (RAG)")
    st.caption("Ajuste top_k e o peso da busca semântica (RRF + BM25).")

    current_top_k = int(st.session_state.get(SESSION_RAG_TOP_K, 5) or 5)
    current_semantic_weight = float(st.session_state.get(SESSION_RAG_SEMANTIC_WEIGHT, 0.8) or 0.8)
    current_cross_encoder_enabled = bool(st.session_state.get(SESSION_RAG_CROSS_ENCODER_ENABLED, False))
    current_top_n = int(st.session_state.get(SESSION_RAG_TOP_N, 3) or 3)

    rag_top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=20,
        value=current_top_k,
        step=1,
        key="settings_rag_top_k_input",
    )
    rag_semantic_weight = st.slider(
        "Peso busca semântica",
        min_value=0.0,
        max_value=1.0,
        value=current_semantic_weight,
        step=0.05,
        key="settings_rag_semantic_weight_input",
    )
    rag_bm25_weight = 1.0 - rag_semantic_weight

    rag_cross_encoder_enabled = st.checkbox(
        "Ativar rerank com Cross-Encoder local",
        value=current_cross_encoder_enabled,
        key="settings_rag_cross_encoder_enabled_input",
        help="Quando ativado, aplica rerank local nos candidatos do top_k e retorna o top_n mais relevantes.",
    )

    max_top_n = int(rag_top_k)
    rag_top_n = st.number_input(
        "Top N (após rerank)",
        min_value=1,
        max_value=max_top_n,
        value=min(current_top_n, max_top_n),
        step=1,
        key="settings_rag_top_n_input",
        disabled=not rag_cross_encoder_enabled,
    )
    if rag_cross_encoder_enabled:
        st.caption("Com Cross-Encoder ativo: recupera top_k e aplica rerank local para selecionar top_n.")
    else:
        st.caption("Cross-Encoder inativo: usa somente busca híbrida padrão com top_k.")
    st.caption(
        f"Peso BM25: {rag_bm25_weight:.2f} (calculado automaticamente como 1 - peso semântico)."
    )

    if st.button("💾 Salvar configuração do RAG", key="settings_save_rag_config"):
        st.session_state[SESSION_RAG_TOP_K] = int(rag_top_k)
        st.session_state[SESSION_RAG_SEMANTIC_WEIGHT] = float(rag_semantic_weight)
        st.session_state[SESSION_RAG_CROSS_ENCODER_ENABLED] = bool(rag_cross_encoder_enabled)
        st.session_state[SESSION_RAG_TOP_N] = int(min(rag_top_n, rag_top_k))
        st.success("Configuração de busca híbrida salva na sessão.")

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
    st.caption(
        "Reconstrói o índice vetorial FAISS (HNSW) usando todos os arquivos PDF/TXT da pasta knowledge_base."
    )

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
