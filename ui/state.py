import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.data_loader import load_clientes
from core.langsmith_tracing import LangSmithTracer
from core.openai_client import SESSION_OPENAI_KEY
from core.tavily_client import SESSION_TAVILY_KEY

SESSION_LANGSMITH_KEY = "user_langsmith_api_key"
SESSION_LANGSMITH_TRACING_ENABLED = "user_langsmith_tracing_enabled"
SESSION_PITCH_TRACE = "pitch_trace_run"
SESSION_MEETING_TRACE = "meeting_trace_run"
SESSION_PITCH_FLOW_STARTED = "pitch_flow_started"
SESSION_TRACING_HEALTH_STATUS = "langsmith_tracing_health_status"
SESSION_LANGSMITH_TRACER = "langsmith_tracer_instance"
SESSION_RAG_TOP_K = "rag_top_k"
SESSION_RAG_SEMANTIC_WEIGHT = "rag_semantic_weight"
SESSION_RAG_CROSS_ENCODER_ENABLED = "rag_cross_encoder_enabled"
SESSION_RAG_TOP_N = "rag_top_n"
TALK_TO_DATA_TEMPLATE_DEFAULT_OPTION = "Quero escrever minha própria pergunta!"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_tracer() -> LangSmithTracer:
    env_langsmith_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    session_langsmith_key = (st.session_state.get(SESSION_LANGSMITH_KEY, "") or "").strip()
    effective_langsmith_key = env_langsmith_key or session_langsmith_key

    cached_tracer = st.session_state.get(SESSION_LANGSMITH_TRACER)
    if isinstance(cached_tracer, LangSmithTracer):
        if cached_tracer.api_key == effective_langsmith_key:
            return cached_tracer

    tracer = LangSmithTracer(
        api_key=effective_langsmith_key,
        enabled=True,
    )
    st.session_state[SESSION_LANGSMITH_TRACER] = tracer
    return tracer


def _format_cliente_value(campo: str, valor):
    campos_monetarios = {
        "Patrimonio_Investido_Conosco",
        "Patrimonio_Investido_Outros",
        "Dinheiro_Disponivel_Para_Investir",
    }
    campos_percentuais = {"Rentabilidade_12_meses", "CDI_12_Meses"}

    if valor is None:
        return "-"

    if campo in campos_monetarios and isinstance(valor, (int, float)):
        valor_formatado = f"R$ {valor:,.2f}"
        return valor_formatado.replace(",", "X").replace(".", ",").replace("X", ".")

    if campo in campos_percentuais and isinstance(valor, (int, float)):
        return f"{valor*100:.2f}%"

    return str(valor)


def build_cliente_sidebar_table(cliente_info: dict) -> pd.DataFrame:
    labels = {
        "Cliente_ID": "ID",
        "Nome": "Nome",
        "Patrimonio_Investido_Conosco": "Patrimônio investido conosco",
        "Patrimonio_Investido_Outros": "Patrimônio investido em outras instituições",
        "Dinheiro_Disponivel_Para_Investir": "Dinheiro disponível para investir",
        "Perfil_Suitability": "Perfil de suitability",
        "Rentabilidade_12_meses": "Rentabilidade (12 meses)",
        "CDI_12_Meses": "CDI (12 meses)",
    }

    dados_formatados = [
        {
            "Campo": labels.get(campo, campo),
            "Valor": _format_cliente_value(campo, valor),
        }
        for campo, valor in cliente_info.items()
    ]
    return pd.DataFrame(dados_formatados)


def init_session_state():
    if "etapa" not in st.session_state:
        st.session_state.etapa = 1

    if "ranking_resultado" not in st.session_state:
        st.session_state.ranking_resultado = None

    if "selected_cliente_id" not in st.session_state:
        clientes_df = load_clientes()
        st.session_state.selected_cliente_id = clientes_df["Cliente_ID"].iloc[0]

    if SESSION_OPENAI_KEY not in st.session_state:
        st.session_state[SESSION_OPENAI_KEY] = ""

    if SESSION_LANGSMITH_KEY not in st.session_state:
        st.session_state[SESSION_LANGSMITH_KEY] = ""

    if SESSION_TAVILY_KEY not in st.session_state:
        st.session_state[SESSION_TAVILY_KEY] = ""

    if SESSION_LANGSMITH_TRACING_ENABLED not in st.session_state:
        st.session_state[SESSION_LANGSMITH_TRACING_ENABLED] = True

    if SESSION_PITCH_TRACE not in st.session_state:
        st.session_state[SESSION_PITCH_TRACE] = None

    if SESSION_MEETING_TRACE not in st.session_state:
        st.session_state[SESSION_MEETING_TRACE] = None

    if SESSION_PITCH_FLOW_STARTED not in st.session_state:
        st.session_state[SESSION_PITCH_FLOW_STARTED] = False

    if SESSION_TRACING_HEALTH_STATUS not in st.session_state:
        st.session_state[SESSION_TRACING_HEALTH_STATUS] = None

    if SESSION_LANGSMITH_TRACER not in st.session_state:
        st.session_state[SESSION_LANGSMITH_TRACER] = None

    if SESSION_RAG_TOP_K not in st.session_state:
        st.session_state[SESSION_RAG_TOP_K] = 5

    if SESSION_RAG_SEMANTIC_WEIGHT not in st.session_state:
        st.session_state[SESSION_RAG_SEMANTIC_WEIGHT] = 0.8

    if SESSION_RAG_CROSS_ENCODER_ENABLED not in st.session_state:
        st.session_state[SESSION_RAG_CROSS_ENCODER_ENABLED] = False

    if SESSION_RAG_TOP_N not in st.session_state:
        st.session_state[SESSION_RAG_TOP_N] = 3

    if "talk_to_data_last_llm_output" not in st.session_state:
        st.session_state.talk_to_data_last_llm_output = None

    if "talk_to_data_generated_sql" not in st.session_state:
        st.session_state.talk_to_data_generated_sql = ""

    if "talk_to_data_saved_sql" not in st.session_state:
        st.session_state.talk_to_data_saved_sql = ""

    if "talk_to_data_can_generate" not in st.session_state:
        st.session_state.talk_to_data_can_generate = True

    if "talk_to_data_template_dropdown" not in st.session_state:
        st.session_state.talk_to_data_template_dropdown = TALK_TO_DATA_TEMPLATE_DEFAULT_OPTION
