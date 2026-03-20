import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.data_loader import load_clientes
from core.langsmith_tracing import LangSmithTracer
from core.openai_client import SESSION_OPENAI_KEY
from rag.config import RAG_SEGMENT_OPTIONS
from core.tavily_client import SESSION_TAVILY_KEY

SESSION_LANGSMITH_KEY = "user_langsmith_api_key"
SESSION_LANGSMITH_TRACING_ENABLED = "user_langsmith_tracing_enabled"
SESSION_PITCH_TRACE = "pitch_trace_run"
SESSION_MEETING_TRACE = "meeting_trace_run"
SESSION_PITCH_FLOW_STARTED = "pitch_flow_started"
SESSION_PITCH_MODE = "pitch_mode"
SESSION_TRACING_HEALTH_STATUS = "langsmith_tracing_health_status"
SESSION_LANGSMITH_TRACER = "langsmith_tracer_instance"
SESSION_SCREEN_RUN_REGISTRY = "screen_run_registry"
SESSION_SCREEN_FEEDBACK_REGISTRY = "screen_feedback_registry"
SESSION_RAG_TOP_K = "rag_top_k"
SESSION_RAG_SEMANTIC_WEIGHT = "rag_semantic_weight"
SESSION_RLS_ALLOWED_SEGMENTS = "rls_allowed_segments"
SESSION_RBAC_ENABLED_TABS = "rbac_enabled_tabs"
TALK_TO_DATA_TEMPLATE_DEFAULT_OPTION = "Quero escrever minha própria pergunta!"

RLS_SEGMENT_OPTIONS = RAG_SEGMENT_OPTIONS.copy()

RBAC_AVAILABLE_TABS = [
    "🏠 Início",
    "👤 Visualização clientes",
    "🚀 Voz do Assessor (Pitch)",
    "📝 Reuniões",
    "📊 Talk to your Data",
    "🤖 Pergunte à IA",
]


def _iso_now() -> str:
    """Retorna o timestamp atual em formato padronizado para registros e rastreabilidade.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    """
    return datetime.now(timezone.utc).isoformat()


def get_tracer() -> LangSmithTracer:
    """Obtém a informação necessária para a etapa atual do processo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
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


def register_screen_run(screen_key: str, run_id: str | None, *, status: str) -> None:
    """Armazena o último run conhecido de uma tela para uso do feedback."""
    if not run_id:
        return

    registry = st.session_state.get(SESSION_SCREEN_RUN_REGISTRY, {}).copy()
    current = registry.get(screen_key, {})
    registry[screen_key] = {
        **current,
        "run_id": run_id,
        "status": status,
        "updated_at": _iso_now(),
    }
    st.session_state[SESSION_SCREEN_RUN_REGISTRY] = registry


def get_screen_run(screen_key: str) -> dict | None:
    """Obtém o último run registrado para a tela informada."""
    return (st.session_state.get(SESSION_SCREEN_RUN_REGISTRY, {}) or {}).get(screen_key)


def register_screen_feedback(
    screen_key: str,
    run_id: str,
    *,
    feedback_id: str,
    score: bool,
) -> None:
    """Salva o feedback enviado para o último run da tela."""
    registry = st.session_state.get(SESSION_SCREEN_FEEDBACK_REGISTRY, {}).copy()
    registry[screen_key] = {
        "run_id": run_id,
        "feedback_id": feedback_id,
        "score": score,
        "updated_at": _iso_now(),
    }
    st.session_state[SESSION_SCREEN_FEEDBACK_REGISTRY] = registry


def get_screen_feedback(screen_key: str, run_id: str | None) -> dict | None:
    """Retorna o feedback salvo em sessão se ele pertencer ao run atual da tela."""
    if not run_id:
        return None

    feedback = (st.session_state.get(SESSION_SCREEN_FEEDBACK_REGISTRY, {}) or {}).get(screen_key)
    if feedback and feedback.get("run_id") == run_id:
        return feedback
    return None


def _format_cliente_value(campo: str, valor):
    """Transforma informações do cliente para uso direto nas telas e decisões do fluxo.

    Args:
        campo: Valor de entrada necessário para processar 'campo'.
        valor: Valor de entrada necessário para processar 'valor'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
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
    """Monta a estrutura de dados usada nas próximas etapas do fluxo.

    Args:
        cliente_info: Dicionário com os dados consolidados do cliente para personalizar a resposta.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    """
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
    """Responsável por processar session state no contexto da aplicação de assessoria.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
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

    if SESSION_PITCH_MODE not in st.session_state:
        st.session_state[SESSION_PITCH_MODE] = "auto_pitch"

    if SESSION_TRACING_HEALTH_STATUS not in st.session_state:
        st.session_state[SESSION_TRACING_HEALTH_STATUS] = None

    if SESSION_LANGSMITH_TRACER not in st.session_state:
        st.session_state[SESSION_LANGSMITH_TRACER] = None

    if SESSION_SCREEN_RUN_REGISTRY not in st.session_state:
        st.session_state[SESSION_SCREEN_RUN_REGISTRY] = {}

    if SESSION_SCREEN_FEEDBACK_REGISTRY not in st.session_state:
        st.session_state[SESSION_SCREEN_FEEDBACK_REGISTRY] = {}

    if SESSION_RAG_TOP_K not in st.session_state:
        st.session_state[SESSION_RAG_TOP_K] = 5

    if SESSION_RAG_SEMANTIC_WEIGHT not in st.session_state:
        st.session_state[SESSION_RAG_SEMANTIC_WEIGHT] = 0.8

    if SESSION_RLS_ALLOWED_SEGMENTS not in st.session_state:
        st.session_state[SESSION_RLS_ALLOWED_SEGMENTS] = RLS_SEGMENT_OPTIONS.copy()

    if SESSION_RBAC_ENABLED_TABS not in st.session_state:
        st.session_state[SESSION_RBAC_ENABLED_TABS] = RBAC_AVAILABLE_TABS.copy()

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
