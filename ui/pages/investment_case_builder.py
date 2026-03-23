from __future__ import annotations

from pathlib import Path

import plotly.io as pio
import streamlit as st

from core.investment_case_builder.orchestrator import InvestmentCaseOrchestrator
from ui.markdown_utils import escape_streamlit_markdown

SESSION_CASE_STATE = "investment_case_builder_case_state"
SESSION_CHAT_HISTORY = "investment_case_builder_chat_history"
SESSION_LAST_CLIENT_ID = "investment_case_builder_last_client_id"
SESSION_RERUN_STEP = "investment_case_builder_rerun_step"
SESSION_INPUT_PROMPT = "investment_case_builder_prompt"
SESSION_INPUT_NOTES = "investment_case_builder_notes"
SESSION_INPUT_TONE = "investment_case_builder_tone"

TONE_OPTIONS = [
    "Consultivo",
    "Executivo",
    "Defensivo",
    "Crescimento",
    "Liquidez",
]


def init_investment_case_session_state(selected_client_id: str) -> None:
    if SESSION_CASE_STATE not in st.session_state:
        st.session_state[SESSION_CASE_STATE] = None
    if SESSION_CHAT_HISTORY not in st.session_state:
        st.session_state[SESSION_CHAT_HISTORY] = []
    if SESSION_RERUN_STEP not in st.session_state:
        st.session_state[SESSION_RERUN_STEP] = "data_relevance"
    if SESSION_INPUT_PROMPT not in st.session_state:
        st.session_state[SESSION_INPUT_PROMPT] = ""
    if SESSION_INPUT_NOTES not in st.session_state:
        st.session_state[SESSION_INPUT_NOTES] = ""
    if SESSION_INPUT_TONE not in st.session_state:
        st.session_state[SESSION_INPUT_TONE] = TONE_OPTIONS[0]

    previous_client = st.session_state.get(SESSION_LAST_CLIENT_ID)
    if previous_client != selected_client_id:
        st.session_state[SESSION_CASE_STATE] = None
        st.session_state[SESSION_CHAT_HISTORY] = []
        st.session_state[SESSION_LAST_CLIENT_ID] = selected_client_id


def _render_workflow(case_state: dict | None) -> None:
    st.subheader("Bloco 2 — Workflow")
    if not case_state:
        st.info("Nenhum workflow executado ainda.")
        return

    for step, metadata in case_state.get("workflow_status", {}).items():
        title = f"{metadata.get('label', step)} — {metadata.get('status', 'pending').upper()}"
        with st.expander(title, expanded=metadata.get("status") in {"running", "error"}):
            st.write(f"**Início:** {metadata.get('started_at') or '-'}")
            st.write(f"**Fim:** {metadata.get('finished_at') or '-'}")
            if metadata.get("details"):
                st.write(metadata["details"])
            if metadata.get("error"):
                st.error(metadata["error"])


def _render_context_decisions(case_state: dict | None) -> None:
    st.subheader("Bloco 3 — Decisões de contexto")
    if not case_state or not case_state.get("data_relevance_decisions"):
        st.info("As decisões de contexto aparecerão aqui após o Data Relevance Agent.")
        return

    data = case_state["data_relevance_decisions"]
    st.markdown("**Justificativa resumida**")
    st.json(data.get("selection_rationale", {}))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dados selecionados**")
        st.json(data.get("selected_context", {}))
    with col2:
        st.markdown("**Dados ignorados**")
        st.json(data.get("ignored_data", {}))

    st.markdown("**Conflitos detectados**")
    if data.get("conflicts_detected"):
        st.json(data.get("conflicts_detected"))
    else:
        st.caption("Nenhum conflito relevante detectado.")

    st.markdown("**Decisões de prioridade/sobreposição**")
    if data.get("priority_decisions"):
        st.json(data.get("priority_decisions"))
    else:
        st.caption("Nenhuma sobreposição aplicada no case atual.")


def _render_diagnosis(case_state: dict | None) -> None:
    st.subheader("Bloco 4 — Diagnóstico")
    diagnosis = (case_state or {}).get("portfolio_diagnosis", {})
    if not diagnosis:
        st.info("O diagnóstico aparecerá aqui após a execução do workflow.")
        return
    st.write(diagnosis.get("executive_summary", ""))
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Pontos fortes**")
        for item in diagnosis.get("strengths", []):
            st.write(f"- {item}")
    with cols[1]:
        st.markdown("**Oportunidades**")
        for item in diagnosis.get("opportunities", []):
            st.write(f"- {item}")
    with cols[2]:
        st.markdown("**Alertas**")
        for item in diagnosis.get("attention_points", []):
            st.write(f"- {item}")


def _render_scenarios(case_state: dict | None) -> None:
    st.subheader("Bloco 5 — Cenários")
    scenarios = (case_state or {}).get("scenarios", [])
    if not scenarios:
        st.info("Os cenários gerados serão exibidos aqui.")
        return
    columns = st.columns(len(scenarios))
    for column, scenario in zip(columns, scenarios):
        with column:
            st.markdown(f"### {scenario.get('name')}")
            st.write(scenario.get("rationale"))
            st.markdown("**Vantagens**")
            for item in scenario.get("advantages", []):
                st.write(f"- {item}")
            st.markdown("**Riscos / trade-offs**")
            for item in (scenario.get("risks", []) + scenario.get("trade_offs", [])):
                st.write(f"- {item}")
            st.caption(f"Liquidez aproximada: {scenario.get('approx_liquidity', '-')}")
            st.caption(f"Aderência ao perfil: {scenario.get('profile_fit', '-')}")


def _render_risks(case_state: dict | None) -> None:
    st.subheader("Bloco 6 — Riscos e suitability")
    risk_review = (case_state or {}).get("risk_review", {})
    if not risk_review:
        st.info("A revisão de riscos aparecerá aqui.")
        return
    st.write(f"**Status geral:** {risk_review.get('overall_status', '-')}")
    st.markdown("**Alertas**")
    for alert in risk_review.get("alerts", []):
        severity = alert.get("severity", "info").upper()
        st.write(f"- [{severity}] {alert.get('scenario', '-')}: {alert.get('message', '')}")
    st.markdown("**Limitações**")
    for item in risk_review.get("limitations", []):
        st.write(f"- {item}")
    st.markdown("**Revisão humana sugerida**")
    for item in risk_review.get("human_review_items", []):
        st.write(f"- {item}")


def _render_proposal(case_state: dict | None) -> None:
    st.subheader("Bloco 7 — Proposta e próximos passos")
    proposal = (case_state or {}).get("proposal", {})
    if not proposal:
        st.info("A proposta consultiva aparecerá aqui.")
        return
    st.markdown("**Resumo executivo**")
    st.write(escape_streamlit_markdown(proposal.get("executive_summary", "")))
    st.markdown("**Proposta central**")
    st.json(proposal.get("central_proposal", {}))
    st.markdown("**Próximos passos**")
    for item in proposal.get("next_steps", []):
        st.write(f"- {item}")
    st.markdown("**Perguntas sugeridas para a reunião**")
    for item in proposal.get("meeting_questions", []):
        st.write(f"- {item}")


def _render_visualizations(case_state: dict | None) -> None:
    st.subheader("Bloco 8 — Gráficos")
    charts = (case_state or {}).get("visualizations", [])
    if not charts:
        st.info("Os gráficos aparecerão aqui após a etapa de visualização.")
        return
    for chart in charts:
        st.markdown(f"**{chart.get('title', 'Gráfico')}**")
        st.caption(f"Fonte: {chart.get('source', '-')}")
        if chart.get("type") == "fallback":
            st.info(chart.get("message", "Sem dados suficientes para este gráfico."))
            continue
        figure_json = chart.get("figure_json")
        if figure_json:
            st.plotly_chart(pio.from_json(figure_json), use_container_width=True)


def _render_pdf(case_state: dict | None, orchestrator: InvestmentCaseOrchestrator) -> None:
    st.subheader("Bloco 9 — PDF")
    if not case_state:
        st.info("Execute o workflow para gerar o PDF final.")
        return

    if st.button("Gerar / atualizar PDF", key="investment_case_refresh_pdf"):
        case_state = orchestrator.run_full_workflow(case_state, start_from="pdf_builder")
        st.session_state[SESSION_CASE_STATE] = case_state
        st.success("PDF atualizado a partir do estado consolidado.")

    pdf_path = case_state.get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        st.success(f"PDF gerado: {pdf_path}")
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                "Baixar PDF",
                data=pdf_file.read(),
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                key="investment_case_download_pdf",
            )
    else:
        st.warning("PDF ainda não gerado ou indisponível.")


def _render_chat(case_state: dict | None, orchestrator: InvestmentCaseOrchestrator) -> None:
    st.subheader("Bloco 10 — Chat consultivo final")
    if not case_state or not case_state.get("proposal"):
        st.info("Gere um case para habilitar o chat consultivo contextual.")
        return

    for message in st.session_state[SESSION_CHAT_HISTORY]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Pergunte sobre o case construído", key="investment_case_chat_input")
    if question:
        st.session_state[SESSION_CHAT_HISTORY].append({"role": "user", "content": question})
        answer = orchestrator.answer_chat(case_state=case_state, question=question)
        st.session_state[SESSION_CHAT_HISTORY].append({"role": "assistant", "content": answer})
        st.rerun()


def render_investment_case_builder_tab(selected_client_id: str, cliente_info: dict) -> None:
    init_investment_case_session_state(selected_client_id)
    orchestrator = InvestmentCaseOrchestrator()
    case_state = st.session_state[SESSION_CASE_STATE]

    st.title("Investment Case Builder")
    st.caption("Arquitetura multiagente com contexto automático do cliente selecionado, rastreabilidade por etapa e artefatos finais.")

    st.subheader("Bloco 1 — Entrada")
    st.write(f"**Cliente atual selecionado:** {cliente_info.get('Nome', '-') } ({selected_client_id})")
    st.text_area(
        "Objetivo / instrução principal do caso",
        key=SESSION_INPUT_PROMPT,
        height=140,
        placeholder="Ex.: construir um case para reduzir concentração, preservar liquidez e preparar uma reunião consultiva.",
    )
    st.text_area(
        "Observações adicionais (opcional)",
        key=SESSION_INPUT_NOTES,
        height=90,
    )
    st.selectbox(
        "Tom ou foco opcional",
        options=TONE_OPTIONS,
        key=SESSION_INPUT_TONE,
    )

    col1, col2, col3 = st.columns([1.3, 1, 1])
    with col1:
        if st.button("Construir Investment Case", key="investment_case_build_button", use_container_width=True):
            prompt = st.session_state[SESSION_INPUT_PROMPT].strip()
            if not prompt:
                st.warning("Informe o objetivo principal antes de iniciar o workflow.")
            else:
                case_state = orchestrator.initialize_case(
                    client_id=selected_client_id,
                    client_name=cliente_info.get("Nome", "Cliente"),
                    advisor_prompt=prompt,
                    additional_notes=st.session_state[SESSION_INPUT_NOTES].strip(),
                    tone_focus=st.session_state[SESSION_INPUT_TONE],
                )
                case_state = orchestrator.run_full_workflow(case_state)
                st.session_state[SESSION_CASE_STATE] = case_state
                st.session_state[SESSION_CHAT_HISTORY] = []
                st.success("Investment case construído com sucesso.")
    with col2:
        st.selectbox(
            "Reexecutar a partir de",
            options=list((case_state or {}).get("workflow_status", {}).keys() or ["data_relevance"]),
            key=SESSION_RERUN_STEP,
        )
    with col3:
        rerun_disabled = not bool(case_state)
        if st.button("Reexecutar workflow parcial", key="investment_case_partial_rerun", disabled=rerun_disabled, use_container_width=True) and case_state:
            case_state = orchestrator.run_full_workflow(case_state, start_from=st.session_state[SESSION_RERUN_STEP])
            st.session_state[SESSION_CASE_STATE] = case_state
            st.success("Workflow parcial reexecutado.")

    case_state = st.session_state[SESSION_CASE_STATE]

    _render_workflow(case_state)
    _render_context_decisions(case_state)
    _render_diagnosis(case_state)
    _render_scenarios(case_state)
    _render_risks(case_state)
    _render_proposal(case_state)
    _render_visualizations(case_state)
    _render_pdf(case_state, orchestrator)
    _render_chat(case_state, orchestrator)
