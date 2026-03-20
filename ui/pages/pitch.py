from datetime import datetime

import streamlit as st

from core.auto_pitch import (
    AUTO_PITCH_COMMUNICATION_MODEL,
    AUTO_PITCH_PRIORITY_MODEL,
    generate_auto_pitch_communication,
    generate_auto_pitch_priorities,
)
from core.data_loader import (
    carteira_summary_for_llm,
    get_jornada_config,
    get_investimentos_by_cliente,
    load_jornadas,
    load_produtos,
)
from core.journey_ranker import rank_journeys
from core.langsmith_tracing import LangSmithTracer
from core.pitch_structurer import build_pitch_options_step5
from core.pitch_writer import generate_final_pitch_step7, generate_prompt_to_pitch, revise_pitch_step8
from core.source_selector import select_sources_step4
from ui.guardrails import (
    evaluate_input_guardrails,
    guardrail_warning_message,
    handle_guardrail_exception,
)
from ui.markdown_utils import escape_streamlit_markdown
from ui.state import (
    SESSION_PITCH_FLOW_STARTED,
    SESSION_PITCH_MODE,
    SESSION_PITCH_TRACE,
    _iso_now,
    get_tracer,
    register_screen_run,
)

PITCH_MODE_GUIDED = "guided"
PITCH_MODE_AUTO_PITCH = "auto_pitch"
PITCH_MODE_PROMPT_TO_PITCH = "prompt_to_pitch"


def _set_pitch_trace_state(run_id: str | None, status: str, *, timestamp_field: str) -> None:
    """Atualiza o estado do tracing do pitch na sessão e no registro de feedback."""
    st.session_state[SESSION_PITCH_TRACE] = {
        "run_id": run_id,
        "status": status,
        timestamp_field: _iso_now(),
    }
    register_screen_run("pitch", run_id, status=status)



def _start_pitch_trace(
    tracer: LangSmithTracer,
    cliente_id,
    prompt_assessor: str,
    mode: str = PITCH_MODE_GUIDED,
) -> str | None:
    """Executa uma etapa de construção do pitch comercial personalizado para o cliente.

    Args:
        tracer: Valor de entrada necessário para processar 'tracer'.
        cliente_id: Identificador único do cliente usado para filtrar dados e arquivos relacionados.
        prompt_assessor: Valor de entrada necessário para processar 'prompt_assessor'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    active_trace = st.session_state.get(SESSION_PITCH_TRACE)
    if active_trace and active_trace.get("run_id") and active_trace.get("status") == "in_progress":
        tracer.log_event(active_trace["run_id"], "pitch_interrupted", {
            "reason": "Novo fluxo iniciado antes da finalização",
            "at": _iso_now(),
        })
        tracer.end_run(active_trace["run_id"], status="interrupted", outputs={"status": "interrupted"})

    if mode == PITCH_MODE_PROMPT_TO_PITCH:
        run_prefix = "prompt_to_pitch_cliente"
    elif mode == PITCH_MODE_AUTO_PITCH:
        run_prefix = "auto_pitch_cliente"
    else:
        run_prefix = "pitch_cliente"

    run_tags = ["pitch", "streamlit"]
    if mode == PITCH_MODE_PROMPT_TO_PITCH:
        run_tags.append("prompt-to-pitch")
    if mode == PITCH_MODE_AUTO_PITCH:
        run_tags.append("auto-pitch")

    run_id = tracer.start_run(
        name=f"{run_prefix}_{cliente_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_type="chain",
        inputs={
            "cliente_id": cliente_id,
            "prompt_assessor": prompt_assessor,
            "mode": mode,
        },
        tags=run_tags,
        metadata={
            "started_at": _iso_now(),
            "cliente_id": cliente_id,
            "prompt_preview": prompt_assessor[:180],
            "mode": mode,
        },
    )
    _set_pitch_trace_state(run_id, "in_progress", timestamp_field="started_at")
    return run_id



def _reset_pitch_flow_state():
    """Executa uma etapa de construção do pitch comercial personalizado para o cliente.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    st.session_state.etapa = 1
    st.session_state.ranking_resultado = None

    keys_to_reset = [
        "editar_descricao",
        "jornada_selecionada",
        "step4_result",
        "step5_result",
        "step5_selection",
        "pitch_draft",
        "pitch_final_text",
        "pitch_version",
        "pitch_step5_release_index",
        "pitch_step5_confirmed_sections",
        "auto_pitch_priorities",
        "auto_pitch_selected_priority_id",
        "auto_pitch_selected_priority",
        "auto_pitch_signal_summary",
        "auto_pitch_communication_result",
        "auto_pitch_communication_revealed",
    ]
    for key in keys_to_reset:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if (
            key.startswith("pitch_chk_")
            or key.startswith("pitch_draft_box_")
        ):
            st.session_state.pop(key, None)


def _render_prompt_to_pitch_result():
    """Renderiza o resultado final do modo prompt-to-pitch."""
    if not st.session_state.get("pitch_final_text"):
        return

    st.divider()
    st.header("⚡ Prompt-to-pitch")
    st.text_area(
        "Texto final:",
        value=st.session_state["pitch_final_text"],
        height=240,
        key="pitch_prompt_to_pitch_final_box",
    )



def _render_auto_pitch_header():
    st.divider()
    st.header("🤖 Auto-pitch")
    st.caption("A IA prioriza as melhores abordagens do momento, o assessor só escolhe a tese vencedora.")


def _render_auto_pitch_result(*, render_header: bool = True):
    prioridades = st.session_state.get("auto_pitch_priorities") or []
    communication_result = st.session_state.get("auto_pitch_communication_result") or {}
    communication_revealed = st.session_state.get("auto_pitch_communication_revealed", False)

    if render_header:
        _render_auto_pitch_header()

    signal_summary = st.session_state.get("auto_pitch_signal_summary")
    if signal_summary:
        with st.expander("📊 Sinais usados na priorização", expanded=False):
            st.json(signal_summary)

    st.subheader("Top 3 prioridades sugeridas")
    priority_ids = []
    priority_index = {}
    for item in prioridades:
        priority_id = item.get("priority_id")
        if not priority_id:
            continue
        priority_ids.append(priority_id)
        priority_index[priority_id] = item

    for item in prioridades:
        st.markdown(
            f"""
**{item.get('priority_rank', '-')}. {item.get('titulo', 'Prioridade')}**  
Categoria: `{item.get('categoria', '-')}`  
**Objetivo:** {item.get('objetivo', '-')}  
**Por que agora:** {item.get('porque_agora', '-')}  
**Abordagem:** {item.get('abordagem_recomendada', '-')}  
**Canal/Tom:** {item.get('canal_recomendado', '-')} / {item.get('tom', '-')}
            """
        )
        sinais = item.get("sinais_dados") or []
        if sinais:
            st.caption("Sinais: " + " | ".join(sinais))
        if item.get("products_selected_ids"):
            st.caption("Produtos candidatos: " + ", ".join(item["products_selected_ids"]))
        if item.get("kb_files_selected"):
            st.caption("Base consultiva: " + ", ".join(item["kb_files_selected"]))
        st.divider()

    if priority_ids and st.session_state.get("auto_pitch_selected_priority_id") not in priority_ids:
        st.session_state["auto_pitch_selected_priority_id"] = priority_ids[0]

    if not priority_ids:
        st.session_state["auto_pitch_selected_priority"] = None
        return

    selected_priority_id = st.radio(
        "Qual prioridade o assessor quer seguir?",
        options=priority_ids,
        format_func=lambda pid: f"{priority_index[pid].get('priority_rank', '-')}. {priority_index[pid].get('titulo', pid)}",
        key="auto_pitch_selected_priority_id",
    )
    st.session_state["auto_pitch_selected_priority"] = priority_index.get(selected_priority_id)

    if communication_result:
        with st.expander("🎯 Racional argumentativo", expanded=False):
            if communication_result.get("resumo_estrategico"):
                st.write(escape_streamlit_markdown(communication_result["resumo_estrategico"]))
            if communication_result.get("racional_argumentativo"):
                st.markdown("**Racional:**")
                for item in communication_result["racional_argumentativo"]:
                    st.write(escape_streamlit_markdown(f"- {item}"))
            if communication_result.get("provas_evidencias"):
                st.markdown("**Provas e evidências:**")
                for item in communication_result["provas_evidencias"]:
                    st.write(escape_streamlit_markdown(f"- {item}"))
            if communication_result.get("observacoes_assessor"):
                st.markdown("**Observações para o assessor:**")
                for item in communication_result["observacoes_assessor"]:
                    st.write(escape_streamlit_markdown(f"- {item}"))

        if communication_revealed:
            st.subheader("💬 Comunicação sugerida")
            st.text_area(
                "Mensagem principal:",
                value=communication_result.get("mensagem_principal", ""),
                height=220,
                key="auto_pitch_message_box",
            )
            if communication_result.get("mensagem_follow_up"):
                st.text_area(
                    "Follow-up sugerido:",
                    value=communication_result.get("mensagem_follow_up", ""),
                    height=140,
                    key="auto_pitch_followup_box",
                )
            if communication_result.get("cta"):
                st.caption(f"CTA sugerido: {communication_result['cta']}")


def _finalize_auto_pitch_run(
    tracer: LangSmithTracer,
    pitch_run_id: str | None,
    *,
    pitch_mode: str,
    selected_priority: dict | None,
    communication_result: dict | None,
) -> None:
    """Finaliza o run do auto-pitch com um resumo serializável e enxuto."""
    tracer.log_event(
        pitch_run_id,
        "auto_pitch_run_finalized_by_user",
        {
            "priority_id": (selected_priority or {}).get("priority_id"),
            "message_chars": len((communication_result or {}).get("mensagem_principal", "")),
        },
    )
    tracer.end_run(
        pitch_run_id,
        status="completed",
        outputs={
            "status": "completed",
            "mode": pitch_mode,
            "selected_priority": {
                "priority_id": (selected_priority or {}).get("priority_id"),
                "priority_rank": (selected_priority or {}).get("priority_rank"),
                "categoria": (selected_priority or {}).get("categoria"),
                "titulo": (selected_priority or {}).get("titulo"),
            },
            "communication_summary": {
                "resumo_estrategico": (communication_result or {}).get("resumo_estrategico"),
                "message_chars": len((communication_result or {}).get("mensagem_principal", "")),
                "has_follow_up": bool((communication_result or {}).get("mensagem_follow_up")),
                "cta": (communication_result or {}).get("cta"),
            },
        },
    )
    _set_pitch_trace_state(pitch_run_id, "completed", timestamp_field="ended_at")

def render_pitch_tab(cliente_id, cliente_info):
    """Renderiza a seção da interface correspondente a este fluxo da aplicação.

    Args:
        cliente_id: Identificador único do cliente usado para filtrar dados e arquivos relacionados.
        cliente_info: Dicionário com os dados consolidados do cliente para personalizar a resposta.

    Returns:
        Não retorna valor; atualiza diretamente os componentes da interface.
    """
    st.header("🚀 Iniciar fluxo de pitch")

    pitch_mode = st.radio(
        "Modo de geração:",
        options=[PITCH_MODE_AUTO_PITCH, PITCH_MODE_GUIDED, PITCH_MODE_PROMPT_TO_PITCH],
        format_func=lambda mode: (
            "Fluxo guiado" if mode == PITCH_MODE_GUIDED else "Auto-pitch" if mode == PITCH_MODE_AUTO_PITCH else "Prompt-to-pitch"
        ),
        horizontal=True,
        key=SESSION_PITCH_MODE,
    )

    prompt_assessor = ""
    if pitch_mode != PITCH_MODE_AUTO_PITCH:
        prompt_assessor = st.text_area(
            "Escreva o objetivo do contato ou um contexto adicional:",
            height=150,
            key="pitch_prompt_assessor"
        )

    tracer = get_tracer()

    start_label = "▶️ Iniciar pitch" if not st.session_state.get(SESSION_PITCH_FLOW_STARTED) else "🔄 Iniciar novo pitch"
    if st.button(start_label, key="pitch_btn_start_new_flow"):
        _reset_pitch_flow_state()
        pitch_run_id = _start_pitch_trace(tracer, cliente_id, prompt_assessor, mode=pitch_mode)
        tracer.log_event(pitch_run_id, "pitch_flow_initialized", {
            "action": "start_new_flow",
            "at": _iso_now(),
            "prompt_chars": len(prompt_assessor.strip()),
            "mode": pitch_mode,
        })

        try:
            guardrail_result = evaluate_input_guardrails(prompt_assessor.strip(), context="pitch")
        except Exception as exc:
            guardrail_result = handle_guardrail_exception(prompt_assessor.strip(), exc)

        tracer.log_event(
            pitch_run_id,
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
                pitch_run_id,
                status="blocked",
                outputs={
                    "status": "blocked",
                    "guardrail": {
                        "violation_type": guardrail_result.violation_type,
                        "reason": guardrail_result.message,
                    },
                },
            )
            _set_pitch_trace_state(pitch_run_id, "blocked", timestamp_field="ended_at")
            st.warning(guardrail_warning_message(guardrail_result.violation_type, context="pitch"))
            return

        st.session_state[SESSION_PITCH_FLOW_STARTED] = True

        if pitch_mode == PITCH_MODE_PROMPT_TO_PITCH:
            tracer.log_event(pitch_run_id, "pitch_prompt_to_pitch_started", {"mode": pitch_mode})
            try:
                with st.spinner("Gerando pitch..."):
                    pitch_result = generate_prompt_to_pitch(
                        cliente_info=cliente_info,
                        prompt_assessor=prompt_assessor,
                        model="gpt-5.1",
                        trace_context={"tracer": tracer, "parent_run_id": pitch_run_id},
                        include_api_metrics=True,
                    )
                pitch = pitch_result["text"]
                tracer.log_event(
                    pitch_run_id,
                    "pitch_api_call",
                    {"step": "prompt_to_pitch", **pitch_result["api_metrics"]},
                )
                tracer.log_event(
                    pitch_run_id,
                    "pitch_prompt_to_pitch_completed",
                    {"final_chars": len(pitch)},
                )
                st.session_state["pitch_draft"] = pitch
                st.session_state["pitch_final_text"] = pitch
                st.session_state["pitch_version"] = 1
                tracer.end_run(
                    pitch_run_id,
                    status="completed",
                    outputs={
                        "status": "completed",
                        "mode": pitch_mode,
                        "final_chars": len(pitch),
                    },
                )
                _set_pitch_trace_state(pitch_run_id, "completed", timestamp_field="ended_at")

            except Exception as exc:
                tracer.log_event(pitch_run_id, "pitch_error", {"step": "prompt_to_pitch", "error": str(exc)})
                tracer.end_run(
                    pitch_run_id,
                    status="error",
                    error=str(exc),
                    outputs={"status": "error", "step": "prompt_to_pitch", "mode": pitch_mode},
                )
                _set_pitch_trace_state(pitch_run_id, "error", timestamp_field="ended_at")
                st.error(f"Erro ao gerar pitch no modo prompt-to-pitch: {exc}")

    if not st.session_state.get(SESSION_PITCH_FLOW_STARTED):
        return

    if pitch_mode == PITCH_MODE_PROMPT_TO_PITCH:
        _render_prompt_to_pitch_result()
        return

    if pitch_mode == PITCH_MODE_AUTO_PITCH:
        investimentos_cliente_df = get_investimentos_by_cliente(cliente_id)
        produtos_df = load_produtos()
        carteira_summary = carteira_summary_for_llm(cliente_info, investimentos_cliente_df)

        _render_auto_pitch_header()

        if st.button("Iniciar Auto-Pitch", key="auto_pitch_btn_priorities"):
            pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
            tracer.log_event(
                pitch_run_id,
                "auto_pitch_priority_generation_started",
                {"model_requested": AUTO_PITCH_PRIORITY_MODEL},
            )
            try:
                with st.spinner("Priorizando abordagens do cliente..."):
                    priority_response = generate_auto_pitch_priorities(
                        cliente_info=cliente_info,
                        carteira_summary=carteira_summary,
                        investimentos_cliente_df=investimentos_cliente_df,
                        produtos_df=produtos_df,
                        prompt_assessor=prompt_assessor,
                        model=AUTO_PITCH_PRIORITY_MODEL,
                        trace_context={"tracer": tracer, "parent_run_id": pitch_run_id},
                        include_api_metrics=True,
                    )
                priority_result = priority_response["result"]
                st.session_state["auto_pitch_priorities"] = priority_result.get("prioridades", [])
                st.session_state["auto_pitch_signal_summary"] = priority_result.get("signal_summary")
                st.session_state["auto_pitch_communication_result"] = None
                tracer.log_event(
                    pitch_run_id,
                    "pitch_api_call",
                    {"step": "auto_pitch_priorities", **priority_response["api_metrics"]},
                )
                tracer.log_event(
                    pitch_run_id,
                    "auto_pitch_priority_generation_completed",
                    {"priorities_count": len(priority_result.get("prioridades", []))},
                )
            except Exception as exc:
                tracer.log_event(pitch_run_id, "pitch_error", {"step": "auto_pitch_priorities", "error": str(exc)})
                tracer.end_run(
                    pitch_run_id,
                    status="error",
                    error=str(exc),
                    outputs={"status": "error", "step": "auto_pitch_priorities"},
                )
                _set_pitch_trace_state(pitch_run_id, "error", timestamp_field="ended_at")
                st.error(f"Erro ao gerar prioridades do auto-pitch: {exc}")

        _render_auto_pitch_result(render_header=False)

        if st.session_state.get("auto_pitch_selected_priority") and not st.session_state.get("auto_pitch_communication_result") and st.button(
            "Gerar comunicação",
            key="auto_pitch_btn_communication",
        ):
            pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
            selected_priority = st.session_state["auto_pitch_selected_priority"]
            tracer.log_event(
                pitch_run_id,
                "auto_pitch_priority_selected",
                {
                    "priority_id": selected_priority.get("priority_id"),
                    "titulo": selected_priority.get("titulo"),
                    "categoria": selected_priority.get("categoria"),
                    "model_requested": AUTO_PITCH_COMMUNICATION_MODEL,
                },
            )
            try:
                with st.spinner("Montando racional argumentativo e mensagem..."):
                    communication_response = generate_auto_pitch_communication(
                        cliente_info=cliente_info,
                        carteira_summary=carteira_summary,
                        investimentos_cliente_df=investimentos_cliente_df,
                        produtos_df=produtos_df,
                        selected_priority=selected_priority,
                        model=AUTO_PITCH_COMMUNICATION_MODEL,
                        trace_context={"tracer": tracer, "parent_run_id": pitch_run_id},
                        include_api_metrics=True,
                    )
                communication_result = communication_response["result"]
                st.session_state["auto_pitch_communication_result"] = communication_result
                st.session_state["auto_pitch_communication_revealed"] = False
                tracer.log_event(
                    pitch_run_id,
                    "pitch_api_call",
                    {"step": "auto_pitch_communication", **communication_response["api_metrics"]},
                )
                tracer.log_event(
                    pitch_run_id,
                    "auto_pitch_communication_completed",
                    {"message_chars": len(communication_result.get("mensagem_principal", ""))},
                )
                st.rerun()
            except Exception as exc:
                tracer.log_event(pitch_run_id, "pitch_error", {"step": "auto_pitch_communication", "error": str(exc)})
                tracer.end_run(
                    pitch_run_id,
                    status="error",
                    error=str(exc),
                    outputs={"status": "error", "step": "auto_pitch_communication"},
                )
                _set_pitch_trace_state(pitch_run_id, "error", timestamp_field="ended_at")
                st.error(f"Erro ao gerar racional e comunicação do auto-pitch: {exc}")

        if st.session_state.get("auto_pitch_communication_result"):
            if not st.session_state.get("auto_pitch_communication_revealed") and st.button(
                "✅ Mostrar pitch",
                key="auto_pitch_btn_finalize_run",
            ):
                pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
                selected_priority = st.session_state.get("auto_pitch_selected_priority")
                communication_result = st.session_state.get("auto_pitch_communication_result")
                st.session_state["auto_pitch_communication_revealed"] = True
                _finalize_auto_pitch_run(
                    tracer,
                    pitch_run_id,
                    pitch_mode=pitch_mode,
                    selected_priority=selected_priority,
                    communication_result=communication_result,
                )
                st.rerun()

            if st.session_state.get("auto_pitch_communication_revealed"):
                st.caption("✅ Pitch Gerado")

        return

    st.header("1️⃣ Definir intenção do contato")

    if st.button("🔎 Sugerir Jornadas", key="pitch_btn_sugerir_jornadas"):  # Gerar jornadas
        pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
        tracer.log_event(pitch_run_id, "pitch_step_1_started", {"action": "sugerir_jornadas"})
        jornadas_df = load_jornadas()

        try:
            with st.spinner("Analisando e ranqueando jornadas..."):
                ranking_result = rank_journeys(
                    cliente_info,
                    prompt_assessor,
                    jornadas_df,
                    trace_context={"tracer": tracer, "parent_run_id": pitch_run_id},
                    include_api_metrics=True,
                )
            resultado = ranking_result["result"]
            tracer.log_event(pitch_run_id, "pitch_api_call", {"step": "step_1", **ranking_result["api_metrics"]})
            tracer.log_event(pitch_run_id, "pitch_step_1_completed", {
                "ranking_count": len(resultado.get("ranking", []))
            })

            st.session_state.ranking_resultado = resultado
            st.session_state.etapa = 2
        except Exception as exc:
            tracer.log_event(pitch_run_id, "pitch_error", {"step": "rank_journeys", "error": str(exc)})
            tracer.end_run(pitch_run_id, status="error", error=str(exc), outputs={"status": "error", "step": "rank_journeys"})
            _set_pitch_trace_state(pitch_run_id, "error", timestamp_field="ended_at")
            st.error(f"Erro ao sugerir jornadas: {exc}")

    if st.session_state.etapa >= 2 and st.session_state.ranking_resultado:  # Jornadas já foram geradas

        jornadas_df = load_jornadas()
        resultado = st.session_state.ranking_resultado

        st.subheader("📊 Jornadas Sugeridas")

        ranking = resultado["ranking"]

        jornadas_dict = {}

        for item in ranking:
            jornada_id = item["jornada_id"]
            jornada_base = get_jornada_config(jornadas_df, jornada_id)

            jornadas_dict[jornada_id] = {
                "nome": item["nome_jornada"],
                "score": round(item["score"], 2),
                "descricao_original": jornada_base["descricao_original"],
                "topicos_llm": jornada_base["topicos_llm"],
                "categoria": jornada_base["categoria"],
                "objetivo_principal": jornada_base["objetivo_principal"],
            }

        jornada_escolhida_id = st.radio(
            "Selecione a jornada:",
            options=list(jornadas_dict.keys()),
            format_func=lambda x: f"{jornadas_dict[x]['nome']} (Score: {jornadas_dict[x]['score']})",
            key="pitch_radio_jornada"
        )

        st.divider()

        if "editar_descricao" not in st.session_state:
            st.session_state["editar_descricao"] = False

        if st.button("✏️ Editar descrição da jornada selecionada (opcional)", key="pitch_btn_editar_descricao"):
            st.session_state["editar_descricao"] = True

        if st.session_state["editar_descricao"]:

            descricao_editada = st.text_area(
                "Ajuste o direcionamento da jornada:",
                value=jornadas_dict[jornada_escolhida_id]["descricao_original"],
                height=150,
                key="pitch_descricao_editada_unica"
            )

        else:
            descricao_editada = jornadas_dict[jornada_escolhida_id]["descricao_original"]

        st.session_state["jornada_selecionada"] = {
            "jornada_id": jornada_escolhida_id,
            "nome": jornadas_dict[jornada_escolhida_id]["nome"],
            "categoria": jornadas_dict[jornada_escolhida_id]["categoria"],
            "objetivo_principal": jornadas_dict[jornada_escolhida_id]["objetivo_principal"],
            "descricao_original": jornadas_dict[jornada_escolhida_id]["descricao_original"],
            "descricao_editada": descricao_editada,
            "topicos_llm": jornadas_dict[jornada_escolhida_id]["topicos_llm"],
        }

        with st.expander("🧩 Tópicos configurados para esta jornada", expanded=False):
            if st.session_state["jornada_selecionada"]["topicos_llm"]:
                st.write(st.session_state["jornada_selecionada"]["topicos_llm"])
            else:
                st.warning("Nenhum tópico foi configurado no Excel para esta jornada.")

    # ---------------------------
    # PASSO 4 - Seleção de fontes
    # ---------------------------
    if "jornada_selecionada" in st.session_state and st.session_state["jornada_selecionada"]:

        st.divider()
        st.header("4️⃣ Seleção de Fontes - LLM as Retriever")

        investimentos_cliente_df = get_investimentos_by_cliente(cliente_id)
        produtos_df = load_produtos()

        carteira_summary = carteira_summary_for_llm(cliente_info, investimentos_cliente_df)

        with st.expander("🔎 Resumo do Cliente e Carteira (inputs do passo 4)", expanded=False):
            st.json(carteira_summary)
            st.dataframe(investimentos_cliente_df, width="stretch")

        if st.button("➡️ Executar Passo 4: Selecionar fontes e produtos", key="pitch_btn_step4"):
            pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
            tracer.log_event(pitch_run_id, "pitch_step_4_started")
            try:
                with st.spinner("Selecionando fontes da Knowledge-Base e produtos candidatos..."):
                    step4_response = select_sources_step4(
                        cliente_info=cliente_info,
                        prompt_assessor=prompt_assessor,
                        jornada_selecionada=st.session_state["jornada_selecionada"],
                        carteira_summary=carteira_summary,
                        produtos_df=produtos_df,
                        investimentos_cliente_df=investimentos_cliente_df,
                        kb_dir="knowledge_base",
                        model="gpt-4o-mini",
                        trace_context={"tracer": tracer, "parent_run_id": pitch_run_id},
                        include_api_metrics=True,
                    )
                step4_result = step4_response["result"]
                tracer.log_event(pitch_run_id, "pitch_api_call", {"step": "step_4", **step4_response["api_metrics"]})
                tracer.log_event(pitch_run_id, "pitch_step_4_completed", {
                    "kb_files_count": len(step4_result.get("kb_files_selected", [])),
                    "products_count": len(step4_result.get("products_selected_ids", [])),
                })

                st.session_state["step4_result"] = step4_result
                st.session_state.etapa = 4
            except Exception as exc:
                tracer.log_event(pitch_run_id, "pitch_error", {"step": "step4", "error": str(exc)})
                tracer.end_run(pitch_run_id, status="error", error=str(exc), outputs={"status": "error", "step": "step4"})
                _set_pitch_trace_state(pitch_run_id, "error", timestamp_field="ended_at")
                st.error(f"Erro no Passo 4: {exc}")

        if "step4_result" in st.session_state and st.session_state["step4_result"]:

            step4_result = st.session_state["step4_result"]

            st.success("✅ Passo 4 concluído: fontes e produtos selecionados")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📚 Knowledge Base selecionada (por nome)")
                st.write(step4_result.get("kb_files_selected", []))

            with col2:
                st.subheader("🧾 Data sources a usar")
                st.write(step4_result.get("data_sources", []))

            selected_ids = step4_result.get("products_selected_ids", [])
            produtos_selecionados_df = produtos_df[produtos_df["Produto_ID"].isin(selected_ids)].copy()

            st.subheader("🧩 Produtos candidatos selecionados")
            st.dataframe(produtos_selecionados_df, width="stretch")

            st.subheader("👤 Investimentos atuais do cliente (filtrados)")
            st.dataframe(investimentos_cliente_df, width="stretch")

            rent_12m = carteira_summary.get("rentabilidade_12_meses")
            cdi_12m = carteira_summary.get("cdi_12_meses")
            spread = carteira_summary.get("spread_vs_cdi_12m")

            st.subheader("📈 Rentabilidade carteira vs CDI (12m)")
            st.write({
                "Rentabilidade_12_meses": rent_12m,
                "CDI_12_Meses": cdi_12m,
                "Spread_vs_CDI": spread
            })

            if step4_result.get("reasoning_short"):
                st.caption(f"Racional do agente: {step4_result['reasoning_short']}")

    # ---------------------------
    # PASSO 5 - Estruturar opções para o pitch (com RAG)
    # ---------------------------
    if "step4_result" in st.session_state and st.session_state["step4_result"]:

        st.divider()
        st.header("5️⃣ Estruturar opções do pitch")

        step4 = st.session_state["step4_result"]

        investimentos_cliente_df = get_investimentos_by_cliente(cliente_id)
        produtos_df = load_produtos()

        selected_ids = step4.get("products_selected_ids", [])
        produtos_selecionados_df = produtos_df[produtos_df["Produto_ID"].isin(selected_ids)].copy()

        kb_files_selected = step4.get("kb_files_selected", [])

        carteira_summary = carteira_summary_for_llm(cliente_info, investimentos_cliente_df)

        if st.button("➡️ Executar Passo 5: Gerar opções estruturadas", key="pitch_btn_step5"):
            pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
            tracer.log_event(pitch_run_id, "pitch_step_5_started")
            try:
                with st.spinner("Gerando blocos dinâmicos do pitch..."):
                    step5_response = build_pitch_options_step5(
                        cliente_info=cliente_info,
                        prompt_assessor=prompt_assessor,
                        jornada_selecionada=st.session_state["jornada_selecionada"],
                        carteira_summary=carteira_summary,
                        investimentos_cliente_df=investimentos_cliente_df,
                        produtos_selecionados_df=produtos_selecionados_df,
                        kb_files_selected=kb_files_selected,
                        model="gpt-4o-mini",
                        trace_context={"tracer": tracer, "parent_run_id": pitch_run_id},
                        include_api_metrics=True,
                    )
                step5_result = step5_response["result"]
                tracer.log_event(pitch_run_id, "pitch_api_call", {"step": "step_5", **step5_response["api_metrics"]})
                tracer.log_event(pitch_run_id, "pitch_step_5_completed", {
                    "content_blocks_count": len(step5_result.get("blocos_conteudo", [])),
                    "topics_requested_count": len(st.session_state["jornada_selecionada"].get("topicos_llm", [])),
                })

                st.session_state["step5_result"] = step5_result
                st.session_state["pitch_step5_release_index"] = 0
                st.session_state["pitch_step5_confirmed_sections"] = []
                st.session_state.etapa = 5
            except Exception as exc:
                tracer.log_event(pitch_run_id, "pitch_error", {"step": "step5", "error": str(exc)})
                tracer.end_run(pitch_run_id, status="error", error=str(exc), outputs={"status": "error", "step": "step5"})
                _set_pitch_trace_state(pitch_run_id, "error", timestamp_field="ended_at")
                st.error(f"Erro no Passo 5: {exc}")

        if "step5_result" in st.session_state and st.session_state["step5_result"]:
            step5 = st.session_state["step5_result"]
            st.success("✅ Passo 5 concluído: selecione o que deve entrar no pitch final")

            if "pitch_step5_release_index" not in st.session_state:
                st.session_state["pitch_step5_release_index"] = 0
            if "pitch_step5_confirmed_sections" not in st.session_state:
                st.session_state["pitch_step5_confirmed_sections"] = []

            def _checkbox_list(title, items, key_prefix):
                """Renderiza uma lista de checkboxes e devolve os itens selecionados."""
                st.subheader(title)
                selected = []
                for item in items:
                    cid = item.get("id")
                    txt = item.get("texto", "")
                    k = f"{key_prefix}_{cid}"
                    checked = st.checkbox(txt, value=True, key=k)
                    if checked:
                        selected.append(item)
                return selected

            def _render_content_block(bloco):
                titulo = bloco.get("titulo", "Bloco")
                items = bloco.get("itens", [])
                if not items:
                    st.subheader(f"🧩 {titulo}")
                    st.info("Nenhum item foi retornado para este tópico.")
                    return {"id": bloco.get("id"), "titulo": titulo, "itens": []}

                selected_items = _checkbox_list(
                    f"🧩 {titulo}",
                    items,
                    f"pitch_chk_block_{bloco.get('id', 'topico')}",
                )
                return {
                    "id": bloco.get("id"),
                    "titulo": titulo,
                    "itens": selected_items,
                }

            def _render_tom():
                st.subheader("🗣 Tom do pitch")
                tom_options = []
                if step5.get("tom_sugerido", {}).get("principal"):
                    tom_options.append(step5["tom_sugerido"]["principal"]["texto"])
                for alt in step5.get("tom_sugerido", {}).get("alternativas", []):
                    tom_options.append(alt["texto"])

                if not tom_options:
                    st.info("Nenhuma sugestão de tom foi retornada para este cliente.")
                    return None

                return st.radio(
                    "Escolha o tom:",
                    options=tom_options,
                    index=0,
                    key="pitch_radio_tom"
                )

            def _render_tamanho():
                st.subheader("📏 Tamanho do pitch")
                size_options = []
                if step5.get("tamanho_pitch", {}).get("principal"):
                    size_options.append(step5["tamanho_pitch"]["principal"]["texto"])
                for alt in step5.get("tamanho_pitch", {}).get("alternativas", []):
                    size_options.append(alt["texto"])

                if not size_options:
                    st.info("Nenhuma sugestão de tamanho foi retornada para este cliente.")
                    return None

                return st.radio(
                    "Escolha o tamanho:",
                    options=size_options,
                    index=0,
                    key="pitch_radio_tamanho"
                )

            selected_content_blocks = []
            tom_escolhido = None
            tamanho_escolhido = None

            content_blocks = step5.get("blocos_conteudo", [])
            sections = [
                *[
                    {
                        "id": f"bloco_{bloco.get('id', idx)}",
                        "type": "content_block",
                        "renderer": (lambda current_block=bloco: _render_content_block(current_block)),
                    }
                    for idx, bloco in enumerate(content_blocks)
                ],
                {"id": "tom_sugerido", "type": "tone", "renderer": _render_tom},
                {"id": "tamanho_pitch", "type": "size", "renderer": _render_tamanho},
            ]

            current_release_index = st.session_state["pitch_step5_release_index"]
            confirmed_sections = st.session_state["pitch_step5_confirmed_sections"]

            for index, section in enumerate(sections):
                if index > current_release_index:
                    break

                section_result = section["renderer"]()

                if section["type"] == "content_block":
                    selected_content_blocks.append(section_result)
                elif section["id"] == "tom_sugerido":
                    tom_escolhido = section_result
                elif section["id"] == "tamanho_pitch":
                    tamanho_escolhido = section_result

                is_last_released = index == current_release_index
                has_next_section = index < len(sections) - 1

                if is_last_released:
                    button_label = "Seguir" if has_next_section else "Finalizar argumentos"
                    if st.button(button_label, key=f"pitch_step5_release_btn_{section['id']}"):
                        st.session_state["pitch_step5_confirmed_sections"] = [
                            *confirmed_sections,
                            section["id"],
                        ]
                        if has_next_section:
                            st.session_state["pitch_step5_release_index"] = current_release_index + 1
                        st.rerun()

                if index < current_release_index:
                    st.divider()

            all_sections_confirmed = len(st.session_state["pitch_step5_confirmed_sections"]) == len(sections)

            st.divider()

            if st.button(
                "💾 Salvar seleção (Passo 6)",
                key="pitch_btn_save_step5",
                disabled=not all_sections_confirmed,
            ):
                pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
                tracer.log_event(pitch_run_id, "pitch_step_6_selection_saved", {
                    "content_blocks_selected": len(selected_content_blocks),
                    "content_items_selected": sum(len(bloco.get("itens", [])) for bloco in selected_content_blocks),
                    "confirmed_sections": st.session_state.get("pitch_step5_confirmed_sections", []),
                })
                st.session_state["step5_selection"] = {
                    "blocos_conteudo": selected_content_blocks,
                    "tom_escolhido": tom_escolhido,
                    "tamanho_escolhido": tamanho_escolhido,
                }
                st.session_state.etapa = 6

    # ---------------------------
    # PASSO 7/8 - Pitch (rascunho + ajustes + finalizar)
    # ---------------------------
    if "step5_selection" in st.session_state and st.session_state["step5_selection"]:

        st.divider()
        st.header("7️⃣ Gerar pitch")

        model_pitch_final = "gpt-5.1"
        model_rewriter = "gpt-5-mini"

        if "pitch_draft" not in st.session_state:
            st.session_state["pitch_draft"] = ""
        if "pitch_final_text" not in st.session_state:
            st.session_state["pitch_final_text"] = None
        if "pitch_version" not in st.session_state:
            st.session_state["pitch_version"] = 0

        if st.button("📝 Gerar pitch (rascunho)", key="pitch_btn_step7"):
            pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
            tracer.log_event(pitch_run_id, "pitch_step_7_started")
            try:
                with st.spinner("Escrevendo pitch..."):
                    pitch_result = generate_final_pitch_step7(
                        cliente_info=cliente_info,
                        prompt_assessor=prompt_assessor,
                        jornada_selecionada=st.session_state["jornada_selecionada"],
                        step5_selection=st.session_state["step5_selection"],
                        model=model_pitch_final,
                        trace_context={"tracer": tracer, "parent_run_id": pitch_run_id},
                        include_api_metrics=True,
                    )
                pitch = pitch_result["text"]
                api_metrics = {"step": "step_7", **pitch_result["api_metrics"]}
                tracer.log_event(pitch_run_id, "pitch_api_call", api_metrics)
                tracer.log_event(pitch_run_id, "pitch_step_7_completed", {"draft_chars": len(pitch)})
                st.session_state["pitch_draft"] = pitch
                st.session_state["pitch_final_text"] = None
                st.session_state["pitch_version"] += 1
            except Exception as exc:
                tracer.log_event(pitch_run_id, "pitch_error", {"step": "step7", "error": str(exc)})
                tracer.end_run(pitch_run_id, status="error", error=str(exc), outputs={"status": "error", "step": "step7"})
                _set_pitch_trace_state(pitch_run_id, "error", timestamp_field="ended_at")
                st.error(f"Erro ao gerar rascunho: {exc}")

        if st.session_state["pitch_draft"]:

            st.subheader("🧾 Pitch (rascunho)")
            st.caption("Você pode ajustar quantas vezes quiser. Ao finalizar, o texto fica pronto para copiar/colar.")

            draft_key = f"pitch_draft_box_{st.session_state['pitch_version']}"

            pitch_in_ui = st.text_area(
                "Rascunho atual:",
                value=st.session_state["pitch_draft"],
                height=240,
                key=draft_key
            )

            st.session_state["pitch_draft"] = pitch_in_ui

            st.subheader("8️⃣ Ajustar pitch (opcional)")
            target_excerpt = st.text_input(
                "Trecho específico (opcional):",
                key="pitch_edit_excerpt"
            )
            edit_instruction = st.text_area(
                "Instrução de ajuste (ex: encurtar, deixar mais consultivo, trocar tom, remover produto, etc.):",
                height=110,
                key="pitch_edit_instruction"
            )

            colA, colB = st.columns(2)

            with colA:
                if st.button("🔁 Aplicar ajuste", key="pitch_btn_step8") and edit_instruction.strip():
                    pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
                    tracer.log_event(pitch_run_id, "pitch_step_8_started", {"has_target_excerpt": bool(target_excerpt.strip())})

                    combined_input = "\n".join([st.session_state["pitch_draft"], target_excerpt.strip(), edit_instruction.strip()]).strip()
                    try:
                        guardrail_result = evaluate_input_guardrails(combined_input, context="pitch")
                    except Exception as exc:
                        guardrail_result = handle_guardrail_exception(combined_input, exc)

                    tracer.log_event(
                        pitch_run_id,
                        "input_guardrail_checked",
                        {
                            "step": "step8",
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
                            pitch_run_id,
                            status="blocked",
                            outputs={
                                "status": "blocked",
                                "blocked_step": "step8",
                                "guardrail": {
                                    "violation_type": guardrail_result.violation_type,
                                    "reason": guardrail_result.message,
                                },
                            },
                        )
                        _set_pitch_trace_state(pitch_run_id, "blocked", timestamp_field="ended_at")
                        st.warning(guardrail_warning_message(guardrail_result.violation_type, context="pitch"))
                        return

                    try:
                        with st.spinner("Aplicando ajuste..."):
                            revised_result = revise_pitch_step8(
                                current_pitch=st.session_state["pitch_draft"],
                                edit_instruction=edit_instruction.strip(),
                                target_excerpt=target_excerpt.strip() if target_excerpt.strip() else None,
                                model=model_rewriter,
                                trace_context={"tracer": tracer, "parent_run_id": pitch_run_id},
                                include_api_metrics=True,
                            )
                        revised = revised_result["text"]
                        tracer.log_event(pitch_run_id, "pitch_api_call", {"step": "step_8", **revised_result["api_metrics"]})
                        tracer.log_event(pitch_run_id, "pitch_step_8_completed", {"draft_chars": len(revised)})
                        st.session_state["pitch_draft"] = revised
                        st.session_state["pitch_version"] += 1
                        st.success("✅ Ajuste aplicado. Veja o pitch atualizado acima.")
                        st.rerun()
                    except Exception as exc:
                        tracer.log_event(pitch_run_id, "pitch_error", {"step": "step8", "error": str(exc)})
                        tracer.end_run(pitch_run_id, status="error", error=str(exc), outputs={"status": "error", "step": "step8"})
                        _set_pitch_trace_state(pitch_run_id, "error", timestamp_field="ended_at")
                        st.error(f"Erro ao aplicar ajuste: {exc}")

            with colB:
                if st.button("✅ Finalizar pitch", key="pitch_btn_finalize"):
                    pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
                    st.session_state["pitch_final_text"] = st.session_state["pitch_draft"]
                    tracer.log_event(pitch_run_id, "pitch_finalized", {"final_chars": len(st.session_state["pitch_final_text"] or "")})
                    tracer.end_run(
                        pitch_run_id,
                        status="completed",
                        outputs={
                            "status": "completed",
                            "final_chars": len(st.session_state["pitch_final_text"] or ""),
                        },
                    )
                    _set_pitch_trace_state(pitch_run_id, "completed", timestamp_field="ended_at")
                    st.success("✅ Pitch finalizado")

            if st.session_state["pitch_final_text"]:
                st.divider()
                st.subheader("📨 Pitch final (copiar e colar)")
                st.text_area(
                    "Texto final:",
                    value=st.session_state["pitch_final_text"],
                    height=240,
                    key="pitch_final_box"
                )

                if st.button("↩️ Voltar para ajustes", key="pitch_btn_back_to_edit"):
                    st.session_state["pitch_final_text"] = None
