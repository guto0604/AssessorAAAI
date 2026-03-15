from datetime import datetime

import streamlit as st

from core.data_loader import (
    carteira_summary_for_llm,
    get_investimentos_by_cliente,
    load_jornadas,
    load_produtos,
)
from core.journey_ranker import rank_journeys
from core.langsmith_tracing import LangSmithTracer
from core.pitch_structurer import build_pitch_options_step5
from core.pitch_writer import generate_final_pitch_step7, revise_pitch_step8
from core.source_selector import select_sources_step4
from ui.guardrails import (
    evaluate_input_guardrails,
    guardrail_warning_message,
    handle_guardrail_exception,
)
from ui.state import (
    SESSION_PITCH_FLOW_STARTED,
    SESSION_PITCH_TRACE,
    _iso_now,
    get_tracer,
)

def _start_pitch_trace(tracer: LangSmithTracer, cliente_id, prompt_assessor: str) -> str | None:
    active_trace = st.session_state.get(SESSION_PITCH_TRACE)
    if active_trace and active_trace.get("run_id") and active_trace.get("status") == "in_progress":
        tracer.log_event(active_trace["run_id"], "pitch_interrupted", {
            "reason": "Novo fluxo iniciado antes da finalização",
            "at": _iso_now(),
        })
        tracer.end_run(active_trace["run_id"], status="interrupted", outputs={"status": "interrupted"})

    run_id = tracer.start_run(
        name=f"pitch_cliente_{cliente_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_type="chain",
        inputs={
            "cliente_id": cliente_id,
            "prompt_assessor": prompt_assessor,
        },
        tags=["pitch", "streamlit"],
        metadata={"started_at": _iso_now(), "cliente_id": cliente_id, "prompt_preview": prompt_assessor[:180]},
    )
    st.session_state[SESSION_PITCH_TRACE] = {
        "run_id": run_id,
        "status": "in_progress",
        "started_at": _iso_now(),
    }
    return run_id



def _reset_pitch_flow_state():
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
    ]
    for key in keys_to_reset:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("pitch_chk_") or key.startswith("pitch_draft_box_"):
            st.session_state.pop(key, None)



def render_pitch_tab(cliente_id, cliente_info):
    st.header("🚀 Iniciar fluxo de pitch")

    prompt_assessor = st.text_area(
        "Escreva o objetivo do contato:",
        height=150,
        key="pitch_prompt_assessor"
    )

    tracer = get_tracer()

    start_label = "▶️ Iniciar pitch" if not st.session_state.get(SESSION_PITCH_FLOW_STARTED) else "🔄 Iniciar novo pitch"
    if st.button(start_label, key="pitch_btn_start_new_flow"):
        _reset_pitch_flow_state()
        pitch_run_id = _start_pitch_trace(tracer, cliente_id, prompt_assessor)
        tracer.log_event(pitch_run_id, "pitch_flow_initialized", {
            "action": "start_new_flow",
            "at": _iso_now(),
            "prompt_chars": len(prompt_assessor.strip()),
        })

        try:
            guardrail_result = evaluate_input_guardrails(prompt_assessor.strip())
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
            st.session_state[SESSION_PITCH_TRACE] = {
                "run_id": pitch_run_id,
                "status": "blocked",
                "ended_at": _iso_now(),
            }
            st.warning(guardrail_warning_message(guardrail_result.violation_type))
            return

        st.session_state[SESSION_PITCH_FLOW_STARTED] = True
        st.success("Fluxo iniciado. Agora siga com as etapas abaixo.")

    if not st.session_state.get(SESSION_PITCH_FLOW_STARTED):
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
            st.session_state[SESSION_PITCH_TRACE] = {
                "run_id": pitch_run_id,
                "status": "error",
                "ended_at": _iso_now(),
            }
            st.error(f"Erro ao sugerir jornadas: {exc}")

    if st.session_state.etapa >= 2 and st.session_state.ranking_resultado:  # Jornadas já foram geradas

        jornadas_df = load_jornadas()
        resultado = st.session_state.ranking_resultado

        st.subheader("📊 Jornadas Sugeridas")

        ranking = resultado["ranking"]

        jornadas_dict = {}

        for item in ranking:
            jornada_id = item["jornada_id"]
            jornada_base = jornadas_df[jornadas_df["Jornada_ID"] == jornada_id].iloc[0]

            jornadas_dict[jornada_id] = {
                "nome": item["nome_jornada"],
                "score": round(item["score"], 2),
                "descricao_original": jornada_base["Descricao_Resumida"]
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

        if st.button("✏️ Editar descrição da jornada selecionada", key="pitch_btn_editar_descricao"):
            st.session_state["editar_descricao"] = True

        if st.session_state["editar_descricao"]:

            descricao_editada = st.text_area(
                "Ajuste o direcionamento da jornada:",
                value=jornadas_dict[jornada_escolhida_id]["descricao_original"],
                height=150,
                key="pitch_descricao_editada_unica"
            )

            st.session_state["jornada_selecionada"] = {
                "jornada_id": jornada_escolhida_id,
                "descricao_editada": descricao_editada
            }

    # ---------------------------
    # PASSO 4 - Seleção de fontes
    # ---------------------------
    if "jornada_selecionada" in st.session_state and st.session_state["jornada_selecionada"]:

        st.divider()
        st.header("4️⃣ Seleção de Fontes (Agent Router)")

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
                st.session_state[SESSION_PITCH_TRACE] = {
                    "run_id": pitch_run_id,
                    "status": "error",
                    "ended_at": _iso_now(),
                }
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
        st.header("5️⃣ Estruturar opções do pitch (RAG + LLM)")

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
                with st.spinner("Gerando diagnóstico, pontos e opções do pitch..."):
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
                    "diagnostico_count": len(step5_result.get("diagnostico", [])),
                    "products_count": len(step5_result.get("produtos_sugeridos", [])),
                })

                st.session_state["step5_result"] = step5_result
                st.session_state.etapa = 5
            except Exception as exc:
                tracer.log_event(pitch_run_id, "pitch_error", {"step": "step5", "error": str(exc)})
                tracer.end_run(pitch_run_id, status="error", error=str(exc), outputs={"status": "error", "step": "step5"})
                st.session_state[SESSION_PITCH_TRACE] = {
                    "run_id": pitch_run_id,
                    "status": "error",
                    "ended_at": _iso_now(),
                }
                st.error(f"Erro no Passo 5: {exc}")

        if "step5_result" in st.session_state and st.session_state["step5_result"]:
            step5 = st.session_state["step5_result"]
            st.success("✅ Passo 5 concluído: selecione o que deve entrar no pitch final")

            def _checkbox_list(title, items, key_prefix):
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

            selected_diagnostico = _checkbox_list(
                "📌 Diagnóstico (carteira / perfil / rendimento)",
                step5.get("diagnostico", []),
                "pitch_chk_diag"
            )

            selected_pontos = _checkbox_list(
                "🎯 Pontos prioritários para abordar",
                step5.get("pontos_prioritarios", []),
                "pitch_chk_pontos"
            )

            selected_gatilhos = _checkbox_list(
                "⚡ Gatilhos comerciais (opcional)",
                step5.get("gatilhos_comerciais", []),
                "pitch_chk_gatilhos"
            )

            st.subheader("🛡 Possíveis objeções e respostas (pré-tratadas)")
            selected_obj = []
            for item in step5.get("objecoes_e_respostas", []):
                oid = item.get("id")
                obj = item.get("objecao", "")
                resp_txt = item.get("resposta", "")
                label = f"Objeção: {obj}\nResposta sugerida: {resp_txt}"
                k = f"pitch_chk_obj_{oid}"
                checked = st.checkbox(label, value=True, key=k)
                if checked:
                    selected_obj.append(item)

            st.subheader("💼 Sugestões de produtos (candidatos)")
            selected_prod = []
            for item in step5.get("produtos_sugeridos", []):
                pid = item.get("id")
                prod_id = item.get("produto_id")
                txt = item.get("texto", "")
                label = f"[{prod_id}] {txt}" if prod_id else txt
                k = f"pitch_chk_prod_{pid}"
                checked = st.checkbox(label, value=True, key=k)
                if checked:
                    selected_prod.append(item)

            st.subheader("🗣 Tom do pitch")
            tom_options = []
            if step5.get("tom_sugerido", {}).get("principal"):
                tom_options.append(step5["tom_sugerido"]["principal"]["texto"])
            for alt in step5.get("tom_sugerido", {}).get("alternativas", []):
                tom_options.append(alt["texto"])

            tom_escolhido = None
            if tom_options:
                tom_escolhido = st.radio(
                    "Escolha o tom:",
                    options=tom_options,
                    index=0,
                    key="pitch_radio_tom"
                )

            st.subheader("📏 Tamanho do pitch")
            size_options = []
            if step5.get("tamanho_pitch", {}).get("principal"):
                size_options.append(step5["tamanho_pitch"]["principal"]["texto"])
            for alt in step5.get("tamanho_pitch", {}).get("alternativas", []):
                size_options.append(alt["texto"])

            tamanho_escolhido = None
            if size_options:
                tamanho_escolhido = st.radio(
                    "Escolha o tamanho:",
                    options=size_options,
                    index=0,
                    key="pitch_radio_tamanho"
                )

            st.divider()

            if st.button("💾 Salvar seleção (Passo 6)", key="pitch_btn_save_step5"):
                pitch_run_id = (st.session_state.get(SESSION_PITCH_TRACE) or {}).get("run_id")
                tracer.log_event(pitch_run_id, "pitch_step_6_selection_saved", {
                    "diagnostico_selected": len(selected_diagnostico),
                    "pontos_selected": len(selected_pontos),
                    "produtos_selected": len(selected_prod),
                })
                st.session_state["step5_selection"] = {
                    "diagnostico": selected_diagnostico,
                    "pontos_prioritarios": selected_pontos,
                    "gatilhos_comerciais": selected_gatilhos,
                    "objecoes_e_respostas": selected_obj,
                    "produtos_sugeridos": selected_prod,
                    "tom_escolhido": tom_escolhido,
                    "tamanho_escolhido": tamanho_escolhido
                }
                st.session_state.etapa = 6
                st.success("✅ Seleção salva. Pronto para o Passo 6/7 (pitch final).")

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
                st.success("✅ Rascunho gerado")
            except Exception as exc:
                tracer.log_event(pitch_run_id, "pitch_error", {"step": "step7", "error": str(exc)})
                tracer.end_run(pitch_run_id, status="error", error=str(exc), outputs={"status": "error", "step": "step7"})
                st.session_state[SESSION_PITCH_TRACE] = {
                    "run_id": pitch_run_id,
                    "status": "error",
                    "ended_at": _iso_now(),
                }
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
                        guardrail_result = evaluate_input_guardrails(combined_input)
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
                        st.session_state[SESSION_PITCH_TRACE] = {
                            "run_id": pitch_run_id,
                            "status": "blocked",
                            "ended_at": _iso_now(),
                        }
                        st.warning(guardrail_warning_message(guardrail_result.violation_type))
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
                        st.session_state[SESSION_PITCH_TRACE] = {
                            "run_id": pitch_run_id,
                            "status": "error",
                            "ended_at": _iso_now(),
                        }
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
                    st.session_state[SESSION_PITCH_TRACE] = {
                        "run_id": pitch_run_id,
                        "status": "completed",
                        "ended_at": _iso_now(),
                    }
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


