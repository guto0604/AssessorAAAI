import os
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from openai_client import SESSION_OPENAI_KEY
from openai_client import get_effective_openai_api_key
from openai_client import get_openai_client
from langsmith_tracing import LangSmithTracer
from data_loader import (
    load_clientes, load_jornadas, get_cliente_by_id,
    load_investimentos, load_produtos,
    get_investimentos_by_cliente, carteira_summary_for_llm
)

from journey_ranker import rank_journeys
from source_selector import select_sources_step4
from pitch_structurer import build_pitch_options_step5

from pitch_writer import generate_final_pitch_step7, revise_pitch_step8
from meetings import (
    list_client_meetings,
    save_meeting,
    process_meeting_with_langchain,
)


st.set_page_config(page_title="POC Jornada Comercial", layout="wide")


SESSION_LANGSMITH_KEY = "user_langsmith_api_key"
SESSION_LANGSMITH_TRACING_ENABLED = "user_langsmith_tracing_enabled"
SESSION_PITCH_TRACE = "pitch_trace_run"
SESSION_MEETING_TRACE = "meeting_trace_run"
SESSION_PITCH_FLOW_STARTED = "pitch_flow_started"
SESSION_TRACING_HEALTH_STATUS = "langsmith_tracing_health_status"
SESSION_LANGSMITH_TRACER = "langsmith_tracer_instance"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TALK_TO_DATA_FILES = {
    "Informacoes_Cliente": DATA_DIR / "informacoes_cliente.parquet",
    "Investimentos_Cliente": DATA_DIR / "investimentos_cliente.parquet",
    "Produtos": DATA_DIR / "produtos.parquet",
}
REFERENCE_FILE_PATH = DATA_DIR / "referencia_base_dados.txt"


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


def _start_meeting_trace(tracer: LangSmithTracer, cliente_id, audio_name: str | None) -> str | None:
    run_id = tracer.start_run(
        name=f"meeting_cliente_{cliente_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_type="chain",
        inputs={"cliente_id": cliente_id, "audio_name": audio_name},
        tags=["meeting", "streamlit"],
        metadata={"started_at": _iso_now(), "cliente_id": cliente_id, "audio_name": audio_name},
    )
    st.session_state[SESSION_MEETING_TRACE] = {
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


def _build_cliente_sidebar_table(cliente_info: dict) -> pd.DataFrame:
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

        model_writer = "gpt-5-mini"

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
                        model=model_writer,
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
                    try:
                        with st.spinner("Aplicando ajuste..."):
                            revised_result = revise_pitch_step8(
                                current_pitch=st.session_state["pitch_draft"],
                                edit_instruction=edit_instruction.strip(),
                                target_excerpt=target_excerpt.strip() if target_excerpt.strip() else None,
                                model=model_writer,
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


def render_meetings_tab(cliente_id, cliente_info):
    st.title("Reuniões")

    tracer = get_tracer()

    if "meetings_last_saved_path" not in st.session_state:
        st.session_state.meetings_last_saved_path = None

    st.subheader("1) Gravar/Enviar áudio")

    uploaded_audio = None
    audio_bytes = None
    audio_name = None
    audio_type = None

    if hasattr(st, "audio_input"):
        recorded_audio = st.audio_input("Gravar áudio", key="meetings_audio_input")
        if recorded_audio is not None:
            audio_bytes = recorded_audio.getvalue()
            audio_name = getattr(recorded_audio, "name", "gravacao_reuniao.wav")
            audio_type = getattr(recorded_audio, "type", "audio/wav")
            st.audio(audio_bytes)
    else:
        st.info("Gravação nativa não disponível nesta versão. Use upload de áudio.")

    uploaded_audio = st.file_uploader(
        "Upload de áudio",
        type=["wav", "mp3", "m4a"],
        key="meetings_audio_upload",
        help="Envie um arquivo de áudio caso prefira não gravar diretamente.",
    )

    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.getvalue()
        audio_name = uploaded_audio.name
        audio_type = uploaded_audio.type
        st.audio(audio_bytes)

    if st.button("Transcrever e resumir", key="meetings_btn_transcrever_resumir"):
        if not audio_bytes:
            st.warning("Grave ou envie um áudio antes de transcrever.")
        else:
            meeting_run_id = _start_meeting_trace(tracer, cliente_id, audio_name)
            tracer.log_event(meeting_run_id, "meeting_transcription_started", {"audio_type": audio_type})
            try:
                with st.spinner("Transcrevendo áudio e gerando resumo da reunião..."):
                    meeting_result = process_meeting_with_langchain(
                        cliente_info=cliente_info,
                        audio_bytes=audio_bytes,
                        audio_name=audio_name,
                        audio_type=audio_type,
                        trace_context={"tracer": tracer, "parent_run_id": meeting_run_id},
                        include_api_metrics=True,
                    )
                transcript = meeting_result["transcript"]
                summary = meeting_result["summary"]
                api_calls = meeting_result.get("api_calls", [])
                for api_call in api_calls:
                    tracer.log_event(meeting_run_id, "meeting_api_call", api_call)
                tracer.log_event(meeting_run_id, "meeting_transcription_completed", {"transcript_chars": len(transcript)})
                tracer.log_event(meeting_run_id, "meeting_summary_completed", {"summary_chars": len(summary)})

                meeting_path = save_meeting(
                    cliente_id=cliente_id,
                    cliente_nome=cliente_info.get("Nome", "Cliente"),
                    cliente_info=cliente_info,
                    transcript=transcript,
                    summary=summary,
                    api_calls=api_calls,
                )
                st.session_state.meetings_last_saved_path = str(meeting_path)
                tracer.end_run(
                    meeting_run_id,
                    status="completed",
                    outputs={
                        "status": "completed",
                        "meeting_path": str(meeting_path),
                    },
                )
                st.session_state[SESSION_MEETING_TRACE] = {
                    "run_id": meeting_run_id,
                    "status": "completed",
                    "ended_at": _iso_now(),
                }
                st.success(f"Resumo salvo em: {meeting_path}")
            except Exception as exc:
                tracer.log_event(meeting_run_id, "meeting_error", {"error": str(exc)})
                tracer.end_run(meeting_run_id, status="error", error=str(exc), outputs={"status": "error"})
                st.session_state[SESSION_MEETING_TRACE] = {
                    "run_id": meeting_run_id,
                    "status": "error",
                    "ended_at": _iso_now(),
                }
                st.error(f"Erro ao processar reunião: {exc}")

    if st.session_state.meetings_last_saved_path:
        st.caption(f"Último arquivo salvo: {st.session_state.meetings_last_saved_path}")

    st.subheader("2) Histórico de reuniões")
    st.button("🔄 Atualizar histórico", key="meetings_btn_refresh_history")

    meeting_files = list_client_meetings(cliente_id)
    if not meeting_files:
        st.info("Nenhuma reunião salva para este cliente ainda.")
    else:
        selected_meeting = st.selectbox(
            "Selecione uma reunião",
            options=meeting_files,
            format_func=lambda p: p.name,
            key="meetings_history_select",
        )

        if selected_meeting:
            content = selected_meeting.read_text(encoding="utf-8")
            st.text_area(
                "Conteúdo da reunião selecionada",
                value=content,
                height=320,
                key="meetings_history_content",
                disabled=True,
            )


def render_talk_to_your_data_page():
    st.title("Talk to your Data")
    st.caption("Faça perguntas em linguagem natural e consulte os dados com SQL via DuckDB.")

    history = st.session_state.setdefault("talk_to_data_history", [])
    for item in history[-4:]:
        with st.container(border=True):
            st.markdown(f"**Pergunta:** {item['question']}")
            st.markdown(f"**Resposta:** {item['answer']}")

    question = st.text_area(
        "Pergunte sobre a base de assessoria:",
        placeholder="Ex.: Quais clientes fazem aniversário neste mês?",
        key="talk_to_data_question",
        height=100,
    )

    if st.button("Enviar pergunta", key="talk_to_data_submit"):
        if not question.strip():
            st.warning("Escreva uma pergunta antes de enviar.")
            return

        try:
            reference_text = load_reference_text()
            prompt = build_llm_prompt(question=question.strip(), reference_text=reference_text)
            llm_output = ask_talk_to_data_llm(prompt)
        except Exception as exc:
            st.error(f"Falha ao interpretar a pergunta com a LLM: {exc}")
            return

        can_answer = bool(llm_output.get("can_answer", False))
        rationale = llm_output.get("rationale", "")
        question_understanding = llm_output.get("question_understanding", "")
        sql = (llm_output.get("sql") or "").strip()
        answer = llm_output.get("answer") or "Não foi possível gerar uma resposta."
        visualization = llm_output.get("visualization") or {"needed": False, "type": "none"}

        st.subheader("Entendimento da pergunta")
        st.write(question_understanding or "A LLM não retornou entendimento explícito.")

        with st.expander("Racional curto", expanded=False):
            st.write(rationale or "Sem racional informado.")
            st.write({
                "tables_used": llm_output.get("tables_used", []),
                "fields_used": llm_output.get("fields_used", []),
            })

        if not can_answer:
            st.info(answer)
            history.append({"question": question.strip(), "answer": answer})
            return

        if not sql:
            st.warning("A pergunta foi marcada como respondível, mas nenhum SQL foi retornado.")
            history.append({"question": question.strip(), "answer": answer})
            return

        with st.expander("SQL gerado", expanded=True):
            st.code(sql, language="sql")

        result_df = pd.DataFrame()
        query_error = None
        try:
            result_df = run_duckdb_query(sql)
        except Exception as exc:
            query_error = str(exc)

        if query_error:
            st.error(f"Erro ao executar SQL no DuckDB: {query_error}")
            return

        st.subheader("Resultado da consulta")
        if result_df.empty:
            st.info("A consulta foi executada, mas não retornou linhas.")
        else:
            st.dataframe(result_df, width="stretch")

        st.subheader("Resposta final")
        st.write(answer)
        render_visual(result_df, visualization)

        history.append({"question": question.strip(), "answer": answer})


def load_reference_text() -> str:
    return REFERENCE_FILE_PATH.read_text(encoding="utf-8")


def build_llm_prompt(question: str, reference_text: str) -> str:
    return f"""
Você é um analista de dados de uma assessoria de investimentos.

Regras:
- Use apenas tabelas e campos descritos na referência.
- Não invente campos.
- Se não for possível responder, use can_answer=false, sql="", visualization.type="none".
- Gere SQL compatível com DuckDB.
- Prefira queries leves (agregações e LIMIT quando fizer sentido).
- Nunca gere código Python para visualização.
- Retorne APENAS JSON válido.

Formato de saída JSON:
{{
  "can_answer": true/false,
  "question_understanding": "texto curto",
  "rationale": "texto curto",
  "tables_used": ["..."],
  "fields_used": ["..."],
  "sql": "...",
  "visualization": {{
    "needed": true/false,
    "type": "bar|line|pie|table|none",
    "x": "campo ou vazio",
    "y": "campo ou vazio",
    "title": "título"
  }},
  "answer": "resposta executiva em português"
}}

Pergunta do usuário:
{question}

Referência completa da base:
{reference_text}
""".strip()


def ask_talk_to_data_llm(prompt: str) -> dict:
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Você responde apenas com JSON válido."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json\n", "", 1).strip()
    return json.loads(content)


def run_duckdb_query(sql: str) -> pd.DataFrame:
    normalized_sql = sql.strip().rstrip(";")
    if not normalized_sql.lower().startswith("select"):
        raise ValueError("Apenas consultas SELECT são permitidas.")

    with duckdb.connect(database=":memory:") as con:
        for table_name, file_path in TALK_TO_DATA_FILES.items():
            con.execute(
                f'CREATE VIEW "{table_name}" AS SELECT * FROM read_parquet(?)',
                [str(file_path)],
            )
        return con.execute(normalized_sql).fetchdf()


def render_visual(result_df: pd.DataFrame, visualization_spec: dict):
    vis_type = str(visualization_spec.get("type", "none")).lower()
    if not visualization_spec.get("needed") or vis_type == "none":
        return

    st.subheader("Visualização")
    if result_df.empty:
        st.info("Sem dados para visualizar.")
        return

    x = visualization_spec.get("x")
    y = visualization_spec.get("y")
    title = visualization_spec.get("title") or "Visual gerado"

    if vis_type == "table":
        st.dataframe(result_df, width="stretch")
        return

    if x and x not in result_df.columns:
        st.info(f"Não foi possível renderizar o visual: coluna x '{x}' não encontrada no resultado.")
        return

    if vis_type in {"bar", "line", "pie"} and y and y not in result_df.columns:
        st.info(f"Não foi possível renderizar o visual: coluna y '{y}' não encontrada no resultado.")
        return

    if vis_type == "bar":
        st.bar_chart(result_df, x=x, y=y)
    elif vis_type == "line":
        st.line_chart(result_df, x=x, y=y)
    elif vis_type == "pie":
        if not x or not y:
            st.info("Visual de pizza requer campos x e y válidos.")
            return
        fig = px.pie(result_df, names=x, values=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tipo de visual não suportado; exibindo tabela.")
        st.dataframe(result_df, width="stretch")


def render_insights_tab():
    st.title("Insights")
    st.write("Em breve")


def render_settings_tab():
    st.title("Configurações")
    st.caption("Preencha apenas as credenciais essenciais da sessão (quando necessário).")

    env_openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    env_langsmith_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    current_openai_key = (st.session_state.get(SESSION_OPENAI_KEY, "") or "").strip()
    current_langsmith_key = (st.session_state.get(SESSION_LANGSMITH_KEY, "") or "").strip()

    effective_openai_key = (get_effective_openai_api_key() or "").strip()
    effective_langsmith_key = env_langsmith_key or current_langsmith_key

    st.subheader("Integrações")

    openai_is_configured = bool(effective_openai_key)
    langsmith_is_configured = bool(effective_langsmith_key)

    tracing_health_status = st.session_state.get(SESSION_TRACING_HEALTH_STATUS)
    tracing_is_ok = tracing_health_status == "ok"

    st.markdown("### Status")
    st.code(
        json.dumps(
            {
                "openai_api_key_configurada": openai_is_configured,
                "langsmith_api_key_configurada": langsmith_is_configured,
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

    if st.button("💾 Salvar configurações", key="settings_save_keys"):
        if not env_openai_key:
            st.session_state[SESSION_OPENAI_KEY] = openai_api_key_input.strip()
        if not env_langsmith_key:
            st.session_state[SESSION_LANGSMITH_KEY] = langsmith_api_key_input.strip()
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


def main():
    init_session_state()

    st.title("Contato Assessor")

    st.sidebar.header("Selecionar Cliente")
    clientes_df = load_clientes()

    cliente_ids = clientes_df["Cliente_ID"].tolist()
    selected_index = 0
    if st.session_state.selected_cliente_id in cliente_ids:
        selected_index = int(cliente_ids.index(st.session_state.selected_cliente_id))

    selected_cliente_id = st.sidebar.selectbox(
        "Cliente",
        cliente_ids,
        index=selected_index,
        key="global_cliente_select"
    )
    st.session_state.selected_cliente_id = selected_cliente_id

    cliente_info = get_cliente_by_id(st.session_state.selected_cliente_id)

    st.sidebar.markdown("### Dados do Cliente")
    dados_cliente_df = _build_cliente_sidebar_table(cliente_info)
    st.sidebar.table(dados_cliente_df)

    tab_pitch, tab_meetings, tab_portfolio, tab_insights, tab_settings = st.tabs([
        "Voz do Assessor (Pitch)",
        "Resumo Reuniões",
        "Talk to your Data",
        "Insights",
        "Configurações"
    ])

    with tab_pitch:
        render_pitch_tab(st.session_state.selected_cliente_id, cliente_info)

    with tab_meetings:
        render_meetings_tab(st.session_state.selected_cliente_id, cliente_info)

    with tab_portfolio:
        render_talk_to_your_data_page()

    with tab_insights:
        render_insights_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
