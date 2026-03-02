import streamlit as st
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
    summarize_transcript,
    transcribe_audio,
)


st.set_page_config(page_title="POC Jornada Comercial", layout="wide")


def init_session_state():
    if "etapa" not in st.session_state:
        st.session_state.etapa = 1

    if "ranking_resultado" not in st.session_state:
        st.session_state.ranking_resultado = None

    if "selected_cliente_id" not in st.session_state:
        clientes_df = load_clientes()
        st.session_state.selected_cliente_id = clientes_df["Cliente_ID"].iloc[0]


def render_pitch_tab(cliente_id, cliente_info):
    st.header("1️⃣ Definir intenção do contato")

    prompt_assessor = st.text_area(
        "Escreva o objetivo do contato:",
        height=150,
        key="pitch_prompt_assessor"
    )

    if st.button("🔎 Sugerir Jornadas", key="pitch_btn_sugerir_jornadas"):  # Gerar jornadas

        jornadas_df = load_jornadas()

        with st.spinner("Analisando e ranqueando jornadas..."):
            resultado = rank_journeys(cliente_info, prompt_assessor, jornadas_df)

        st.session_state.ranking_resultado = resultado
        st.session_state.etapa = 2

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
            st.dataframe(investimentos_cliente_df, use_container_width=True)

        if st.button("➡️ Executar Passo 4: Selecionar fontes e produtos", key="pitch_btn_step4"):

            with st.spinner("Selecionando fontes da Knowledge-Base e produtos candidatos..."):
                step4_result = select_sources_step4(
                    cliente_info=cliente_info,
                    prompt_assessor=prompt_assessor,
                    jornada_selecionada=st.session_state["jornada_selecionada"],
                    carteira_summary=carteira_summary,
                    produtos_df=produtos_df,
                    investimentos_cliente_df=investimentos_cliente_df,
                    kb_dir="knowledge_base",
                    model="gpt-4o-mini"
                )

            st.session_state["step4_result"] = step4_result
            st.session_state.etapa = 4

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
            st.dataframe(produtos_selecionados_df, use_container_width=True)

            st.subheader("👤 Investimentos atuais do cliente (filtrados)")
            st.dataframe(investimentos_cliente_df, use_container_width=True)

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
            with st.spinner("Gerando diagnóstico, pontos e opções do pitch..."):
                step5_result = build_pitch_options_step5(
                    cliente_info=cliente_info,
                    prompt_assessor=prompt_assessor,
                    jornada_selecionada=st.session_state["jornada_selecionada"],
                    carteira_summary=carteira_summary,
                    investimentos_cliente_df=investimentos_cliente_df,
                    produtos_selecionados_df=produtos_selecionados_df,
                    kb_files_selected=kb_files_selected,
                    model="gpt-4o-mini"
                )

            st.session_state["step5_result"] = step5_result
            st.session_state.etapa = 5

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
            with st.spinner("Escrevendo pitch..."):
                pitch = generate_final_pitch_step7(
                    cliente_info=cliente_info,
                    prompt_assessor=prompt_assessor,
                    jornada_selecionada=st.session_state["jornada_selecionada"],
                    step5_selection=st.session_state["step5_selection"],
                    model=model_writer
                )
            st.session_state["pitch_draft"] = pitch
            st.session_state["pitch_final_text"] = None
            st.session_state["pitch_version"] += 1
            st.success("✅ Rascunho gerado")

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
                    with st.spinner("Aplicando ajuste..."):
                        revised = revise_pitch_step8(
                            current_pitch=st.session_state["pitch_draft"],
                            edit_instruction=edit_instruction.strip(),
                            target_excerpt=target_excerpt.strip() if target_excerpt.strip() else None,
                            model=model_writer
                        )
                    st.session_state["pitch_draft"] = revised
                    st.session_state["pitch_version"] += 1
                    st.success("✅ Ajuste aplicado. Veja o pitch atualizado acima.")
                    st.rerun()

            with colB:
                if st.button("✅ Finalizar pitch", key="pitch_btn_finalize"):
                    st.session_state["pitch_final_text"] = st.session_state["pitch_draft"]
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
            with st.spinner("Transcrevendo áudio..."):
                transcript = transcribe_audio(audio_bytes, audio_name, audio_type)
            with st.spinner("Gerando resumo da reunião..."):
                summary = summarize_transcript(cliente_info, transcript)

            meeting_path = save_meeting(
                cliente_id=cliente_id,
                cliente_nome=cliente_info.get("Nome", "Cliente"),
                cliente_info=cliente_info,
                transcript=transcript,
                summary=summary,
            )
            st.session_state.meetings_last_saved_path = str(meeting_path)
            st.success(f"Resumo salvo em: {meeting_path}")

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


def render_portfolio_tab():
    st.title("Carteira (Talk to Data)")
    st.write("Em breve")


def render_insights_tab():
    st.title("Insights")
    st.write("Em breve")


def render_settings_tab():
    st.title("Configurações")
    st.write("Em breve")


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
    st.sidebar.json(cliente_info)

    tab_pitch, tab_meetings, tab_portfolio, tab_insights, tab_settings = st.tabs([
        "Pitch (POC)",
        "Reuniões",
        "Carteira (Talk to Data)",
        "Insights",
        "Configurações"
    ])

    with tab_pitch:
        render_pitch_tab(st.session_state.selected_cliente_id, cliente_info)

    with tab_meetings:
        render_meetings_tab(st.session_state.selected_cliente_id, cliente_info)

    with tab_portfolio:
        render_portfolio_tab()

    with tab_insights:
        render_insights_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
