from datetime import datetime

import streamlit as st

from core.langsmith_tracing import LangSmithTracer
from core.meetings import list_client_meetings, process_meeting_with_langchain, save_meeting
from ui.state import SESSION_MEETING_TRACE, _iso_now, get_tracer

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


