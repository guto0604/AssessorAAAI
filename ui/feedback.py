import streamlit as st

from ui.state import (
    SESSION_MEETING_TRACE,
    SESSION_PITCH_TRACE,
    get_screen_feedback,
    get_screen_run,
    get_tracer,
    register_screen_feedback,
)


def _resolve_screen_run(screen_key: str) -> dict | None:
    """Resolve o run mais recente da tela para vincular feedback ao LangSmith."""
    if screen_key == "pitch":
        return st.session_state.get(SESSION_PITCH_TRACE)
    if screen_key == "meetings":
        return st.session_state.get(SESSION_MEETING_TRACE)
    return get_screen_run(screen_key)


def render_screen_feedback(screen_key: str, screen_label: str) -> None:
    """Renderiza botões de like/dislike associados ao último run da tela."""
    tracer = get_tracer()
    screen_run = _resolve_screen_run(screen_key) or {}
    run_id = screen_run.get("run_id")
    run_status = screen_run.get("status")
    feedback = get_screen_feedback(screen_key, run_id)
    current_score = feedback.get("score") if feedback else None

    st.divider()
    st.caption("Feedback da tela")

    if not tracer.enabled:
        st.info("Configure a LANGSMITH_API_KEY para registrar feedback no LangSmith.")
        return

    if not run_id:
        st.info("Execute uma ação nesta tela para habilitar o like/dislike associado ao run.")
        return

    if run_status == "in_progress":
        st.info("Aguarde a conclusão do processamento atual para enviar feedback desta tela.")
        return

    like_col, dislike_col = st.columns(2)
    with like_col:
        like_clicked = st.button(
            "👍 Like",
            key=f"{screen_key}_feedback_like",
            type="primary" if current_score is True else "secondary",
            use_container_width=True,
        )
    with dislike_col:
        dislike_clicked = st.button(
            "👎 Dislike",
            key=f"{screen_key}_feedback_dislike",
            type="primary" if current_score is False else "secondary",
            use_container_width=True,
        )

    selected_score = True if like_clicked else False if dislike_clicked else None
    if selected_score is not None:
        feedback_id = tracer.submit_feedback(
            run_id=run_id,
            score=selected_score,
            screen_key=screen_key,
            screen_label=screen_label,
            feedback_id=(feedback or {}).get("feedback_id"),
        )
        if feedback_id:
            register_screen_feedback(
                screen_key,
                run_id,
                feedback_id=feedback_id,
                score=selected_score,
            )
            current_score = selected_score
            st.success("Feedback registrado no LangSmith com sucesso.")
        else:
            st.error(tracer.last_error or "Não foi possível registrar o feedback no LangSmith.")
            return

    if current_score is True:
        st.caption("Último feedback enviado para este run: 👍 Like")
    elif current_score is False:
        st.caption("Último feedback enviado para este run: 👎 Dislike")
