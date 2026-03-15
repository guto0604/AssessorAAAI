import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from core.openai_client import get_openai_client
from ui.guardrails import (
    evaluate_input_guardrails,
    guardrail_warning_message,
    handle_guardrail_exception,
)
from ui.state import TALK_TO_DATA_TEMPLATE_DEFAULT_OPTION, _iso_now, get_tracer

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
TALK_TO_DATA_FILES = {
    "Informacoes_Cliente": DATA_DIR / "informacoes_cliente.parquet",
    "Investimentos_Cliente": DATA_DIR / "investimentos_cliente.parquet",
    "Produtos": DATA_DIR / "produtos.parquet",
}
REFERENCE_FILE_PATH = DATA_DIR / "referencia_base_dados.txt"
LOGGER = logging.getLogger(__name__)

def _reset_talk_to_data_state() -> None:
    st.session_state.talk_to_data_question = ""
    st.session_state.talk_to_data_last_llm_output = None
    st.session_state.talk_to_data_generated_sql = ""
    st.session_state.talk_to_data_saved_sql = ""
    st.session_state.talk_to_data_can_generate = True


def _reset_talk_to_data_page() -> None:
    try:
        _reset_talk_to_data_state()
        st.session_state["talk_to_data_next_question_feedback"] = {
            "status": "success",
            "message": "Tela reiniciada. Faça sua próxima pergunta.",
        }
    except Exception as exc:
        st.session_state["talk_to_data_next_question_feedback"] = {
            "status": "error",
            "message": (
                "Erro ao atualizar para a próxima pergunta. "
                "Selecione novamente a checkbox para funcionar."
            ),
            "details": str(exc),
        }


def _render_talk_to_data_samples() -> None:

    for table_name, file_path in TALK_TO_DATA_FILES.items():
        with st.expander(f"📦 Prévia tabela {table_name}", expanded=False):
            try:
                table_df = pd.read_parquet(file_path)
            except Exception as exc:
                st.error(f"Não foi possível carregar {table_name}: {exc}")
                continue

            sample_size = min(6, len(table_df))
            if sample_size == 0:
                st.info("Tabela vazia.")
                continue

            sampled_df = table_df.sample(n=sample_size, random_state=42)
            st.dataframe(sampled_df, width="stretch")


def _apply_talk_to_data_template_question(question: str) -> None:
    _reset_talk_to_data_page()
    st.session_state.talk_to_data_question = question


def render_talk_to_your_data_page():
    tracer = get_tracer()

    sample_questions = {
        "Perguntas Cliente": [
            "Mostre a distribuição da carteira do cliente A_001.",
            "Quais os clientes que resgataram mais que aportaram nos últimos 12 meses que não fazemos contato a mais de 50 dias",
            "Quais clientes têm dinheiro disponível para investir acima de 1 milhão?",
        ],
        "Perguntas sobre investimentos": [
            "Qual é a distribuição da carteira por categoria de investimento?",
            "Quais clientes possuem maior exposição em renda variável?",
        ],
        "Perguntas sobre produtos": [
            "Quais produtos em campanha são adequados para clientes moderados?",
        ],
        "Perguntas cruzando clientes + investimentos": [
            "Quais clientes com perfil arrojado e mais de 1 milhão investidos tem exposição alta em cripto?",
        ],
        "Pergunta mais analítica / avançada": [
            "Quais clientes têm maior risco de churn e possuem mais de 50 mil disponíveis para investir?",
        ],
        "Perguntas com maior potencial para visualização": [
            "Quero um gráfico de dispersão mostrando a relação de salário e total investido.",
            "Mostre a distribuição da carteira dos clientes arrojados."
        ],
    }

    st.title("Talk to your Data")
    st.caption("Faça perguntas em linguagem natural, e visualize os resultados.")

    feedback = st.session_state.pop("talk_to_data_next_question_feedback", None)
    if feedback:
        if feedback.get("status") == "error":
            st.error(feedback.get("message", "Erro ao atualizar a tela."))
            if feedback.get("details"):
                st.caption(f"Detalhes técnicos: {feedback['details']}")
        else:
            st.success(feedback.get("message", "Tela atualizada."))

    _render_talk_to_data_samples()

    dropdown_options = [TALK_TO_DATA_TEMPLATE_DEFAULT_OPTION]
    for category, questions in sample_questions.items():
        for sample_question in questions:
            dropdown_options.append(f"{category} — {sample_question}")

    selected_template = st.selectbox(
        "Escolha uma pergunta modelo ou escreva sua própria pergunta ",
        options=dropdown_options,
        key="talk_to_data_template_dropdown",
    )

    if selected_template != TALK_TO_DATA_TEMPLATE_DEFAULT_OPTION:
        selected_question = selected_template.split(" — ", 1)[1]
        if selected_question != st.session_state.get("talk_to_data_question", ""):
            _apply_talk_to_data_template_question(selected_question)
            st.rerun()

    question = st.text_area(
        "Pergunte sobre a base de assessoria:",
        placeholder="Ex.: Quais clientes fazem aniversário neste mês?",
        key="talk_to_data_question",
        height=100,
    )

    controls_col_1, controls_col_2 = st.columns(2)
    with controls_col_1:
        generate_pressed = st.button(
            "Gerar consulta",
            key="talk_to_data_submit",
            disabled=not st.session_state.get("talk_to_data_can_generate", True),
        )

    with controls_col_2:
        if st.button("➡️ Próxima pergunta", key="talk_to_data_next_question"):
            _reset_talk_to_data_page()
            st.rerun()

    if generate_pressed:
        if not question.strip():
            st.warning("Escreva uma pergunta antes de enviar.")
            return

        talk_to_data_run_id = tracer.start_run(
            name=f"talk_to_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            run_type="chain",
            inputs={"question": question.strip()},
            tags=["talk_to_data", "streamlit"],
            metadata={"started_at": _iso_now()},
        )

        try:
            try:
                guardrail_result = evaluate_input_guardrails(question.strip())
            except Exception as exc:
                guardrail_result = handle_guardrail_exception(question.strip(), exc)

            tracer.log_event(
                talk_to_data_run_id,
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
                    talk_to_data_run_id,
                    status="blocked",
                    outputs={
                        "status": "blocked",
                        "guardrail": {
                            "violation_type": guardrail_result.violation_type,
                            "reason": guardrail_result.message,
                        },
                    },
                )
                st.warning(guardrail_warning_message(guardrail_result.violation_type))
                return

            reference_text = load_reference_text()
            prompt = build_llm_prompt(question=question.strip(), reference_text=reference_text)
            tracer.log_event(talk_to_data_run_id, "talk_to_data_llm_call_started")
            llm_started_at = time.perf_counter()
            llm_result = ask_talk_to_data_llm(prompt, include_api_metrics=True)
            llm_output = llm_result["output"]
            api_metrics = llm_result.get("api_metrics", {})
            call_duration_ms = int((time.perf_counter() - llm_started_at) * 1000)

            tracer.log_event(
                talk_to_data_run_id,
                "talk_to_data_api_call",
                {
                    "provider": "openai",
                    "model": api_metrics.get("model"),
                    "input_tokens": api_metrics.get("input_tokens"),
                    "output_tokens": api_metrics.get("output_tokens"),
                    "total_tokens": api_metrics.get("total_tokens"),
                    "duration_ms": call_duration_ms,
                },
            )
            tracer.log_event(
                talk_to_data_run_id,
                "talk_to_data_llm_call_completed",
                {
                    "can_answer": bool(llm_output.get("can_answer", False)),
                    "has_sql": bool((llm_output.get("sql") or "").strip()),
                    "visualization_type": (llm_output.get("visualization") or {}).get("type", "none"),
                },
            )
            tracer.end_run(
                talk_to_data_run_id,
                status="success",
                outputs={
                    "status": "success",
                    "can_answer": bool(llm_output.get("can_answer", False)),
                    "api_metrics": {
                        **api_metrics,
                        "duration_ms": call_duration_ms,
                    },
                },
            )
        except Exception as exc:
            tracer.log_event(talk_to_data_run_id, "talk_to_data_error", {"error": str(exc)})
            tracer.end_run(
                talk_to_data_run_id,
                status="error",
                error=str(exc),
                outputs={"status": "error"},
            )
            st.error(f"Falha ao interpretar a pergunta com a LLM: {exc}")
            return

        st.session_state.talk_to_data_last_llm_output = llm_output
        st.session_state.talk_to_data_can_generate = False

    llm_output = st.session_state.get("talk_to_data_last_llm_output") or {}
    if not llm_output:
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
        return

    if not sql:
        st.warning("A pergunta foi marcada como respondível, mas nenhum SQL foi retornado.")
        return

    if not st.session_state.talk_to_data_generated_sql:
        st.session_state.talk_to_data_generated_sql = sql

    st.subheader("Consulta SQL (permitida edição)")
    generated_sql = st.text_area(
        "SQL gerado",
        key="talk_to_data_generated_sql",
        height=220,
    )

    col_save, col_clear = st.columns([1, 1])
    with col_save:
        if st.button("💾 Salvar consulta", key="talk_to_data_save_sql"):
            if not generated_sql.strip():
                st.warning("Não é possível salvar uma consulta vazia.")
            else:
                st.session_state.talk_to_data_saved_sql = generated_sql.strip()
                st.success("Consulta salva. Agora você pode executá-la quando quiser.")

    with col_clear:
        if st.button("🧹 Limpar consulta salva", key="talk_to_data_clear_saved_sql"):
            st.session_state.talk_to_data_saved_sql = ""
            st.info("Consulta salva removida.")

    saved_sql = (st.session_state.get("talk_to_data_saved_sql") or "").strip()
    if not saved_sql:
        st.info("Salve a consulta para habilitar a execução.")
        return

    with st.expander("Consulta salva", expanded=False):
        st.code(saved_sql, language="sql")

    if st.button("▶️ Executar consulta salva", key="talk_to_data_execute_saved_sql"):
        result_df = pd.DataFrame()
        query_error: Exception | None = None
        final_sql = ""
        try:
            final_sql = sanitize_duckdb_sql(saved_sql)
            validate_read_only_sql(final_sql)
            result_df = run_duckdb_query(final_sql)
        except Exception as exc:
            query_error = exc

        if query_error:
            st.error("Erro ao executar SQL no DuckDB.")
            st.exception(query_error)
            return

        st.subheader("Resultado da consulta")
        if result_df.empty:
            st.info("A consulta foi executada, mas não retornou linhas.")
        else:
            st.dataframe(result_df, width="stretch")

        st.subheader("Resposta final")
        st.write(answer)
        render_visual(result_df, visualization)



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
- Priorize SQL portátil para DuckDB: evite window functions (OVER/PARTITION BY), funções analíticas avançadas e recursos menos comuns quando houver alternativa.
- Quando possível, substitua window functions por CTEs com agregações e JOINs simples.
- Sempre use uma data explícita no SQL no formato DATE 'YYYY-MM-DD'.
- Nunca use funções de data/hora atual (CURRENT_DATE, NOW, CURRENT_TIMESTAMP, TODAY), a data atual é "{date.today().isoformat()}"
- Use DATE '{date.today().isoformat()}' como data atual.
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
    "type": "bar|line|pie|scatter|table|none",
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


def ask_talk_to_data_llm(prompt: str, include_api_metrics: bool = False) -> dict:
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-5.1",
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

    parsed_output = json.loads(content)
    if not include_api_metrics:
        return parsed_output

    usage = response.usage or {}
    return {
        "output": parsed_output,
        "api_metrics": {
            "provider": "openai",
            "model": response.model,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


def sanitize_duckdb_sql(sql: str) -> str:
    normalized_sql = sql.strip()
    normalized_sql = re.sub(r";\s*$", "", normalized_sql)
    normalized_sql = normalized_sql.replace("`", "")
    system_date_sql = f"DATE '{date.today().isoformat()}'"
    normalized_sql = re.sub(
        r"strftime\(\s*'%m'\s*,\s*([^)]+?)\s*\)\s*=\s*strftime\(\s*'%m'\s*,\s*'now'\s*\)",
        rf"EXTRACT(MONTH FROM \1) = EXTRACT(MONTH FROM {system_date_sql})",
        normalized_sql,
        flags=re.IGNORECASE,
    )
    normalized_sql = re.sub(
        r"strftime\(\s*'%m'\s*,\s*'now'\s*\)",
        f"EXTRACT(MONTH FROM {system_date_sql})",
        normalized_sql,
        flags=re.IGNORECASE,
    )
    normalized_sql = re.sub(
        r"\bCURRENT_DATE\b|\bCURRENT_TIMESTAMP\b|\bNOW\s*\(\s*\)|\bTODAY\s*\(\s*\)",
        system_date_sql,
        normalized_sql,
        flags=re.IGNORECASE,
    )
    return normalized_sql.strip()


def validate_read_only_sql(sql: str) -> None:
    if not sql:
        raise ValueError("A consulta SQL está vazia.")
    if ";" in sql:
        raise ValueError("Apenas uma instrução SQL é permitida.")
    if not re.match(r"^(select|with)\b", sql, flags=re.IGNORECASE):
        raise ValueError("Apenas consultas SELECT/CTE de leitura são permitidas.")

    blocked_keywords = (
        "insert", "update", "delete", "drop", "alter", "create", "replace", "truncate",
        "attach", "detach", "copy", "call", "grant", "revoke", "merge", "execute", "prepare",
    )
    blocked_pattern = r"\b(" + "|".join(blocked_keywords) + r")\b"
    if re.search(blocked_pattern, sql, flags=re.IGNORECASE):
        raise ValueError("A consulta contém comandos bloqueados para segurança.")


def _create_talk_to_data_views(con: duckdb.DuckDBPyConnection) -> None:
    for table_name, file_path in TALK_TO_DATA_FILES.items():
        escaped_path = str(file_path).replace("'", "''")
        con.execute(f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM read_parquet(\'{escaped_path}\')')


def run_duckdb_query(sql: str, conn: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    normalized_sql = sanitize_duckdb_sql(sql)
    validate_read_only_sql(normalized_sql)
    LOGGER.info("Executando SQL DuckDB: %s", normalized_sql)

    if conn is not None:
        _create_talk_to_data_views(conn)
        return conn.execute(normalized_sql).fetchdf()

    with duckdb.connect(database=":memory:") as con:
        _create_talk_to_data_views(con)
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

    if vis_type in {"bar", "line", "pie", "scatter"} and y and y not in result_df.columns:
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
        st.plotly_chart(fig, width="stretch")
    elif vis_type == "scatter":
        if not x or not y:
            st.info("Visual de dispersão requer campos x e y válidos.")
            return
        fig = px.scatter(result_df, x=x, y=y, title=title)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Tipo de visual não suportado; exibindo tabela.")
        st.dataframe(result_df, width="stretch")

