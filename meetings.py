from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool

from langchain_runtime import build_runnable_config, get_chat_model, str_output_parser
from openai_client import get_openai_client

BASE_DIR = Path(__file__).resolve().parent
MEETINGS_DIR = BASE_DIR / "meetings"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_client_meetings_dir(cliente_id) -> Path:
    client_dir = MEETINGS_DIR / str(cliente_id)
    client_dir.mkdir(parents=True, exist_ok=True)
    return client_dir


def list_client_meetings(cliente_id) -> list[Path]:
    client_dir = ensure_client_meetings_dir(cliente_id)
    return sorted(client_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)


def _build_cliente_context(cliente_info: dict) -> str:
    perfil = cliente_info.get("Perfil_Suitability", "não mencionado")
    patrimonio = cliente_info.get("Patrimonio_Investido_Conosco", "não mencionado")
    disponivel = cliente_info.get("Dinheiro_Disponivel_Para_Investir", "não mencionado")
    rent = cliente_info.get("Rentabilidade_12_meses", "não mencionado")
    cdi = cliente_info.get("CDI_12_Meses", "não mencionado")

    return (
        f"Perfil suitability: {perfil}. "
        f"Patrimônio investido conosco: {patrimonio}. "
        f"Dinheiro disponível para investir: {disponivel}. "
        f"Rentabilidade 12m: {rent}. CDI 12m: {cdi}."
    )


def save_meeting(cliente_id, cliente_nome, cliente_info, transcript, summary) -> Path:
    client_dir = ensure_client_meetings_dir(cliente_id)
    now = datetime.now()
    file_name = f"{now.strftime('%Y-%m-%d_%H%M%S')}_reuniao.txt"
    file_path = client_dir / file_name

    title = f"Reunião com {cliente_nome} ({cliente_id}) – {now.strftime('%Y-%m-%d %H:%M')}"
    contexto = _build_cliente_context(cliente_info)

    body = (
        f"{title}\n\n"
        f"Contexto do cliente:\n{contexto}\n\n"
        f"{summary.strip()}\n\n"
        f"Transcrição:\n{transcript.strip()}\n"
    )

    file_path.write_text(body, encoding="utf-8")
    return file_path


@tool("transcribe_meeting_audio")
def transcribe_audio_tool(payload: dict) -> str:
    """Transcreve o áudio de uma reunião em português."""
    file_bytes = payload["file_bytes"]
    filename = payload.get("filename")
    audio_stream = BytesIO(file_bytes)
    audio_stream.name = filename or "audio_reuniao.wav"
    transcription = get_openai_client().audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_stream,
        language="pt",
    )
    return transcription.text.strip()


def transcribe_audio(file_bytes, filename, mime_type, trace_context: dict | None = None) -> str:
    config = build_runnable_config(
        run_name="meeting_transcription",
        tags=["meeting", "transcription", "langchain"],
        metadata={
            "feature": "meeting",
            "mime_type": mime_type,
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )
    return transcribe_audio_tool.invoke({"file_bytes": file_bytes, "filename": filename}, config=config)


def summarize_transcript(cliente_info, transcript, trace_context: dict | None = None) -> str:
    system_prompt = """
Você é um assistente para assessores de investimentos.

Tarefa:
- Resumir reunião em português do Brasil com base no contexto do cliente e na transcrição.
- Não inventar fatos. Se algo não estiver explícito, escreva "não mencionado".

Formato obrigatório de saída:
Principais tópicos:
- ...

Decisões/compromissos:
- ...

Próximos passos sugeridos para o assessor:
- ...
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{payload}")]
    )
    llm = get_chat_model(model="gpt-5-mini", temperature=1)
    chain = prompt | llm | str_output_parser

    config = build_runnable_config(
        run_name="meeting_summary",
        tags=["meeting", "summary", "langchain"],
        metadata={
            "feature": "meeting",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    return chain.invoke(
        {"payload": json.dumps({"cliente_info": cliente_info, "transcricao": transcript}, ensure_ascii=False)},
        config=config,
    ).strip()


def process_meeting_with_langchain(cliente_info, audio_bytes, audio_name, audio_type, trace_context: dict | None = None) -> dict:
    config = build_runnable_config(
        run_name="meeting_end_to_end",
        tags=["meeting", "langchain", "e2e"],
        metadata={
            "feature": "meeting",
            "audio_type": audio_type,
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {
            "transcript": transcribe_audio(x["audio_bytes"], x["audio_name"], x["audio_type"], trace_context=trace_context),
            "cliente_info": x["cliente_info"],
        })
        | RunnableLambda(lambda x: {
            "transcript": x["transcript"],
            "summary": summarize_transcript(x["cliente_info"], x["transcript"], trace_context=trace_context),
        })
    )
    return chain.invoke(
        {
            "cliente_info": cliente_info,
            "audio_bytes": audio_bytes,
            "audio_name": audio_name,
            "audio_type": audio_type,
        },
        config=config,
    )
