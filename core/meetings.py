from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import json
from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool

from core.langchain_runtime import build_runnable_config, get_chat_model, str_output_parser
from core.openai_client import get_openai_client

BASE_DIR = Path(__file__).resolve().parent.parent
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


def save_meeting(cliente_id, cliente_nome, cliente_info, transcript, summary, api_calls: list[dict] | None = None) -> Path:
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


def transcribe_audio(file_bytes, filename, mime_type, trace_context: dict | None = None, include_api_metrics: bool = False):
    config = build_runnable_config(
        run_name="meeting_transcription",
        tags=["meeting", "transcription", "langchain"],
        metadata={
            "feature": "meeting",
            "mime_type": mime_type,
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )
    started = perf_counter()
    transcript = transcribe_audio_tool.invoke({"file_bytes": file_bytes, "filename": filename}, config=config)
    latency_ms = round((perf_counter() - started) * 1000, 2)

    if include_api_metrics:
        return {
            "text": transcript,
            "api_metrics": {
                "step": "transcription",
                "provider": "openai",
                "model": "gpt-4o-mini-transcribe",
                "latency_ms": latency_ms,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "response_id": None,
            },
        }
    return transcript


def summarize_transcript(cliente_info, transcript, trace_context: dict | None = None, include_api_metrics: bool = False):
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
    config = build_runnable_config(
        run_name="meeting_summary",
        tags=["meeting", "summary", "langchain"],
        metadata={
            "feature": "meeting",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    payload = json.dumps({"cliente_info": cliente_info, "transcricao": transcript}, ensure_ascii=False)
    messages = prompt.invoke(
        {"payload": payload},
        config=config,
    )
    response = llm.invoke(messages, config=config)
    summary = str_output_parser.invoke(response, config=config).strip()

    if include_api_metrics:
        usage = getattr(response, "usage", {}) or {}
        return {
            "text": summary,
            "api_metrics": {
                "step": "summary",
                "provider": "openai",
                "model": getattr(response, "model", None),
                "latency_ms": getattr(response, "elapsed_ms", None),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "response_id": getattr(response, "response_id", None),
                "prompt": {
                    "system": system_prompt.strip(),
                    "user": payload,
                },
                "output": summary,
            },
        }
    return summary


def process_meeting_with_langchain(
    cliente_info,
    audio_bytes,
    audio_name,
    audio_type,
    trace_context: dict | None = None,
    include_api_metrics: bool = False,
) -> dict:
    config = build_runnable_config(
        run_name="meeting_end_to_end",
        tags=["meeting", "langchain", "e2e"],
        metadata={
            "feature": "meeting",
            "audio_type": audio_type,
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    if not include_api_metrics:
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

    transcription_result = transcribe_audio(
        audio_bytes,
        audio_name,
        audio_type,
        trace_context=trace_context,
        include_api_metrics=True,
    )
    summary_result = summarize_transcript(
        cliente_info,
        transcription_result["text"],
        trace_context=trace_context,
        include_api_metrics=True,
    )

    return {
        "transcript": transcription_result["text"],
        "summary": summary_result["text"],
        "api_calls": [transcription_result["api_metrics"], summary_result["api_metrics"]],
    }
