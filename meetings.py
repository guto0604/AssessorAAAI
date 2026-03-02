from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
MEETINGS_DIR = BASE_DIR / "meetings"


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


def transcribe_audio(file_bytes, filename, mime_type) -> str:
    audio_stream = BytesIO(file_bytes)
    audio_stream.name = filename or "audio_reuniao.wav"

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_stream,
        language="pt"
    )

    return transcription.text.strip()


def summarize_transcript(cliente_info, transcript) -> str:
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

    payload = {
        "cliente_info": cliente_info,
        "transcricao": transcript,
    }

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    return resp.choices[0].message.content.strip()
