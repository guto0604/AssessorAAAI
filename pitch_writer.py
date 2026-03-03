import json
from datetime import datetime, timezone
from time import perf_counter

from openai_client import get_openai_client


def _usage_dict(response):
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)

    if input_tokens is None:
        input_tokens = prompt_tokens
    if output_tokens is None:
        output_tokens = completion_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_final_pitch_step7(
    cliente_info: dict,
    prompt_assessor: str,
    jornada_selecionada: dict,
    step5_selection: dict,
    model: str = "gpt-5.1",
    trace_context: dict | None = None,
):
    """
    Gera o pitch final com base na seleção do assessor.
    """
    system_prompt = """
Você é um assessor de investimentos escrevendo uma mensagem para um cliente.

Objetivo:
- Escrever um pitch final pronto para envio (WhatsApp/Email curto), em português do Brasil.
- Seguir o tom e o tamanho selecionados.
- Usar somente os pontos aprovados (seleção do assessor).
- Ser claro, humano e direto, sem agressividade.

Regras:
- Não invente dados numéricos de rentabilidade de produtos que não estejam explicitamente nos insumos.
- Se houver produtos sugeridos com Produto_ID, cite o nome do produto de forma natural (não precisa mencionar o ID).
- Evite jargões excessivos e excesso de promessas.
- Se existirem objeções/respostas selecionadas, incorpore de forma sutil (uma ou duas frases) para reduzir fricção.
- Responda APENAS com o texto final do pitch (sem JSON, sem markdown, sem explicações).
"""

    user_payload = {
        "cliente": {
            "nome": cliente_info.get("Nome"),
            "perfil": cliente_info.get("Perfil_Suitability"),
            "patrimonio_conosco": cliente_info.get("Patrimonio_Investido_Conosco"),
            "dinheiro_para_investir": cliente_info.get("Dinheiro_Disponivel_Para_Investir"),
            "rentabilidade_12_meses": cliente_info.get("Rentabilidade_12_meses"),
            "cdi_12_meses": cliente_info.get("CDI_12_Meses"),
        },
        "prompt_inicial_assessor": prompt_assessor,
        "jornada_escolhida": jornada_selecionada,
        "selecoes_aprovadas": step5_selection,
    }

    call_start_iso = _iso_now()
    call_start_perf = perf_counter()
    resp = get_openai_client().chat.completions.create(
        model=model,
        temperature=1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    call_end_iso = _iso_now()
    call_duration_s = perf_counter() - call_start_perf

    content = resp.choices[0].message.content.strip()

    if trace_context:
        tracer = trace_context.get("tracer")
        parent_run_id = trace_context.get("parent_run_id")
        if tracer and parent_run_id:
            tracer.log_child_run(
                parent_run_id,
                name="pitch_step_7_writer_llm",
                run_type="llm",
                inputs={
                    "model": model,
                    "temperature": 1,
                    "system_prompt": system_prompt,
                    "user_payload": user_payload,
                },
                outputs={
                    "response": content,
                    "model_used": getattr(resp, "model", model),
                    "openai_latency_seconds": round(call_duration_s, 4),
                    "usage": _usage_dict(resp),
                },
                metadata={"step": "step_7"},
                tags=["pitch", "llm", "step_7"],
                start_time=call_start_iso,
                end_time=call_end_iso,
            )

    return content


def revise_pitch_step8(
    current_pitch: str,
    edit_instruction: str,
    target_excerpt: str | None = None,
    model: str = "gpt-5-mini",
    trace_context: dict | None = None,
):
    """
    Ajuste iterativo do pitch (Passo 8 simples).
    Se target_excerpt vier preenchido, o modelo prioriza editar aquele trecho.
    """
    system_prompt = """
Você é um revisor de texto comercial para assessoria de investimentos.

Regras:
- Mantenha o mesmo objetivo e contexto do pitch.
- Preserve o restante do texto o máximo possível.
- Aplique exatamente a instrução do assessor.
- Retorne APENAS o pitch revisado (sem markdown, sem explicações).
"""

    user_prompt = {
        "pitch_atual": current_pitch,
        "trecho_alvo": target_excerpt,
        "instrucao_de_edicao": edit_instruction
    }

    call_start_iso = _iso_now()
    call_start_perf = perf_counter()
    resp = get_openai_client().chat.completions.create(
        model=model,
        temperature=1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    )
    call_end_iso = _iso_now()
    call_duration_s = perf_counter() - call_start_perf
    revised = resp.choices[0].message.content.strip()

    if trace_context:
        tracer = trace_context.get("tracer")
        parent_run_id = trace_context.get("parent_run_id")
        if tracer and parent_run_id:
            tracer.log_child_run(
                parent_run_id,
                name="pitch_step_8_reviser_llm",
                run_type="llm",
                inputs={
                    "model": model,
                    "temperature": 1,
                    "system_prompt": system_prompt,
                    "user_payload": user_prompt,
                },
                outputs={
                    "response": revised,
                    "model_used": getattr(resp, "model", model),
                    "openai_latency_seconds": round(call_duration_s, 4),
                    "usage": _usage_dict(resp),
                },
                metadata={"step": "step_8"},
                tags=["pitch", "llm", "step_8"],
                start_time=call_start_iso,
                end_time=call_end_iso,
            )

    return revised
