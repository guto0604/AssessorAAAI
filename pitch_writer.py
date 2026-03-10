import json

from langchain_core.prompts import ChatPromptTemplate

from langchain_runtime import build_runnable_config, get_chat_model, str_output_parser


def _build_api_metrics(response, *, provider: str = "openai") -> dict:
    usage = getattr(response, "usage", {}) or {}
    return {
        "provider": provider,
        "model": getattr(response, "model", None),
        "latency_ms": getattr(response, "elapsed_ms", None),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "response_id": getattr(response, "response_id", None),
    }


def generate_final_pitch_step7(
    cliente_info: dict,
    prompt_assessor: str,
    jornada_selecionada: dict,
    step5_selection: dict,
    model: str = "gpt-5.1",
    trace_context: dict | None = None,
    include_api_metrics: bool = False,
):
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

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_payload}")]
    )
    llm = get_chat_model(model=model, temperature=1)
    config = build_runnable_config(
        run_name="pitch_step_7_writer",
        tags=["pitch", "step_7", "langchain"],
        metadata={
            "feature": "pitch",
            "step": "step_7",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    messages = prompt.invoke({"user_payload": json.dumps(user_payload, ensure_ascii=False)}, config=config)
    response = llm.invoke(messages, config=config)
    pitch_text = str_output_parser.invoke(response, config=config).strip()

    if include_api_metrics:
        return {
            "text": pitch_text,
            "api_metrics": _build_api_metrics(response),
        }

    return pitch_text


def revise_pitch_step8(
    current_pitch: str,
    edit_instruction: str,
    target_excerpt: str | None = None,
    model: str = "gpt-5-mini",
    trace_context: dict | None = None,
    include_api_metrics: bool = False,
):
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
        "instrucao_de_edicao": edit_instruction,
    }

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_payload}")]
    )
    llm = get_chat_model(model=model, temperature=1)
    config = build_runnable_config(
        run_name="pitch_step_8_reviser",
        tags=["pitch", "step_8", "langchain"],
        metadata={
            "feature": "pitch",
            "step": "step_8",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    messages = prompt.invoke({"user_payload": json.dumps(user_prompt, ensure_ascii=False)}, config=config)
    response = llm.invoke(messages, config=config)
    revised_text = str_output_parser.invoke(response, config=config).strip()

    if include_api_metrics:
        return {
            "text": revised_text,
            "api_metrics": _build_api_metrics(response),
        }

    return revised_text
