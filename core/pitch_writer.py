import json

from langchain_core.prompts import ChatPromptTemplate

from core.langchain_runtime import build_runnable_config, get_chat_model, str_output_parser


def _build_api_metrics(response, *, provider: str = "openai", prompt: dict | None = None, output: str | None = None) -> dict:
    """Responsável por montar api metrics no contexto da aplicação de assessoria.

    Args:
        response: Valor de entrada necessário para processar 'response'.
        provider: Identificador usado para referenciar 'provider'.
        prompt: Valor de entrada necessário para processar 'prompt'.
        output: Valor de entrada necessário para processar 'output'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    usage = getattr(response, "usage", {}) or {}
    return {
        "provider": provider,
        "model": getattr(response, "model", None),
        "latency_ms": getattr(response, "elapsed_ms", None),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "response_id": getattr(response, "response_id", None),
        "prompt": prompt or {},
        "output": output,
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
    """Executa uma etapa de construção do pitch comercial personalizado para o cliente.

    Args:
        cliente_info: Dicionário com os dados consolidados do cliente para personalizar a resposta.
        prompt_assessor: Valor de entrada necessário para processar 'prompt_assessor'.
        jornada_selecionada: Valor de entrada necessário para processar 'jornada_selecionada'.
        step5_selection: Valor de entrada necessário para processar 'step5_selection'.
        model: Modelo utilizado para executar a etapa 'model'.
        trace_context: Contexto de rastreio da execução para observabilidade.
        include_api_metrics: Indica se a função deve retornar métricas de uso de API junto ao resultado.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
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
- Use como base apenas os blocos de conteúdo aprovados em `selecoes_aprovadas.blocos_conteudo`.
- Não introduza novas seções ou argumentos fora do que foi aprovado para a jornada selecionada.
- Evite jargões excessivos e excesso de promessas.
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
            "api_metrics": _build_api_metrics(
                response,
                prompt={"messages": str(messages)},
                output=str_output_parser.invoke(response, config=config),
            ),
        }

    return pitch_text


def generate_prompt_to_pitch(
    cliente_info: dict,
    prompt_assessor: str,
    model: str = "gpt-5.1",
    trace_context: dict | None = None,
    include_api_metrics: bool = False,
):
    """Gera um pitch direto a partir do prompt do assessor, sem etapas intermediárias."""
    system_prompt = """
Você é um assessor de investimentos escrevendo uma mensagem comercial final para um cliente.

Objetivo:
- Gerar um pitch final pronto para envio em português do Brasil.
- Partir apenas do contexto do cliente e do pedido livre do assessor.
- Não depender de etapas intermediárias, estruturações ou aprovações humanas.

Regras:
- Responda APENAS com o texto final do pitch.
- Seja claro, humano, objetivo e comercialmente útil.
- Não invente números, retornos, garantias ou características não presentes nos insumos.
- Se o prompt pedir tom, canal, tamanho ou foco, siga essas instruções.
- Quando o prompt não especificar tamanho, prefira uma mensagem curta a média.
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
    }

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_payload}")]
    )
    llm = get_chat_model(model=model, temperature=1)
    config = build_runnable_config(
        run_name="pitch_prompt_to_pitch_writer",
        tags=["pitch", "prompt-to-pitch", "langchain"],
        metadata={
            "feature": "pitch",
            "step": "prompt_to_pitch",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    messages = prompt.invoke({"user_payload": json.dumps(user_payload, ensure_ascii=False)}, config=config)
    response = llm.invoke(messages, config=config)
    pitch_text = str_output_parser.invoke(response, config=config).strip()

    if include_api_metrics:
        return {
            "text": pitch_text,
            "api_metrics": _build_api_metrics(
                response,
                prompt={"messages": str(messages)},
                output=str_output_parser.invoke(response, config=config),
            ),
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
    """Executa uma etapa de construção do pitch comercial personalizado para o cliente.

    Args:
        current_pitch: Valor de entrada necessário para processar 'current_pitch'.
        edit_instruction: Valor de entrada necessário para processar 'edit_instruction'.
        target_excerpt: Valor de entrada necessário para processar 'target_excerpt'.
        model: Modelo utilizado para executar a etapa 'model'.
        trace_context: Contexto de rastreio da execução para observabilidade.
        include_api_metrics: Indica se a função deve retornar métricas de uso de API junto ao resultado.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
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
            "api_metrics": _build_api_metrics(
                response,
                prompt={"messages": str(messages)},
                output=str_output_parser.invoke(response, config=config),
            ),
        }

    return revised_text
