import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from core.langchain_runtime import build_runnable_config, get_chat_model, parse_json_output, str_output_parser


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


def list_kb_files(kb_dir: str = "knowledge_base"):
    """Lista os elementos disponíveis para apoiar a navegação e seleção no fluxo.

    Args:
        kb_dir: Valor de entrada necessário para processar 'kb_dir'.

    Returns:
        Lista com os itens encontrados para a etapa solicitada.
    """
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return []
    return sorted([str(p.as_posix()) for p in kb_path.rglob("*.txt")])


def _sanitize_step4_output(parsed: dict) -> dict:
    """Responsável por sanitizar step4 output no contexto da aplicação de assessoria.

    Args:
        parsed: Valor de entrada necessário para processar 'parsed'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    parsed["data_sources"] = parsed.get("data_sources", [])
    parsed["products_selected_ids"] = parsed.get("products_selected_ids", [])
    parsed["kb_files_selected"] = parsed.get("kb_files_selected", [])[:5]
    parsed["reasoning_short"] = parsed.get("reasoning_short", "")
    return parsed


def select_sources_step4(
    cliente_info: dict,
    prompt_assessor: str,
    jornada_selecionada: dict,
    carteira_summary: dict,
    produtos_df,
    investimentos_cliente_df,
    kb_dir: str = "knowledge_base",
    model: str = "gpt-5-mini",
    trace_context: dict | None = None,
    include_api_metrics: bool = False,
):
    """Responsável por selecionar sources step4 no contexto da aplicação de assessoria.

    Args:
        cliente_info: Dicionário com os dados consolidados do cliente para personalizar a resposta.
        prompt_assessor: Valor de entrada necessário para processar 'prompt_assessor'.
        jornada_selecionada: Valor de entrada necessário para processar 'jornada_selecionada'.
        carteira_summary: Valor de entrada necessário para processar 'carteira_summary'.
        produtos_df: Valor de entrada necessário para processar 'produtos_df'.
        investimentos_cliente_df: Valor de entrada necessário para processar 'investimentos_cliente_df'.
        kb_dir: Valor de entrada necessário para processar 'kb_dir'.
        model: Modelo utilizado para executar a etapa 'model'.
        trace_context: Contexto de rastreio da execução para observabilidade.
        include_api_metrics: Indica se a função deve retornar métricas de uso de API junto ao resultado.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    kb_files = list_kb_files(kb_dir)

    produtos_catalogo = produtos_df[[
        "Produto_ID", "Nome_Produto", "Categoria", "Subcategoria", "Risco_Nivel", "Suitability_Ideal"
    ]].to_dict(orient="records")

    investimentos_list = investimentos_cliente_df[[
        "Produto", "Categoria", "Valor_Investido"
    ]].to_dict(orient="records")

    system_prompt = """
Você é um agente de seleção de fontes para um copiloto de assessoria de investimentos.

IMPORTANTE: Fluxo em duas etapas dentro desta mesma resposta:
1) Selecionar PRODUTOS candidatos (por Produto_ID) com base em jornada, prompt, perfil e momento (rentabilidade vs CDI, liquidez, gaps), selecione pelo menos 3 produtos.
2) Com base na jornada, prompt, cliente e produtos, escolher quais documentos .txt da knowledge_base fazem sentido CONSULTAR NO PRÓXIMO PASSO.
   - Você NÃO deve ler conteúdo dos documentos; selecione APENAS PELO NOME do arquivo.
   - Selecione exatamente 5 arquivos da knowledge_base.

Regras:
- Retorne APENAS JSON válido (sem markdown).
- Selecione entre 3 e 5 produtos candidatos (Produto_ID).
- Suitability não precisa ser exato, mas evite escolhas obviamente incompatíveis (ex: cripto para conservador), a menos que o prompt/jornada justifique.
- Sempre inclua as fontes de dados:
  - "investimentos_do_cliente"
  - "base_de_produtos"
  - "metricas_rentabilidade_carteira_e_cdi"

Formato obrigatório:
{
  "data_sources": ["..."],
  "products_selected_ids": ["P1","P2", "..."],
  "kb_files_selected": ["..."], 
  "reasoning_short": "..."
}
"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_payload}")]
    )
    llm = get_chat_model(model=model, temperature=1, response_format={"type": "json_object"})

    payload_builder = RunnableParallel(
        user_payload=RunnablePassthrough() | RunnableLambda(lambda x: json.dumps(x, ensure_ascii=False))
    )

    config = build_runnable_config(
        run_name="pitch_step_4_source_selector",
        tags=["pitch", "step_4", "langchain"],
        metadata={
            "feature": "pitch",
            "step": "step_4",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    payload = {
        "cliente_info": cliente_info,
        "prompt_assessor": prompt_assessor,
        "jornada_selecionada": jornada_selecionada,
        "carteira_summary": carteira_summary,
        "investimentos_atuais": investimentos_list,
        "produtos_catalogo": produtos_catalogo,
        "kb_files_available": kb_files,
    }
    messages = (payload_builder | prompt).invoke(payload, config=config)
    response = llm.invoke(messages, config=config)
    parsed = _sanitize_step4_output(parse_json_output(str_output_parser.invoke(response, config=config)))

    if include_api_metrics:
        return {
            "result": parsed,
            "api_metrics": _build_api_metrics(
                response,
                prompt={"messages": str(messages)},
                output=str_output_parser.invoke(response, config=config),
            ),
        }

    return parsed
