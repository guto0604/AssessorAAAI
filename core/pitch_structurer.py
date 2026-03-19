import json
import re
import unicodedata

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

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


@tool("read_kb_files")
def read_kb_files_tool(payload: dict) -> list[dict]:
    """Lê conteúdo dos .txt selecionados (RAG simples)."""
    file_paths = payload.get("file_paths", [])
    max_chars_each = payload.get("max_chars_each", 3500)
    docs = []
    for path in file_paths:
        normalized = str(path or "").strip()
        if not normalized.endswith(".txt"):
            continue
        try:
            with open(normalized, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if len(txt) > max_chars_each:
                    txt = txt[:max_chars_each] + "\n...[TRUNCADO]"
                docs.append({"path": normalized, "content": txt})
        except Exception as e:
            docs.append({"path": normalized, "content": f"[ERRO AO LER: {e}]"})
    return docs


def _slugify_topic(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return normalized or "topico"


def _build_topics_instructions(topics: list[str]) -> str:
    instructions = []
    for index, topic in enumerate(topics, start=1):
        block_id = _slugify_topic(topic)
        instructions.append(
            f'- Bloco {index}: use exatamente o título "{topic}" com id "{block_id}". '
            f'Os itens desse bloco devem usar ids como "{block_id}_i1", "{block_id}_i2" e assim por diante.'
        )
    return "\n".join(instructions)


def build_pitch_options_step5(
    cliente_info: dict,
    prompt_assessor: str,
    jornada_selecionada: dict,
    carteira_summary: dict,
    investimentos_cliente_df,
    produtos_selecionados_df,
    kb_files_selected: list[str],
    model: str = "gpt-5.1",
    trace_context: dict | None = None,
    include_api_metrics: bool = False,
):
    """Monta a estrutura de dados usada nas próximas etapas do fluxo.

    Args:
        cliente_info: Dicionário com os dados consolidados do cliente para personalizar a resposta.
        prompt_assessor: Valor de entrada necessário para processar 'prompt_assessor'.
        jornada_selecionada: Valor de entrada necessário para processar 'jornada_selecionada'.
        carteira_summary: Valor de entrada necessário para processar 'carteira_summary'.
        investimentos_cliente_df: Valor de entrada necessário para processar 'investimentos_cliente_df'.
        produtos_selecionados_df: Valor de entrada necessário para processar 'produtos_selecionados_df'.
        kb_files_selected: Caminho ou arquivo de entrada relacionado a 'kb_files_selected'.
        model: Modelo utilizado para executar a etapa 'model'.
        trace_context: Contexto de rastreio da execução para observabilidade.
        include_api_metrics: Indica se a função deve retornar métricas de uso de API junto ao resultado.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    """
    kb_docs = read_kb_files_tool.invoke({"file_paths": kb_files_selected, "max_chars_each": 3500})


    tracer = (trace_context or {}).get("tracer")
    parent_run_id = (trace_context or {}).get("parent_run_id")
    if tracer and parent_run_id:
        tracer.log_event(
            parent_run_id,
            "pitch_step_5_documents_consulted",
            {
                "documents": [doc.get("path") for doc in kb_docs],
                "total_documents": len(kb_docs),
            },
        )

    investimentos_list = investimentos_cliente_df[["Produto", "Categoria", "Valor_Investido"]].to_dict(orient="records")

    produtos_list = []
    if produtos_selecionados_df is not None and not produtos_selecionados_df.empty:
        cols = ["Produto_ID", "Nome_Produto", "Categoria", "Subcategoria", "Risco_Nivel", "Suitability_Ideal"]
        produtos_list = produtos_selecionados_df[cols].to_dict(orient="records")

    topicos_llm = list(jornada_selecionada.get("topicos_llm") or [])
    if not topicos_llm:
        raise ValueError("A jornada selecionada não possui tópicos configurados em 'Topicos_LLM'.")

    system_prompt = """
Você é um estrategista de conteúdo comercial para um assessor de investimentos.

Tarefa:
Com base em:
- contexto do cliente e carteira,
- prompt do assessor,
- jornada selecionada (com descrição),
- tópicos configurados para a jornada,
- produtos candidatos (já pré-selecionados),
- documentos internos e research (conteúdo RAG),

Gere opções estruturadas e PRIORITÁRIAS (ordem importa) para o pitch.

Regras:
- Responda APENAS JSON válido (sem markdown).
- NÃO invente números de rentabilidade de produtos que não estejam nos dados. Se não existir, fale de forma qualitativa.
- Gere APENAS os blocos listados em `topicos_llm`, na MESMA ORDEM e com o MESMO título informado.
- Não crie blocos extras além dos tópicos configurados para a jornada.
- Cada bloco deve conter de 2 a 4 itens curtos e acionáveis. Se um tópico estiver pouco suportado pelos dados, retorne 1 item curto explicando a cautela.
- Tom e tamanho: selecione 1 sugestão principal e até 2 alternativas.

ESTRUTURA DOS BLOCOS DINÂMICOS (obrigatório):
{topics_instructions}

QUANTIDADE (obrigatório):
- "blocos_conteudo": exatamente 1 bloco por tópico configurado
- "tom_sugerido": 1 principal + 3 alternativas
- "tamanho_pitch": 1 principal + 2 alternativas (Pequeno, Médio, Longo)

Formato obrigatório:

{
  "blocos_conteudo": [
    {
      "id":"identificador_do_topico",
      "titulo":"titulo_exato_do_topico",
      "itens":[
        {"id":"identificador_do_topico_i1","texto":"..."}
      ]
    }
  ],
  "tom_sugerido": {
    "principal":{"id":"t1","texto":"..."},
    "alternativas":[{"id":"t2","texto":"..."},{"id":"t3","texto":"..."}]
  },
  "tamanho_pitch": {
    "principal":{"id":"l2","texto":"Pequeno|Médio|Longo"},
    "alternativas":[{"id":"l1","texto":"..."},{"id":"l3","texto":"..."}]
  }
}
"""

    user_payload = {
        "cliente_info": cliente_info,
        "prompt_assessor": prompt_assessor,
        "jornada_selecionada": jornada_selecionada,
        "carteira_summary": carteira_summary,
        "investimentos_atuais": investimentos_list,
        "produtos_candidatos": produtos_list,
        "kb_context": kb_docs,
    }

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_payload}")]
    )
    llm = get_chat_model(model=model, temperature=1, response_format={"type": "json_object"})
    config = build_runnable_config(
        run_name="pitch_step_5_structurer",
        tags=["pitch", "step_5", "langchain"],
        metadata={
            "feature": "pitch",
            "step": "step_5",
            "parent_run_id": (trace_context or {}).get("parent_run_id"),
        },
    )

    messages = prompt.invoke(
        {
            "user_payload": json.dumps(user_payload, ensure_ascii=False),
            "topics_instructions": _build_topics_instructions(topicos_llm),
        },
        config=config,
    )
    response = llm.invoke(messages, config=config)
    parsed = parse_json_output(str_output_parser.invoke(response, config=config))

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
