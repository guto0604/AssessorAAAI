import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from core.langchain_runtime import build_runnable_config, get_chat_model, parse_json_output, str_output_parser


def _build_api_metrics(response, *, provider: str = "openai", prompt: dict | None = None, output: str | None = None) -> dict:
    """ build api metrics.

    Args:
        response: Descrição do parâmetro `response`.
        provider: Descrição do parâmetro `provider`.
        prompt: Descrição do parâmetro `prompt`.
        output: Descrição do parâmetro `output`.

    Returns:
        Valor de retorno da função.
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
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if len(txt) > max_chars_each:
                    txt = txt[:max_chars_each] + "\n...[TRUNCADO]"
                docs.append({"path": path, "content": txt})
        except Exception as e:
            docs.append({"path": path, "content": f"[ERRO AO LER: {e}]"})
    return docs


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
    """Build pitch options step5.

    Args:
        cliente_info: Descrição do parâmetro `cliente_info`.
        prompt_assessor: Descrição do parâmetro `prompt_assessor`.
        jornada_selecionada: Descrição do parâmetro `jornada_selecionada`.
        carteira_summary: Descrição do parâmetro `carteira_summary`.
        investimentos_cliente_df: Descrição do parâmetro `investimentos_cliente_df`.
        produtos_selecionados_df: Descrição do parâmetro `produtos_selecionados_df`.
        kb_files_selected: Descrição do parâmetro `kb_files_selected`.
        model: Descrição do parâmetro `model`.
        trace_context: Descrição do parâmetro `trace_context`.
        include_api_metrics: Descrição do parâmetro `include_api_metrics`.

    Returns:
        Valor de retorno da função.
    """
    kb_docs = read_kb_files_tool.invoke({"file_paths": kb_files_selected, "max_chars_each": 3500})

    investimentos_list = investimentos_cliente_df[["Produto", "Categoria", "Valor_Investido"]].to_dict(orient="records")

    produtos_list = []
    if produtos_selecionados_df is not None and not produtos_selecionados_df.empty:
        cols = ["Produto_ID", "Nome_Produto", "Categoria", "Subcategoria", "Risco_Nivel", "Suitability_Ideal"]
        produtos_list = produtos_selecionados_df[cols].to_dict(orient="records")

    system_prompt = """
Você é um estrategista de conteúdo comercial para um assessor de investimentos.

Tarefa:
Com base em:
- contexto do cliente e carteira,
- prompt do assessor,
- jornada selecionada (com descrição),
- produtos candidatos (já pré-selecionados),
- documentos internos e research (conteúdo RAG),

Gere opções estruturadas e PRIORITÁRIAS (ordem importa) para o pitch.

Regras:
- Responda APENAS JSON válido (sem markdown).
- NÃO invente números de rentabilidade de produtos que não estejam nos dados. Se não existir, fale de forma qualitativa.
- Diagnóstico deve ser simples e legível.
- Sugira objeções com respostas curtas e úteis (pré-respostas).
- Produtos sugeridos devem referenciar Produto_ID quando possível (dos candidatos).
- Tom e tamanho: selecione 1 sugestão principal e até 2 alternativas.

QUANTIDADE (obrigatório):
- "diagnostico": 3 a 4 itens (curtos)
- "pontos_prioritarios": 3 a 5 itens
- "gatilhos_comerciais": 1 a 3 itens (inclua opções suaves/consultivas e opções mais diretas)
- "objecoes_e_respostas": 1 a 3 itens
- "produtos_sugeridos": sem limite de itens (preferir Produto_ID dos candidatos; se sugerir algo fora, explique e deixe produto_id vazio)
- "tom_sugerido": 1 principal + 3 alternativas
- "tamanho_pitch": 1 principal + 2 alternativas (Pequeno, Médio, Longo)

Formato obrigatório:

{
  "diagnostico": [
    {"id":"d1","texto":"..."}
  ],
  "pontos_prioritarios": [
    {"id":"p1","texto":"..."}
  ],
  "gatilhos_comerciais": [
    {"id":"g1","texto":"..."}
  ],
  "objecoes_e_respostas": [
    {"id":"o1","objecao":"...","resposta":"..."}
  ],
  "produtos_sugeridos": [
    {"id":"s1","produto_id":"P12","texto":"..."}
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

    messages = prompt.invoke({"user_payload": json.dumps(user_payload, ensure_ascii=False)}, config=config)
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
