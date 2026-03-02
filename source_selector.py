import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def list_kb_files(kb_dir: str = "knowledge_base"):
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return []
    return sorted([str(p.as_posix()) for p in kb_path.rglob("*.txt")])

def select_sources_step4(
    cliente_info: dict,
    prompt_assessor: str,
    jornada_selecionada: dict,
    carteira_summary: dict,
    produtos_df,
    investimentos_cliente_df,
    kb_dir: str = "knowledge_base",
    model: str = "gpt-5-mini"
):
    kb_files = list_kb_files(kb_dir)

    produtos_catalogo = produtos_df[[
        "Produto_ID", "Nome_Produto", "Categoria", "Subcategoria", "Risco_Nivel (1-5)", "Suitability_Ideal"
    ]].to_dict(orient="records")

    investimentos_list = investimentos_cliente_df[[
        "Produto", "Categoria", "Valor_Investido"
    ]].to_dict(orient="records")

    system_prompt = """
Você é um agente de seleção de fontes para um copiloto de assessoria de investimentos.

IMPORTANTE: Fluxo em duas etapas dentro desta mesma resposta:
1) Selecionar PRODUTOS candidatos (por Produto_ID) com base em jornada, prompt, perfil e momento (rentabilidade vs CDI, liquidez, gaps), selecione pelo menos 3 produtos.
2) Com base nos PRODUTOS selecionados (categoria/subcategoria/risco), escolher quais documentos .txt da knowledge_base fazem sentido CONSULTAR NO PRÓXIMO PASSO.
   - Você NÃO deve ler conteúdo dos documentos; selecione APENAS PELO NOME do arquivo.
   - Selecione no máximo 5 arquivos da knowledge_base.

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

    user_payload = {
        "cliente_info": cliente_info,
        "prompt_assessor": prompt_assessor,
        "jornada_selecionada": jornada_selecionada,
        "carteira_summary": carteira_summary,
        "investimentos_atuais": investimentos_list,
        "produtos_catalogo": produtos_catalogo,
        "kb_files_available": kb_files
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
    )

    parsed = json.loads(resp.choices[0].message.content)

    # saneamento
    parsed["data_sources"] = parsed.get("data_sources", [])
    parsed["products_selected_ids"] = parsed.get("products_selected_ids", [])
    parsed["kb_files_selected"] = parsed.get("kb_files_selected", [])
    parsed["reasoning_short"] = parsed.get("reasoning_short", "")

    # enforce KB limit <= 5, just in case
    parsed["kb_files_selected"] = parsed["kb_files_selected"][:5]

    return parsed