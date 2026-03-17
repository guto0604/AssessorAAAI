import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

CLIENTES_PATH = DATA_DIR / "informacoes_cliente.parquet"
JORNADAS_PATH = DATA_DIR / "jornadas_comerciais_poc.xlsx"
INVESTIMENTOS_PATH = DATA_DIR / "investimentos_cliente.parquet"
PRODUTOS_PATH = DATA_DIR / "produtos.parquet"

CLIENTES_COLUMNS = [
    "Cliente_ID",
    "Nome",
    "Patrimonio_Investido_Conosco",
    "Patrimonio_Investido_Outros",
    "Dinheiro_Disponivel_Para_Investir",
    "Perfil_Suitability",
    "Rentabilidade_12_meses",
    "CDI_12_Meses",
]

INVESTIMENTOS_COLUMNS = [
    "Cliente_ID",
    "Produto",
    "Categoria",
    "Valor_Investido",
]

PRODUTOS_COLUMNS = [
    "Produto_ID",
    "Nome_Produto",
    "Categoria",
    "Subcategoria",
    "Risco_Nivel",
    "Suitability_Ideal",
]

def load_clientes():
    """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    return pd.read_parquet(CLIENTES_PATH, columns=CLIENTES_COLUMNS)


def load_clientes_full():
    """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    return pd.read_parquet(CLIENTES_PATH)

def load_jornadas():
    """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    return pd.read_excel(JORNADAS_PATH)

def load_investimentos():
    """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    return pd.read_parquet(INVESTIMENTOS_PATH, columns=INVESTIMENTOS_COLUMNS)


def load_investimentos_full():
    """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    return pd.read_parquet(INVESTIMENTOS_PATH)

def load_produtos():
    """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    return pd.read_parquet(PRODUTOS_PATH, columns=PRODUTOS_COLUMNS)


def load_produtos_full():
    """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    return pd.read_parquet(PRODUTOS_PATH)

def get_cliente_by_id(cliente_id):
    """Obtém a informação necessária para a etapa atual do processo.

    Args:
        cliente_id: Identificador único do cliente usado para filtrar dados e arquivos relacionados.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    df = load_clientes()
    return df[df["Cliente_ID"] == cliente_id].iloc[0].to_dict()

def get_investimentos_by_cliente(cliente_id):
    """Obtém a informação necessária para a etapa atual do processo.

    Args:
        cliente_id: Identificador único do cliente usado para filtrar dados e arquivos relacionados.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    df = load_investimentos()
    return df[df["Cliente_ID"] == cliente_id].copy()

def carteira_summary_for_llm(cliente_info, investimentos_df):
    """Responsável por processar summary for llm no contexto da aplicação de assessoria.

    Args:
        cliente_info: Dicionário com os dados consolidados do cliente para personalizar a resposta.
        investimentos_df: Valor de entrada necessário para processar 'investimentos_df'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    total = float(investimentos_df["Valor_Investido"].sum()) if not investimentos_df.empty else 0.0
    categorias = (
        investimentos_df.groupby("Categoria")["Valor_Investido"].sum().sort_values(ascending=False).to_dict()
        if not investimentos_df.empty else {}
    )

    # Se você já colocou esses campos no excel de clientes, ótimo.
    rent_12m = cliente_info.get("Rentabilidade_12_meses")
    cdi_12m = cliente_info.get("CDI_12_Meses")

    return {
        "cliente_id": cliente_info.get("Cliente_ID"),
        "perfil_suitability": cliente_info.get("Perfil_Suitability"),
        "patrimonio_conosco": float(cliente_info.get("Patrimonio_Investido_Conosco", 0)),
        "patrimonio_fora": float(cliente_info.get("Patrimonio_Investido_Fora", 0)),
        "dinheiro_para_investir": float(cliente_info.get("Dinheiro_Disponivel_Para_Investir", 0)),
        "carteira_total_conosco_calculado": total,
        "carteira_por_categoria": categorias,
        "rentabilidade_12_meses": rent_12m,
        "cdi_12_meses": cdi_12m,
        "spread_vs_cdi_12m": (rent_12m - cdi_12m) if (isinstance(rent_12m, (int, float)) and isinstance(cdi_12m, (int, float))) else None,
    }
