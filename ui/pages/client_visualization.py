from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st

from core.data_loader import load_clientes_full, load_investimentos_full, load_produtos_full


def _is_missing(value: Any) -> bool:
    """Responsável por processar missing no contexto da aplicação de assessoria.

    Args:
        value: Valor de entrada necessário para processar 'value'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    return value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    """Responsável por processar float no contexto da aplicação de assessoria.

    Args:
        value: Valor de entrada necessário para processar 'value'.
        default: Valor de entrada necessário para processar 'default'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_datetime(value: Any) -> datetime | None:
    """Responsável por processar datetime no contexto da aplicação de assessoria.

    Args:
        value: Valor de entrada necessário para processar 'value'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if _is_missing(value):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _format_currency(value: Any) -> str:
    """Responsável por formatar currency no contexto da aplicação de assessoria.

    Args:
        value: Valor de entrada necessário para processar 'value'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    number = _to_float(value, default=0.0)
    formatted = f"R$ {number:,.2f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _format_percent(value: Any, as_fraction: bool = True) -> str:
    """Responsável por formatar percent no contexto da aplicação de assessoria.

    Args:
        value: Valor de entrada necessário para processar 'value'.
        as_fraction: Valor de entrada necessário para processar 'as_fraction'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    number = _to_float(value, default=0.0)
    pct = number * 100 if as_fraction else number
    return f"{pct:.1f}%"


def _format_date(value: Any) -> str:
    """Responsável por formatar date no contexto da aplicação de assessoria.

    Args:
        value: Valor de entrada necessário para processar 'value'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    dt = _to_datetime(value)
    if not dt:
        return "-"
    return dt.strftime("%d/%m/%Y")


def _safe_text(value: Any, default: str = "-") -> str:
    """Responsável por processar text no contexto da aplicação de assessoria.

    Args:
        value: Valor de entrada necessário para processar 'value'.
        default: Valor de entrada necessário para processar 'default'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if _is_missing(value):
        return default
    text = str(value).strip()
    return text if text else default


def _priority_badge(label: str, value: str) -> str:
    """Responsável por processar badge no contexto da aplicação de assessoria.

    Args:
        label: Valor de entrada necessário para processar 'label'.
        value: Valor de entrada necessário para processar 'value'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    palette = {
        "Ativo": "#1B5E20",
        "Morno": "#9A3412",
        "Em Risco": "#991B1B",
        "Alta": "#4A148C",
        "Média": "#1E3A8A",
        "Baixa": "#374151",
        "Alto": "#166534",
        "Médio": "#92400E",
        "Baixo": "#4B5563",
    }
    color = palette.get(value, "#1F2937")
    return (
        f"<span style='background:{color}; color:#FFFFFF; border:1px solid #111827; padding:4px 10px; border-radius:14px; margin-right:6px; display:inline-block;'>"
        f"<b>{label}:</b> {value}</span>"
    )


def _section_title(title: str, description: str) -> None:
    """Responsável por processar title no contexto da aplicação de assessoria.

    Args:
        title: Valor de entrada necessário para processar 'title'.
        description: Valor de entrada necessário para processar 'description'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    st.subheader(title, help=description)


def _calculate_kpis(cliente: dict[str, Any]) -> dict[str, float]:
    """Responsável por calcular kpis no contexto da aplicação de assessoria.

    Args:
        cliente: Valor de entrada necessário para processar 'cliente'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    patrimonio_conosco = _to_float(cliente.get("Patrimonio_Investido_Conosco"))
    patrimonio_fora = _to_float(cliente.get("Patrimonio_Investido_Outros"))
    total = patrimonio_conosco + patrimonio_fora
    pct_conosco = (patrimonio_conosco / total) if total > 0 else 0.0

    return {
        "patrimonio_conosco": patrimonio_conosco,
        "patrimonio_fora": patrimonio_fora,
        "pct_conosco": _to_float(cliente.get("Percentual_Carteira_Conosco"), pct_conosco),
        "dinheiro_disponivel": _to_float(cliente.get("Dinheiro_Disponivel_Para_Investir")),
        "receita_12m": _to_float(cliente.get("Receita_Gerada_12M")),
        "capacidade_aporte": _to_float(cliente.get("Capacidade_Mensal_Aporte")),
        "aportes_12m": _to_float(cliente.get("Aportes_12M")),
        "resgates_12m": _to_float(cliente.get("Resgates_12M")),
    }


def _aggregate_carteira(investimentos_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Responsável por processar carteira no contexto da aplicação de assessoria.

    Args:
        investimentos_df: Valor de entrada necessário para processar 'investimentos_df'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if investimentos_df.empty:
        empty = pd.DataFrame(columns=["Grupo", "Valor"])
        return {
            "categoria": empty,
            "subcategoria": empty,
            "indexador": empty,
            "liquidez": empty,
            "risco": empty,
            "top_produtos": empty,
            "exposicao": pd.DataFrame(columns=["Exposicao_Internacional", "Valor_Atual"]),
        }

    base = investimentos_df.copy()
    if "Valor_Atual" not in base.columns:
        base["Valor_Atual"] = base.get("Valor_Investido", 0)

    def _group(col: str, out_col: str = "Grupo") -> pd.DataFrame:
        """Responsável por executar uma etapa do fluxo da aplicação de assessoria.

        Args:
            col: Valor de entrada necessário para processar 'col'.
            out_col: Valor de entrada necessário para processar 'out_col'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        if col not in base.columns:
            return pd.DataFrame(columns=[out_col, "Valor"])
        g = (
            base.groupby(col, dropna=False)["Valor_Atual"]
            .sum()
            .reset_index()
            .rename(columns={col: out_col, "Valor_Atual": "Valor"})
            .sort_values("Valor", ascending=False)
        )
        total = g["Valor"].sum()
        g["Percentual"] = g["Valor"] / total if total > 0 else 0.0
        g[out_col] = g[out_col].fillna("Não informado")
        return g

    return {
        "categoria": _group("Categoria"),
        "subcategoria": _group("Subcategoria"),
        "indexador": _group("Indexador"),
        "liquidez": _group("Liquidez"),
        "risco": _group("Risco_Nivel", out_col="Risco_Nivel"),
        "top_produtos": _group("Produto", out_col="Produto").head(5),
        "exposicao": _group("Exposicao_Internacional", out_col="Exposicao_Internacional"),
    }


def _calculate_aderencia(cliente: dict[str, Any], investimentos_df: pd.DataFrame) -> dict[str, float]:
    """Responsável por calcular aderencia no contexto da aplicação de assessoria.

    Args:
        cliente: Valor de entrada necessário para processar 'cliente'.
        investimentos_df: Valor de entrada necessário para processar 'investimentos_df'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if investimentos_df.empty:
        return {
            "pct_adequado": 0.0,
            "pct_inadequado": 0.0,
            "pct_liquidez_incompativel": 0.0,
            "concentracao_top3": 0.0,
        }

    base = investimentos_df.copy()
    if "Valor_Atual" not in base.columns:
        base["Valor_Atual"] = base.get("Valor_Investido", 0)

    total = base["Valor_Atual"].sum()
    if total <= 0:
        total = 1

    adequado_mask = base.get("Adequado_ao_Perfil", "").astype(str).str.lower().eq("sim")
    pct_adequado = base.loc[adequado_mask, "Valor_Atual"].sum() / total
    pct_inadequado = 1 - pct_adequado

    necessidade_liquidez = _safe_text(cliente.get("Necessidade_Liquidez_Curto_Prazo"), "Baixa")
    liquidez_incompativel = pd.Series(False, index=base.index)
    if "Liquidez" in base.columns and necessidade_liquidez in {"Alta", "Média"}:
        if necessidade_liquidez == "Alta":
            permitidos = {"D+0", "D+1", "D+2"}
        else:
            permitidos = {"D+0", "D+1", "D+2", "D+15", "D+30"}
        liquidez_incompativel = ~base["Liquidez"].fillna("Não informado").isin(permitidos)

    pct_liquidez_incompativel = base.loc[liquidez_incompativel, "Valor_Atual"].sum() / total

    top3 = base.nlargest(3, "Valor_Atual")
    concentracao_top3 = top3["Valor_Atual"].sum() / total

    return {
        "pct_adequado": max(min(pct_adequado, 1.0), 0.0),
        "pct_inadequado": max(min(pct_inadequado, 1.0), 0.0),
        "pct_liquidez_incompativel": max(min(pct_liquidez_incompativel, 1.0), 0.0),
        "concentracao_top3": max(min(concentracao_top3, 1.0), 0.0),
    }


def _generate_alertas(cliente: dict[str, Any], kpis: dict[str, float], aderencia: dict[str, float]) -> list[dict[str, str]]:
    """Responsável por gerar alertas no contexto da aplicação de assessoria.

    Args:
        cliente: Valor de entrada necessário para processar 'cliente'.
        kpis: Valor de entrada necessário para processar 'kpis'.
        aderencia: Valor de entrada necessário para processar 'aderencia'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    alertas: list[dict[str, str]] = []

    score_churn = _to_float(cliente.get("Score_Risco_Churn"))
    if score_churn >= 75:
        alertas.append({"prioridade": "critico", "titulo": "Risco de churn elevado", "descricao": f"Score de churn em {score_churn:.0f}."})

    engajamento = _to_float(cliente.get("Score_Engajamento"))
    if 0 < engajamento < 40:
        alertas.append({"prioridade": "atencao", "titulo": "Engajamento baixo", "descricao": f"Score de engajamento em {engajamento:.0f}."})

    dias_contato = _to_float(cliente.get("Dias_Desde_Ultimo_Contato"))
    if dias_contato > 45:
        alertas.append({"prioridade": "atencao", "titulo": "Contato desatualizado", "descricao": f"Sem interação há {dias_contato:.0f} dias."})

    data_suitability = _to_datetime(cliente.get("Data_Ultima_Atualizacao_Suitability"))
    if data_suitability and (datetime.now() - data_suitability).days > 365:
        alertas.append({"prioridade": "atencao", "titulo": "Suitability desatualizado", "descricao": "Atualização há mais de 12 meses."})

    if aderencia["pct_inadequado"] > 0.25:
        alertas.append({"prioridade": "critico", "titulo": "Carteira desalinhada ao perfil", "descricao": f"{_format_percent(aderencia['pct_inadequado'])} da carteira inadequada."})

    if aderencia["pct_liquidez_incompativel"] > 0.2:
        alertas.append({"prioridade": "atencao", "titulo": "Liquidez incompatível", "descricao": "Parte relevante da carteira não atende necessidade de curto prazo."})

    if kpis["pct_conosco"] < 0.35 and kpis["patrimonio_fora"] > kpis["patrimonio_conosco"]:
        alertas.append({"prioridade": "info", "titulo": "Baixo share of wallet", "descricao": "Patrimônio fora da casa acima do valor investido conosco."})

    if kpis["dinheiro_disponivel"] > 100000 and _safe_text(cliente.get("Potencial_Captacao")) == "Alto":
        alertas.append({"prioridade": "info", "titulo": "Caixa disponível com alto potencial", "descricao": "Cliente com recursos imediatos para alocação."})

    return alertas


def _insights_automaticos(cliente: dict[str, Any], kpis: dict[str, float], aderencia: dict[str, float]) -> list[str]:
    """Responsável por processar automaticos no contexto da aplicação de assessoria.

    Args:
        cliente: Valor de entrada necessário para processar 'cliente'.
        kpis: Valor de entrada necessário para processar 'kpis'.
        aderencia: Valor de entrada necessário para processar 'aderencia'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    insights: list[str] = []
    if _safe_text(cliente.get("Potencial_Captacao")) == "Alto" and kpis["pct_conosco"] < 0.4:
        insights.append("Cliente com alto potencial de captação e baixo share of wallet.")
    if aderencia["pct_inadequado"] > 0.2:
        insights.append("Cliente com carteira parcialmente desalinhada ao perfil.")
    if _to_float(cliente.get("Dias_Desde_Ultimo_Contato")) > 45 and _to_float(cliente.get("Score_Risco_Churn")) > 70:
        insights.append("Cliente sem contato recente e com churn elevado.")
    if _to_float(cliente.get("Dinheiro_Disponivel_Para_Investir")) > 0:
        insights.append("Há caixa disponível para novas alocações no curto prazo.")
    return insights


def _horizonte_compatibilidade(horizonte: str, prazo_produto: str) -> bool:
    """Responsável por processar compatibilidade no contexto da aplicação de assessoria.

    Args:
        horizonte: Valor de entrada necessário para processar 'horizonte'.
        prazo_produto: Valor de entrada necessário para processar 'prazo_produto'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if not horizonte or not prazo_produto:
        return True
    mapa = {
        "Curto Prazo": {"Liquidez imediata", "Curto/Médio Prazo", "Médio Prazo"},
        "Médio Prazo": {"Curto/Médio Prazo", "Médio Prazo", "Médio/Longo Prazo"},
        "Longo Prazo": {"Médio/Longo Prazo", "Longo Prazo", "Sem vencimento"},
    }
    return prazo_produto in mapa.get(horizonte, set())


def _suitability_compatibilidade(perfil: str, suitability_ideal: str) -> bool:
    """Responsável por processar compatibilidade no contexto da aplicação de assessoria.

    Args:
        perfil: Valor de entrada necessário para processar 'perfil'.
        suitability_ideal: Identificador usado para referenciar 'suitability_ideal'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if not perfil or not suitability_ideal:
        return True
    perfis_aceitos = {s.strip() for s in suitability_ideal.split("/")}
    return perfil in perfis_aceitos


def _liquidez_compatibilidade(necessidade: str, liquidez_produto: str) -> bool:
    """Responsável por processar compatibilidade no contexto da aplicação de assessoria.

    Args:
        necessidade: Identificador usado para referenciar 'necessidade'.
        liquidez_produto: Identificador usado para referenciar 'liquidez_produto'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if not necessidade or not liquidez_produto:
        return True
    regras = {
        "Alta": {"D+0", "D+1", "D+2", "Liquidez imediata"},
        "Média": {"D+0", "D+1", "D+2", "D+15", "D+30", "Liquidez imediata"},
        "Baixa": {"D+0", "D+1", "D+2", "D+15", "D+30", "No vencimento", "Mercado secundário", "Liquidez imediata"},
    }
    return liquidez_produto in regras.get(necessidade, set())


def _montar_oportunidades(cliente: dict[str, Any], produtos_df: pd.DataFrame, kpis: dict[str, float]) -> pd.DataFrame:
    """Responsável por processar oportunidades no contexto da aplicação de assessoria.

    Args:
        cliente: Valor de entrada necessário para processar 'cliente'.
        produtos_df: Valor de entrada necessário para processar 'produtos_df'.
        kpis: Valor de entrada necessário para processar 'kpis'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if produtos_df.empty:
        return pd.DataFrame()

    perfil = _safe_text(cliente.get("Perfil_Suitability"), "")
    horizonte = _safe_text(cliente.get("Horizonte_Investimento"), "")
    necessidade_liquidez = _safe_text(cliente.get("Necessidade_Liquidez_Curto_Prazo"), "")
    capacidade = kpis["dinheiro_disponivel"] + kpis["capacidade_aporte"]

    df = produtos_df.copy()
    if "Aplicacao_Minima" not in df.columns:
        df["Aplicacao_Minima"] = 0

    registros = []
    for _, row in df.iterrows():
        suit_ok = _suitability_compatibilidade(perfil, _safe_text(row.get("Suitability_Ideal"), ""))
        liq_ok = _liquidez_compatibilidade(necessidade_liquidez, _safe_text(row.get("Liquidez"), ""))
        min_ok = capacidade >= _to_float(row.get("Aplicacao_Minima"))
        horizon_ok = _horizonte_compatibilidade(horizonte, _safe_text(row.get("Prazo"), ""))
        camp = _safe_text(row.get("Produto_em_Campanha"), "Não") == "Sim"

        score = 0
        score += 40 if suit_ok else 0
        score += 20 if liq_ok else 0
        score += 15 if min_ok else 0
        score += 15 if horizon_ok else 0
        score += 10 if camp else 0

        motivos = []
        if suit_ok:
            motivos.append("Aderente ao suitability")
        if liq_ok:
            motivos.append("Compatível com liquidez")
        if min_ok:
            motivos.append("Aplicação mínima viável")
        if horizon_ok:
            motivos.append("Alinhado ao horizonte")
        if camp:
            motivos.append("Produto em campanha")

        if motivos:
            registros.append(
                {
                    "Produto": _safe_text(row.get("Nome_Produto")),
                    "Categoria": _safe_text(row.get("Categoria")),
                    "Suitability ideal": _safe_text(row.get("Suitability_Ideal")),
                    "Liquidez": _safe_text(row.get("Liquidez")),
                    "Aplicação mínima": _format_currency(row.get("Aplicacao_Minima")),
                    "Em campanha": "Sim" if camp else "Não",
                    "Motivo da recomendação": " | ".join(motivos),
                    "_score": score,
                }
            )

    if not registros:
        return pd.DataFrame()

    oportunidades = pd.DataFrame(registros).sort_values("_score", ascending=False).head(10)
    return oportunidades.drop(columns=["_score"])


def _render_alertas(alertas: list[dict[str, str]]) -> None:
    """Responsável por renderizar alertas no contexto da aplicação de assessoria.

    Args:
        alertas: Valor de entrada necessário para processar 'alertas'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if not alertas:
        st.info("Sem alertas prioritários para este cliente no momento.")
        return

    color_map = {
        "critico": {"bg": "#7F1D1D", "border": "#FCA5A5"},
        "atencao": {"bg": "#78350F", "border": "#FCD34D"},
        "info": {"bg": "#1E3A8A", "border": "#93C5FD"},
    }
    for alerta in alertas:
        style = color_map.get(alerta["prioridade"], {"bg": "#374151", "border": "#D1D5DB"})
        st.markdown(
            f"""
            <div style='background:{style['bg']}; color:#FFFFFF; border:1px solid {style['border']}; padding:12px; border-radius:10px; margin-bottom:8px;'>
                <strong>{alerta['titulo']}</strong><br/>
                <span>{alerta['descricao']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _filter_selected_client(clientes_df: pd.DataFrame, selected_cliente_id: Any) -> pd.DataFrame:
    """Responsável por filtrar selected client no contexto da aplicação de assessoria.

    Args:
        clientes_df: Valor de entrada necessário para processar 'clientes_df'.
        selected_cliente_id: Identificador usado para referenciar 'selected_cliente_id'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if "Cliente_ID" not in clientes_df.columns:
        return pd.DataFrame()
    return clientes_df[clientes_df["Cliente_ID"] == selected_cliente_id]


def _build_objetivo_financeiro(cliente: dict[str, Any], patrimonio_conosco: float) -> dict[str, Any]:
    """Responsável por montar objetivo financeiro no contexto da aplicação de assessoria.

    Args:
        cliente: Valor de entrada necessário para processar 'cliente'.
        patrimonio_conosco: Valor de entrada necessário para processar 'patrimonio_conosco'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    valor_alvo = _to_float(cliente.get("Valor_Objetivo_Financeiro"))
    progresso = (patrimonio_conosco / valor_alvo) if valor_alvo > 0 else 0.0
    progresso = max(progresso, 0.0)
    faltante = max(valor_alvo - patrimonio_conosco, 0.0) if valor_alvo > 0 else 0.0
    return {
        "objetivo": _safe_text(cliente.get("Objetivo_Principal"), "Objetivo ainda não definido"),
        "horizonte": _safe_text(cliente.get("Horizonte_Investimento"), "Não informado"),
        "valor_alvo": valor_alvo,
        "progresso": progresso,
        "faltante": faltante,
    }


def _format_positions_table(inv_cliente: pd.DataFrame) -> pd.DataFrame:
    """Responsável por formatar positions table no contexto da aplicação de assessoria.

    Args:
        inv_cliente: Valor de entrada necessário para processar 'inv_cliente'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if inv_cliente.empty:
        return pd.DataFrame(columns=["Produto", "Categoria", "Valor atual", "Peso na carteira", "Liquidez"])

    base = inv_cliente.copy()
    base["Valor_Atual"] = pd.to_numeric(base.get("Valor_Atual", base.get("Valor_Investido", 0)), errors="coerce").fillna(0)
    total = base["Valor_Atual"].sum()
    base["Peso"] = base["Valor_Atual"] / total if total > 0 else 0.0

    tabela = pd.DataFrame(
        {
            "Produto": base.get("Produto", "Não informado").fillna("Não informado"),
            "Categoria": base.get("Categoria", "Não informado").fillna("Não informado"),
            "Valor atual": base["Valor_Atual"],
            "Peso na carteira": base["Peso"],
            "Liquidez": base.get("Liquidez", "Não informado").fillna("Não informado"),
        }
    )
    return tabela.sort_values("Valor atual", ascending=False).head(15)


def _render_cliente_view(cliente: dict[str, Any], inv_cliente: pd.DataFrame, kpis: dict[str, float], carteira: dict[str, pd.DataFrame]) -> None:
    """Responsável por renderizar cliente view no contexto da aplicação de assessoria.

    Args:
        cliente: Valor de entrada necessário para processar 'cliente'.
        inv_cliente: Valor de entrada necessário para processar 'inv_cliente'.
        kpis: Valor de entrada necessário para processar 'kpis'.
        carteira: Valor de entrada necessário para processar 'carteira'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    st.header("Acompanhamento do seu patrimônio")
    st.caption("Aqui você acompanha sua carteira, seus objetivos e a composição dos seus investimentos.")

    st.subheader("Resumo patrimonial")
    c1, c2, c3 = st.columns(3)
    c1.metric("Seu patrimônio investido", _format_currency(kpis["patrimonio_conosco"]))

    rent = _to_float(cliente.get("Rentabilidade_12_meses"))
    cdi = _to_float(cliente.get("CDI_12_Meses"))
    diff = rent - cdi
    c2.metric("Rentabilidade em 12 meses", _format_percent(rent), delta=f"vs CDI: {_format_percent(diff)}")
    c3.metric("Saldo disponível", _format_currency(kpis["dinheiro_disponivel"]))

    st.subheader("Objetivo financeiro")
    objetivo = _build_objetivo_financeiro(cliente, kpis["patrimonio_conosco"])
    st.markdown(f"**{objetivo['objetivo']}**")
    st.caption(f"Horizonte de investimento: {objetivo['horizonte']}")

    if objetivo["valor_alvo"] <= 0:
        st.info("Ainda não há um valor alvo definido para acompanhar este objetivo.")
    else:
        progresso_texto = min(objetivo["progresso"], 1.0)
        if progresso_texto >= 1:
            st.success("Parabéns! Você já atingiu seu objetivo financeiro.")
        elif progresso_texto >= 0.75:
            st.info(f"Você já percorreu {_format_percent(progresso_texto)} do caminho para seu objetivo.")
        else:
            st.info(f"Você já percorreu {_format_percent(progresso_texto)} do caminho para seu objetivo.")

        st.progress(progresso_texto, text=f"Progresso estimado: {_format_percent(progresso_texto)}")
        o1, o2, o3 = st.columns(3)
        o1.metric("Valor acumulado", _format_currency(kpis["patrimonio_conosco"]))
        o2.metric("Valor alvo", _format_currency(objetivo["valor_alvo"]))
        o3.metric("Valor faltante", _format_currency(objetivo["faltante"]))

    st.subheader("Alocação da carteira")
    if inv_cliente.empty:
        st.info("Ainda não encontramos posições para mostrar a composição da carteira.")
    else:
        categorias_df = carteira["categoria"][["Grupo", "Valor", "Percentual"]].rename(columns={"Grupo": "Categoria"})
        if categorias_df.empty:
            st.info("Não foi possível consolidar a alocação por categoria com os dados disponíveis.")
        else:
            donut_df = categorias_df.rename(columns={"Categoria": "category", "Valor": "value"})
            col_donut, col_leg = st.columns([1, 1])
            col_donut.vega_lite_chart(
                donut_df,
                {
                    "mark": {"type": "arc", "innerRadius": 55},
                    "encoding": {
                        "theta": {"field": "value", "type": "quantitative"},
                        "color": {"field": "category", "type": "nominal", "legend": None},
                        "tooltip": [
                            {"field": "category", "type": "nominal", "title": "Categoria"},
                            {"field": "value", "type": "quantitative", "title": "Valor"},
                        ],
                    },
                },
                width="stretch",
            )
            leg = categorias_df.copy()
            leg["Percentual"] = leg["Percentual"].apply(_format_percent)
            leg["Valor"] = leg["Valor"].apply(_format_currency)
            col_leg.dataframe(leg, width="stretch", hide_index=True)

    st.subheader("Liquidez e prazo")
    liquidez_df = carteira["liquidez"]
    if liquidez_df.empty:
        st.info("Não há dados suficientes para classificar a carteira por liquidez.")
    else:
        st.caption("Parte da sua carteira está disponível para resgate rápido, enquanto outra parte está alocada em investimentos com prazo maior.")
        st.bar_chart(liquidez_df.set_index("Grupo")["Valor"], horizontal=True)

    st.subheader("Diversificação e exposição")
    expo = carteira["exposicao"]
    total_carteira = _to_float(inv_cliente.get("Valor_Atual", inv_cliente.get("Valor_Investido", 0)).sum() if not inv_cliente.empty else 0)
    internacional = 0.0
    if not expo.empty and "Exposicao_Internacional" in expo.columns:
        mask = expo["Exposicao_Internacional"].astype(str).str.lower().isin({"sim", "internacional", "true"})
        internacional = expo.loc[mask, "Valor"].sum()
    percentual_internacional = (internacional / total_carteira) if total_carteira > 0 else 0.0

    d1, d2 = st.columns(2)
    d1.metric("Exposição internacional", _format_percent(percentual_internacional))

    top_posicoes = _format_positions_table(inv_cliente).head(3)
    if top_posicoes.empty:
        d2.metric("Concentração nas maiores posições", "-")
    else:
        concentracao_top3 = top_posicoes["Peso na carteira"].sum()
        d2.metric("Concentração nas 3 maiores posições", _format_percent(concentracao_top3))
        st.caption("Principais posições da carteira:")
        resumo_top = top_posicoes[["Produto", "Peso na carteira"]].copy()
        resumo_top["Peso na carteira"] = resumo_top["Peso na carteira"].apply(_format_percent)
        st.dataframe(resumo_top, width="stretch", hide_index=True)

    st.subheader("Detalhe dos investimentos")
    tabela = _format_positions_table(inv_cliente)
    if tabela.empty:
        st.info("Não há investimentos para detalhar no momento.")
    else:
        tabela_exibicao = tabela.copy()
        tabela_exibicao["Valor atual"] = tabela_exibicao["Valor atual"].apply(_format_currency)
        tabela_exibicao["Peso na carteira"] = tabela_exibicao["Peso na carteira"].apply(_format_percent)
        st.dataframe(tabela_exibicao, width="stretch", hide_index=True)

    st.subheader("Perfil do investidor")
    p1, p2 = st.columns(2)
    p1.markdown(f"**Perfil suitability:** {_safe_text(cliente.get('Perfil_Suitability'), 'Não informado')}")
    p1.markdown(f"**Horizonte de investimento:** {_safe_text(cliente.get('Horizonte_Investimento'), 'Não informado')}")
    p2.markdown(
        f"**Necessidade de liquidez de curto prazo:** {_safe_text(cliente.get('Necessidade_Liquidez_Curto_Prazo'), 'Não informada')}"
    )
    p2.markdown(f"**Nível de conhecimento financeiro:** {_safe_text(cliente.get('Nivel_Conhecimento_Financeiro'), 'Não informado')}")
    st.caption("Seu perfil atual indica como equilibrar segurança, liquidez e potencial de retorno para a sua carteira.")


def render_visualizacao_clientes_tab(selected_cliente_id: Any) -> None:
    """Renderiza a seção da interface correspondente a este fluxo da aplicação.

    Args:
        selected_cliente_id: Identificador usado para referenciar 'selected_cliente_id'.

    Returns:
        Não retorna valor; atualiza diretamente os componentes da interface.
    """
    clientes_df = load_clientes_full()
    cliente_df = _filter_selected_client(clientes_df, selected_cliente_id)

    if cliente_df.empty:
        st.warning("Cliente não encontrado na base de informações.")
        return

    cliente = cliente_df.iloc[0].to_dict()
    investimentos = load_investimentos_full()
    inv_cliente = investimentos[investimentos["Cliente_ID"] == selected_cliente_id].copy() if "Cliente_ID" in investimentos.columns else pd.DataFrame()
    produtos = load_produtos_full()

    kpis = _calculate_kpis(cliente)
    carteira = _aggregate_carteira(inv_cliente)
    aderencia = _calculate_aderencia(cliente, inv_cliente)
    alertas = _generate_alertas(cliente, kpis, aderencia)
    insights = _insights_automaticos(cliente, kpis, aderencia)
    oportunidades = _montar_oportunidades(cliente, produtos, kpis)

    st.header("Visualização clientes", help="Painel consolidado da carteira do cliente com modo assessor e cliente.")
    modo_cliente = st.checkbox("Modo Cliente", value=False, help="Ative para visualizar a experiência simplificada do cliente final.")

    if modo_cliente:
        _render_cliente_view(cliente, inv_cliente, kpis, carteira)
        return

    _section_title("1) Cabeçalho executivo do cliente", "Resumo rápido com dados de relacionamento, perfil e contexto do cliente.")
    nome = _safe_text(cliente.get("Nome"), "Cliente sem nome")
    st.markdown(f"### {nome}")
    st.caption(
        f"Classe: {_safe_text(cliente.get('Classe_Cliente'))} | Assessor: {_safe_text(cliente.get('Assessor_Responsavel'))} | "
        f"Canal: {_safe_text(cliente.get('Canal_Preferencial'))}"
    )

    badges_html = "".join(
        [
            _priority_badge("Status", _safe_text(cliente.get("Status_Relacionamento"))),
            _priority_badge("Prioridade", _safe_text(cliente.get("Prioridade_Comercial"))),
            _priority_badge("Potencial", _safe_text(cliente.get("Potencial_Captacao"))),
        ]
    )
    st.markdown(badges_html, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dias desde último contato", f"{int(_to_float(cliente.get('Dias_Desde_Ultimo_Contato')))}")
    c2.metric("Último contato", _format_date(cliente.get("Data_Ultimo_Contato")))
    c3.metric("Tempo como cliente", f"{_to_float(cliente.get('Tempo_Como_Cliente')):.1f} anos")
    st.caption(
        f"Entrada: {_format_date(cliente.get('Data_Entrada_Base'))} • Profissão: {_safe_text(cliente.get('Profissão'))} • "
        f"Localidade: {_safe_text(cliente.get('Cidade'))}/{_safe_text(cliente.get('UF'))}"
    )

    _section_title("2) KPIs principais", "Indicadores de patrimônio, receita e potencial de alocação para priorização comercial.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Patrimônio conosco", _format_currency(kpis["patrimonio_conosco"]))
    k2.metric("Patrimônio fora", _format_currency(kpis["patrimonio_fora"]))
    k3.metric("% carteira conosco", _format_percent(kpis["pct_conosco"]))
    st.progress(max(min(kpis["pct_conosco"], 1.0), 0.0), text="Share of wallet conosco")

    k4, k5, k6 = st.columns(3)
    k4.metric("Dinheiro disponível", _format_currency(kpis["dinheiro_disponivel"]))
    k5.metric("Receita gerada (12m)", _format_currency(kpis["receita_12m"]))
    k6.metric("Capacidade mensal de aporte", _format_currency(kpis["capacidade_aporte"]))

    _section_title("3) Alertas e sinais prioritários", "Sinais de atenção e oportunidades imediatas com base em churn, engajamento e carteira.")
    _render_alertas(alertas)

    st.markdown("**Insights automáticos**")
    if insights:
        for insight in insights:
            st.write(f"- {insight}")
    else:
        st.write("- Sem insights adicionais gerados pelas regras atuais.")

    _section_title("4) Performance resumida", "Comparativo de retorno da carteira contra o CDI em 12 meses.")
    rent = _to_float(cliente.get("Rentabilidade_12_meses"))
    cdi = _to_float(cliente.get("CDI_12_Meses"))
    perf_df = pd.DataFrame(
        {
            "Indicador": ["Carteira", "CDI"],
            "Percentual": [rent, cdi],
        }
    )
    st.bar_chart(perf_df.set_index("Indicador"), horizontal=True)

    _section_title("5) Composição da carteira", "Distribuição dos investimentos por categoria, subcategoria, liquidez e exposição internacional.")
    if inv_cliente.empty:
        st.info("Sem dados de posições para este cliente.")
    else:
        left, right = st.columns(2)
        left.markdown("**Categorias**")
        categorias_df = carteira["categoria"][["Grupo", "Valor", "Percentual"]].rename(columns={"Grupo": "Categoria"})
        left.dataframe(categorias_df, width="stretch", hide_index=True)

        if not categorias_df.empty:
            pizza_df = categorias_df.rename(columns={"Categoria": "category", "Valor": "value"})
            left.markdown("**Categorias**")
            left.vega_lite_chart(
                pizza_df,
                {
                    "mark": {"type": "arc", "innerRadius": 35},
                    "encoding": {
                        "theta": {"field": "value", "type": "quantitative"},
                        "color": {"field": "category", "type": "nominal", "legend": {"title": "Categoria"}},
                        "tooltip": [
                            {"field": "category", "type": "nominal", "title": "Categoria"},
                            {"field": "value", "type": "quantitative", "title": "Valor"},
                        ],
                    },
                },
                width="stretch",
            )

        right.markdown("**Subcategorias**")
        right.bar_chart(carteira["subcategoria"].set_index("Grupo")["Valor"], horizontal=True)

        right.markdown("**Liquidez**")
        right.bar_chart(carteira["liquidez"].set_index("Grupo")["Valor"], horizontal=True)

        expo = carteira["exposicao"]
        if not expo.empty:
            right.markdown("**Exposição internacional**")
            right.dataframe(expo.rename(columns={"Exposicao_Internacional": "Exposição", "Valor": "Valor"}), width="stretch", hide_index=True)

    _section_title("6) Risco e aderência ao perfil", "Compatibilidade da carteira com suitability, liquidez desejada e concentração de risco.")
    st.write(
        f"Perfil suitability: **{_safe_text(cliente.get('Perfil_Suitability'))}** | "
        f"Horizonte: **{_safe_text(cliente.get('Horizonte_Investimento'))}** | "
        f"Necessidade de liquidez: **{_safe_text(cliente.get('Necessidade_Liquidez_Curto_Prazo'))}**"
    )

    a1, a2, a3 = st.columns(3)
    a1.metric("Carteira adequada ao perfil", _format_percent(aderencia["pct_adequado"]))
    a2.metric("Carteira com liquidez incompatível", _format_percent(aderencia["pct_liquidez_incompativel"]))
    a3.metric("Concentração top 3", _format_percent(aderencia["concentracao_top3"]))

    if not carteira["risco"].empty:
        st.markdown("**Distribuição por nível de risco**")
        st.bar_chart(carteira["risco"].set_index("Risco_Nivel")["Valor"], horizontal=True)

    if not carteira["top_produtos"].empty:
        st.markdown("**Top 5 maiores posições**")
        st.dataframe(carteira["top_produtos"].rename(columns={"Valor": "Valor atual"}), width="stretch", hide_index=True)

    _section_title("7) Objetivo financeiro", "Acompanhamento de progresso do objetivo principal, valor alvo e prazo para conclusão.")
    objetivo = _safe_text(cliente.get("Objetivo_Principal"), "Não informado")
    valor_obj = _to_float(cliente.get("Valor_Objetivo_Financeiro"))
    progresso_raw = (kpis["patrimonio_conosco"] / valor_obj) if valor_obj > 0 else 0.0
    progresso = max(min(progresso_raw, 1.0), 0.0)
    faltante = max(valor_obj - kpis["patrimonio_conosco"], 0)

    st.markdown(f"**Objetivo principal:** {objetivo}")
    st.progress(progresso, text=f"Progresso estimado: {_format_percent(progresso)}")

    o1, o2, o3 = st.columns(3)
    o1.metric("Valor atual considerado", _format_currency(kpis["patrimonio_conosco"]))
    o2.metric("Valor alvo", _format_currency(valor_obj))
    o3.metric("Quanto falta", _format_currency(faltante))

    _section_title("8) Movimentação e relacionamento", "Evolução recente de aportes e resgates junto ao histórico de relacionamento com o cliente.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Aportes (12m)", _format_currency(kpis["aportes_12m"]))
    m2.metric("Resgates (12m)", _format_currency(kpis["resgates_12m"]))
    m3.metric("Saldo líquido (12m)", _format_currency(kpis["aportes_12m"] - kpis["resgates_12m"]))

    mov_df = pd.DataFrame({"Tipo": ["Aportes", "Resgates"], "Valor": [kpis["aportes_12m"], kpis["resgates_12m"]]})
    st.bar_chart(mov_df.set_index("Tipo"))

    timeline = pd.DataFrame(
        [
            {"Evento": "Entrada como cliente", "Data": _format_date(cliente.get("Data_Entrada_Base"))},
            {"Evento": "Último contato", "Data": _format_date(cliente.get("Data_Ultimo_Contato"))},
            {"Evento": "Atualização suitability", "Data": _format_date(cliente.get("Data_Ultima_Atualizacao_Suitability"))},
            {"Evento": "Data objetivo financeiro", "Data": _format_date(cliente.get("Data_Objetivo_Financeiro"))},
        ]
    )
    st.dataframe(timeline, width="stretch", hide_index=True)
    st.caption(
        f"Canal preferencial: {_safe_text(cliente.get('Canal_Preferencial'))} | "
        f"Frequência de contato: {_safe_text(cliente.get('Frequencia_Contato'))} | "
        f"Score de engajamento: {_to_float(cliente.get('Score_Engajamento')):.0f}"
    )

    _section_title("9) Oportunidades comerciais", "Lista de produtos elegíveis com justificativas de recomendação.")
    if oportunidades.empty:
        st.info("Sem oportunidades elegíveis com as regras atuais.")
    else:
        st.dataframe(oportunidades, width="stretch", hide_index=True)
