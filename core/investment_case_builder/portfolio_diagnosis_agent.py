from __future__ import annotations

from collections import Counter

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent

PROFILE_TARGET_RISK = {"conservador": 1, "moderado": 2, "arrojado": 3, "agressivo": 3}


class PortfolioDiagnosisAgent(BaseInvestmentCaseAgent):
    agent_name = "portfolio_diagnosis"
    instruction = (
        "Analisar a situação atual da carteira do cliente para destacar concentração, liquidez, aderência ao perfil, oportunidades e pontos de atenção."
    )

    def run(self, *, case_state: dict) -> AgentResult:
        selected_context = case_state.get("selected_client_context", {})
        client_summary = selected_context.get("client_summary", {})
        categories = selected_context.get("relevant_financial_data", [])
        holdings = selected_context.get("relevant_holdings", [])
        profile = (client_summary.get("perfil_suitability") or "Não informado").lower()

        top_category = categories[0] if categories else None
        concentration_pct = top_category.get("allocation_pct", 0.0) if top_category else 0.0
        concentration_flag = concentration_pct >= 40
        liquidity_counts = Counter(item.get("liquidity_hint", "não informada") for item in holdings)
        liquidity_profile = liquidity_counts.most_common(1)[0][0] if liquidity_counts else "não informada"

        opportunities = []
        alerts = []
        strengths = []
        if concentration_flag:
            alerts.append(f"A maior categoria responde por {concentration_pct:.1f}% do patrimônio analisado, sugerindo concentração elevada.")
            opportunities.append("Avaliar rebalanceamento gradual para reduzir dependência da principal alocação.")
        else:
            strengths.append("A composição principal não sinaliza concentração crítica no recorte selecionado.")

        if client_summary.get("dinheiro_disponivel"):
            opportunities.append("Existe caixa disponível para implementar ajustes sem necessidade imediata de resgates.")
        else:
            alerts.append("Não há caixa claramente disponível; mudanças podem exigir realocação com gestão de liquidez.")

        rent_12m = client_summary.get("rentabilidade_12_meses")
        cdi_12m = client_summary.get("cdi_12_meses")
        if isinstance(rent_12m, (int, float)) and isinstance(cdi_12m, (int, float)):
            if rent_12m < cdi_12m:
                opportunities.append("O histórico de 12 meses abaixo do CDI sugere espaço para revisão de eficiência da carteira.")
            else:
                strengths.append("A carteira superou ou acompanhou o CDI no período informado, criando base positiva para ajustes táticos.")

        diagnosis = {
            "current_state": {
                "top_category": top_category,
                "liquidity_profile": liquidity_profile,
                "profile": client_summary.get("perfil_suitability"),
                "cash_available": client_summary.get("dinheiro_disponivel"),
                "top_holdings": holdings,
            },
            "key_findings": alerts + strengths,
            "opportunities": opportunities,
            "attention_points": alerts,
            "strengths": strengths,
            "executive_summary": (
                "Carteira com leitura inicial centrada em alocação atual, liquidez e aderência ao objetivo do assessor, "
                "priorizando oportunidades de ajuste com impacto consultivo."
            ),
        }
        return AgentResult(payload=diagnosis, summary=diagnosis["executive_summary"])
