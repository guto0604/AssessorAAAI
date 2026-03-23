from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent


class ScenarioBuilderAgent(BaseInvestmentCaseAgent):
    agent_name = "scenario_builder"
    instruction = (
        "Construir cenários comparáveis e rastreáveis, com racional, vantagens, riscos, trade-offs, liquidez aproximada e aderência ao objetivo."
    )

    def run(self, *, case_state: dict) -> AgentResult:
        diagnosis = case_state.get("portfolio_diagnosis", {})
        client_summary = case_state.get("selected_client_context", {}).get("client_summary", {})
        prompt = case_state.get("advisor_prompt", "")
        tone_focus = case_state.get("tone_focus", "")
        opportunities = diagnosis.get("opportunities", [])
        liquidity_profile = diagnosis.get("current_state", {}).get("liquidity_profile", "média")
        suitability = client_summary.get("perfil_suitability", "Não informado")

        scenarios = [
            {
                "name": "Cenário Conservador de Rebalanceamento",
                "profile_anchor": "conservador",
                "rationale": "Reduzir volatilidade e reforçar previsibilidade de caixa, preservando margem para ajustes graduais.",
                "advantages": [
                    "Maior previsibilidade de retorno esperado.",
                    "Melhor alinhamento com necessidades de liquidez de curto prazo.",
                ],
                "risks": ["Potencial menor de captura de upside em ciclos de risco."],
                "trade_offs": ["Menor agressividade em troca de mais estabilidade."],
                "goal_fit": f"Adequado quando o objetivo é proteger patrimônio e responder ao prompt: {prompt[:80]}",
                "profile_fit": f"Alta aderência a perfis conservadores ou moderados. Suitability cadastrado: {suitability}.",
                "approx_liquidity": "alta",
                "best_use_case": "Quando o cliente prioriza preservação de capital, caixa e disciplina de alocação.",
                "allocation_outline": [
                    {"bucket": "Renda fixa e caixa", "pct": 60},
                    {"bucket": "Multimercados/estratégias balanceadas", "pct": 25},
                    {"bucket": "Temas de crescimento", "pct": 15},
                ],
            },
            {
                "name": "Cenário Moderado de Eficiência",
                "profile_anchor": "moderado",
                "rationale": "Buscar melhor equilíbrio entre proteção, diversificação e captura de retorno, usando o diagnóstico atual como ponto de partida.",
                "advantages": [
                    "Combina defesa e opcionalidade para crescimento.",
                    "Permite endereçar concentração com transição mais suave.",
                ],
                "risks": ["Pode exigir maior tolerância a marcação a mercado."],
                "trade_offs": ["Aceita volatilidade moderada para elevar eficiência da carteira."],
                "goal_fit": "Bom para clientes que precisam combinar disciplina tática com retorno acima do benchmark.",
                "profile_fit": f"Aderência natural a perfis moderados; útil também como ponte para o suitability atual ({suitability}).",
                "approx_liquidity": liquidity_profile,
                "best_use_case": "Quando há espaço para diversificação e revisão de concentração sem ruptura brusca.",
                "allocation_outline": [
                    {"bucket": "Renda fixa e caixa", "pct": 40},
                    {"bucket": "Multimercados e crédito", "pct": 35},
                    {"bucket": "Renda variável/temas", "pct": 25},
                ],
            },
            {
                "name": "Cenário Tático Focado no Objetivo",
                "profile_anchor": "objetivo",
                "rationale": f"Estruturar a carteira com foco explícito no objetivo do assessor ({prompt[:120]}) e no tom/foco desejado ({tone_focus or 'consultivo'}).",
                "advantages": [
                    "Conecta diretamente a tese consultiva ao contexto da reunião.",
                    "Facilita storytelling comercial e priorização de próximos passos.",
                ],
                "risks": ["Dependência maior da hipótese central do case e da validação humana de implementação."],
                "trade_offs": ["Maior personalização pode exigir mais ajustes posteriores."],
                "goal_fit": "Máxima aderência ao objetivo declarado pelo assessor.",
                "profile_fit": f"Deve ser calibrado contra o suitability cadastrado ({suitability}) antes da execução.",
                "approx_liquidity": "média",
                "best_use_case": "Quando a reunião exige uma tese clara, diferenciada e alinhada ao motivo da interação.",
                "allocation_outline": [
                    {"bucket": "Reserva tática", "pct": 30},
                    {"bucket": "Estratégias núcleo", "pct": 45},
                    {"bucket": "Estratégias satélite ligadas ao objetivo", "pct": 25},
                ],
                "opportunity_links": opportunities[:3],
            },
        ]

        payload = {
            "summary": "Foram estruturados três cenários complementares para comparação em reunião consultiva.",
            "scenarios": scenarios,
        }
        return AgentResult(payload=payload, summary=payload["summary"])
