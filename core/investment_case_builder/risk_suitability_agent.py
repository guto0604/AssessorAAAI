from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent

PROFILE_LEVELS = {"conservador": 1, "moderado": 2, "arrojado": 3, "agressivo": 3}
SCENARIO_LEVELS = {"conservador": 1, "moderado": 2, "objetivo": 3}


class RiskSuitabilityAgent(BaseInvestmentCaseAgent):
    agent_name = "risk_suitability"
    instruction = (
        "Revisar diagnóstico e cenários, classificando alertas de suitability, limitações e pontos que exigem validação humana."
    )

    def run(self, *, case_state: dict) -> AgentResult:
        suitability = (case_state.get("selected_client_context", {}).get("client_summary", {}).get("perfil_suitability") or "Não informado").lower()
        scenario_list = case_state.get("scenarios", [])
        profile_level = PROFILE_LEVELS.get(suitability, 2)
        alerts = []
        human_review = []

        for scenario in scenario_list:
            scenario_level = SCENARIO_LEVELS.get(scenario.get("profile_anchor"), 2)
            if scenario_level > profile_level:
                alerts.append(
                    {
                        "scenario": scenario.get("name"),
                        "severity": "high",
                        "message": "Cenário pode exceder o apetite de risco cadastrado e deve ser revalidado antes da recomendação formal.",
                    }
                )
                human_review.append(f"Validar suitability do cenário '{scenario.get('name')}' em relação ao cadastro atual ({suitability}).")
            else:
                alerts.append(
                    {
                        "scenario": scenario.get("name"),
                        "severity": "medium",
                        "message": "Cenário parece compatível no nível macro, mas ainda requer confirmação de liquidez, tributação e horizonte.",
                    }
                )

        conflicts = case_state.get("data_relevance_decisions", {}).get("conflicts_detected", [])
        if conflicts:
            human_review.append("Existem conflitos entre prompt do assessor e dados cadastrais que precisam de validação explícita.")

        payload = {
            "overall_status": "review_required" if human_review else "within_expected_limits",
            "alerts": alerts,
            "inconsistencies": conflicts,
            "limitations": [
                "A análise usa apenas dados consolidados disponíveis localmente no app.",
                "Tributação específica e custos operacionais não foram detalhados por produto nesta etapa.",
            ],
            "human_review_required": bool(human_review),
            "human_review_items": human_review,
            "suggested_disclaimers": [
                "Material para uso consultivo interno, sujeito à validação de suitability, liquidez e condições de mercado.",
                "Qualquer implementação final depende de confirmação do cliente e checagens operacionais/compliance.",
            ],
        }
        return AgentResult(payload=payload, summary="Riscos e suitability revisados com classificação de severidade e pontos de validação humana.")
