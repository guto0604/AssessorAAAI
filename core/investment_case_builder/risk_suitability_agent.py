from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent
from core.investment_case_builder.llm_support import try_json_completion
from core.investment_case_builder.prompts import build_risk_prompts

PROFILE_LEVELS = {"conservador": 1, "moderado": 2, "arrojado": 3, "agressivo": 3}
SCENARIO_LEVELS = {"conservador": 1, "moderado": 2, "objetivo": 3}


class RiskSuitabilityAgent(BaseInvestmentCaseAgent):
    agent_name = "risk_suitability"
    instruction = (
        "Revisar diagnóstico e cenários, classificando alertas de suitability, limitações e pontos que exigem validação humana."
    )

    def _build_fallback_review(self, *, case_state: dict) -> dict:
        suitability = (case_state.get("selected_client_context", {}).get("client_summary", {}).get("perfil_suitability") or "Não informado").lower()
        scenario_list = case_state.get("scenarios", [])
        profile_level = PROFILE_LEVELS.get(suitability, 2)
        alerts = []
        human_review = []

        for scenario in scenario_list:
            scenario_level = SCENARIO_LEVELS.get(scenario.get("profile_anchor") or scenario.get("scenario_type"), 2)
            if scenario_level > profile_level:
                alerts.append(
                    {
                        "scenario": scenario.get("name"),
                        "severity": "high",
                        "type": "suitability",
                        "message": "Cenário pode exceder o apetite de risco cadastrado e deve ser revalidado antes da recomendação formal.",
                        "recommended_action": "Validar perfil, horizonte e aceitação de volatilidade antes de seguir.",
                    }
                )
                human_review.append(f"Validar suitability do cenário '{scenario.get('name')}' em relação ao cadastro atual ({suitability}).")
            else:
                alerts.append(
                    {
                        "scenario": scenario.get("name"),
                        "severity": "medium",
                        "type": "implementation",
                        "message": "Cenário parece compatível no nível macro, mas ainda requer confirmação de liquidez, tributação e horizonte.",
                        "recommended_action": "Validar operacionalização antes da implementação.",
                    }
                )

        conflicts = case_state.get("data_relevance_decisions", {}).get("conflicts_detected", [])
        if conflicts:
            human_review.append("Existem conflitos entre prompt do assessor e dados cadastrais que precisam de validação explícita.")

        return {
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

    def run(self, *, case_state: dict) -> AgentResult:
        fallback = self._build_fallback_review(case_state=case_state)
        system_prompt, user_prompt = build_risk_prompts(case_state=case_state, heuristic_baseline=fallback)
        llm_payload = try_json_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=self.temperature,
        )
        payload = llm_payload if isinstance(llm_payload, dict) and llm_payload.get("overall_status") else fallback
        return AgentResult(payload=payload, summary="Riscos e suitability revisados com classificação de severidade e pontos de validação humana.")
