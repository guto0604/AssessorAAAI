from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent


class NarrativeProposalAgent(BaseInvestmentCaseAgent):
    agent_name = "narrative_proposal"
    instruction = (
        "Transformar o material técnico em narrativa consultiva clara, com visão executiva, comparação de cenários e próximos passos de reunião."
    )

    def run(self, *, case_state: dict) -> AgentResult:
        diagnosis = case_state.get("portfolio_diagnosis", {})
        scenarios = case_state.get("scenarios", [])
        risk_review = case_state.get("risk_review", {})
        advisor_prompt = case_state.get("advisor_prompt", "")
        tone_focus = case_state.get("tone_focus", "consultivo")

        central_scenario = scenarios[1] if len(scenarios) > 1 else (scenarios[0] if scenarios else {})
        next_steps = [
            "Validar com o cliente o objetivo prioritário, horizonte e restrições de liquidez.",
            "Apresentar os três cenários de forma comparativa, destacando trade-offs e suitability.",
            "Confirmar quais hipóteses do prompt dependem de atualização cadastral ou aprovação humana.",
        ]

        proposal = {
            "executive_summary": (
                f"O Investment Case foi estruturado para responder ao objetivo '{advisor_prompt}', "
                "conectando diagnóstico atual, alternativas de alocação e alertas de suitability em um fluxo rastreável."
            ),
            "internal_readout": diagnosis.get("executive_summary", ""),
            "client_friendly_readout": "Há uma oportunidade de reorganizar a carteira com mais clareza sobre risco, liquidez e aderência ao objetivo atual.",
            "central_proposal": {
                "recommended_scenario": central_scenario.get("name"),
                "why": central_scenario.get("rationale"),
                "tone_focus": tone_focus or "consultivo",
            },
            "scenario_comparison": [
                {
                    "scenario": scenario.get("name"),
                    "best_use_case": scenario.get("best_use_case"),
                    "key_trade_off": (scenario.get("trade_offs") or [""])[0],
                }
                for scenario in scenarios
            ],
            "risks": risk_review.get("alerts", []),
            "supporting_arguments": diagnosis.get("opportunities", []),
            "next_steps": next_steps,
            "meeting_questions": [
                "Qual objetivo precisa ser priorizado nos próximos 6 a 12 meses?",
                "Qual nível de liquidez o cliente deseja preservar após a eventual realocação?",
                "Há abertura para uma transição gradual entre cenários ou a implementação deve ser mais imediata?",
            ],
        }
        return AgentResult(payload=proposal, summary=proposal["executive_summary"])
