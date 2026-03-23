from __future__ import annotations

from core.investment_case_builder.agents_base import BaseInvestmentCaseAgent
from core.investment_case_builder.llm_support import try_json_completion


class FinalConsultativeChatAgent(BaseInvestmentCaseAgent):
    agent_name = "final_consultative_chat"
    instruction = (
        "Responder apenas com base no case_state final e nas saídas intermediárias, explicando decisões, riscos, priorizações e ajustes sugeridos."
    )

    def _fallback_answer(self, *, question: str, case_state: dict) -> str:
        lowered = (question or "").lower()
        proposal = case_state.get("proposal", {})
        risk_review = case_state.get("risk_review", {})
        context = case_state.get("data_relevance_decisions", {})
        scenarios = case_state.get("scenarios", [])

        if "risco" in lowered or "alerta" in lowered:
            alerts = risk_review.get("alerts", [])
            if not alerts:
                return "Não há alertas estruturados no case atual."
            return "Principais alertas considerados: " + " | ".join(
                f"{item.get('scenario')}: {item.get('message')}" for item in alerts[:3]
            )

        if "dado" in lowered or "contexto" in lowered or "prioriz" in lowered or "sobrepos" in lowered:
            decisions = context.get("priority_decisions", [])
            conflicts = context.get("conflicts_detected", [])
            base = context.get("selection_rationale", {}).get("rationale", "O recorte priorizou o que era mais material para o objetivo do assessor.")
            if decisions or conflicts:
                return base + " Conflitos/sobreposições: " + " | ".join(
                    f"{item.get('field')}: {item.get('decision', item.get('reason'))}" for item in (decisions or conflicts)[:3]
                )
            return base

        if "cenário" in lowered or "cenario" in lowered:
            if not scenarios:
                return "Os cenários ainda não foram gerados para este case."
            return "Os cenários foram criados para oferecer comparação entre preservação, eficiência moderada e uma tese tática focada no objetivo do assessor. " + " | ".join(
                f"{scenario.get('name')}: {scenario.get('rationale')}" for scenario in scenarios[:3]
            )

        if "reunião" in lowered or "resum" in lowered:
            return proposal.get("executive_summary", "Resumo executivo indisponível no case atual.")

        return (
            "Posso responder com base no case gerado sobre cenário, risco, priorização de dados, proposta central ou resumo para reunião. "
            f"Proposta central atual: {proposal.get('central_proposal', {}).get('recommended_scenario', 'não definida')}."
        )

    def answer(self, *, question: str, case_state: dict) -> str:
        llm_payload = try_json_completion(
            system_prompt=(
                "Você é o Final Consultative Chat Agent. Responda somente com base no case_state recebido. "
                "Devolva JSON com a chave answer. Se faltar base, explique a limitação em vez de inventar."
            ),
            user_prompt=str({
                "question": question,
                "case_state": {
                    "proposal": case_state.get("proposal", {}),
                    "scenarios": case_state.get("scenarios", []),
                    "risk_review": case_state.get("risk_review", {}),
                    "data_relevance_decisions": case_state.get("data_relevance_decisions", {}),
                    "portfolio_diagnosis": case_state.get("portfolio_diagnosis", {}),
                },
            }),
            model=self.model,
            temperature=self.temperature,
        )
        if llm_payload and llm_payload.get("answer"):
            return llm_payload["answer"]
        return self._fallback_answer(question=question, case_state=case_state)
