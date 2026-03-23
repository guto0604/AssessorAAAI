from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent


class PlannerAgent(BaseInvestmentCaseAgent):
    agent_name = "planner"
    instruction = (
        "Definir um plano rastreável para executar os agentes restantes, com etapas, dependências, paralelismo possível e saídas esperadas."
    )

    def run(self, *, case_state: dict) -> AgentResult:
        prompt = case_state.get("advisor_prompt", "")
        themes = case_state.get("data_relevance_decisions", {}).get("selected_context", {}).get("relevant_behavioral_signals", {}).get("themes", [])
        scenario_focus = "objetivo principal do assessor"
        if themes:
            scenario_focus = ", ".join(themes)

        steps = [
            {
                "step_id": "portfolio_diagnosis",
                "objective": "Mapear situação atual, concentração, liquidez e ineficiências.",
                "dependencies": ["data_relevance"],
                "can_run_in_parallel": False,
                "expected_output": "Diagnóstico estruturado da carteira atual.",
            },
            {
                "step_id": "scenario_builder",
                "objective": f"Construir cenários aderentes ao prompt: {prompt[:120] or scenario_focus}.",
                "dependencies": ["portfolio_diagnosis"],
                "can_run_in_parallel": False,
                "expected_output": "Ao menos três cenários comparáveis.",
            },
            {
                "step_id": "risk_suitability",
                "objective": "Revisar cenários com foco em suitability, compliance e alertas.",
                "dependencies": ["portfolio_diagnosis", "scenario_builder"],
                "can_run_in_parallel": False,
                "expected_output": "Alertas classificados por severidade.",
            },
            {
                "step_id": "narrative_proposal",
                "objective": "Transformar diagnóstico e cenários em proposta consultiva pronta para reunião.",
                "dependencies": ["portfolio_diagnosis", "scenario_builder", "risk_suitability"],
                "can_run_in_parallel": False,
                "expected_output": "Resumo executivo, proposta central e próximos passos.",
            },
            {
                "step_id": "visualization",
                "objective": "Gerar gráficos úteis e seus metadados a partir do estado compartilhado.",
                "dependencies": ["portfolio_diagnosis", "scenario_builder"],
                "can_run_in_parallel": True,
                "expected_output": "Especificações estruturadas de gráficos e renderização plotly.",
            },
            {
                "step_id": "pdf_builder",
                "objective": "Montar o PDF final a partir do estado consolidado, sem novo raciocínio solto.",
                "dependencies": ["narrative_proposal", "risk_suitability", "visualization"],
                "can_run_in_parallel": False,
                "expected_output": "Arquivo PDF local pronto para download.",
            },
        ]

        payload = {
            "summary": "Plano linear com validação de risco antes da proposta e geração de artefatos ao final.",
            "scenario_focus": scenario_focus,
            "steps": steps,
        }
        return AgentResult(payload=payload, summary=payload["summary"])
