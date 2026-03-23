from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent
from core.investment_case_builder.llm_support import try_json_completion
from core.investment_case_builder.prompts import build_planner_prompts


class PlannerAgent(BaseInvestmentCaseAgent):
    agent_name = "planner"
    instruction = (
        "Definir um plano rastreável para executar os agentes restantes, com etapas, dependências, paralelismo possível e saídas esperadas."
    )

    def _build_fallback_plan(self, *, case_state: dict) -> dict:
        prompt = case_state.get("advisor_prompt", "")
        themes = case_state.get("data_relevance_decisions", {}).get("selected_context", {}).get("relevant_behavioral_signals", {}).get("themes", [])
        scenario_focus = "objetivo principal do assessor"
        if themes:
            scenario_focus = ", ".join(themes)

        return {
            "summary": "Plano linear com validação de risco antes da proposta e geração de artefatos ao final.",
            "planning_assumptions": [
                "O Data Relevance Agent já entregou o recorte mínimo necessário para reduzir ruído.",
                "As etapas posteriores devem operar apenas no estado compartilhado consolidado.",
            ],
            "steps": [
                {
                    "step_id": "portfolio_diagnosis",
                    "agent_name": "Portfolio Diagnosis Agent",
                    "objective": "Mapear situação atual, concentração, liquidez e ineficiências.",
                    "dependencies": ["data_relevance"],
                    "can_run_in_parallel": False,
                    "expected_output": "Diagnóstico estruturado da carteira atual.",
                    "priority": 1,
                },
                {
                    "step_id": "scenario_builder",
                    "agent_name": "Scenario Builder Agent",
                    "objective": f"Construir cenários aderentes ao prompt: {prompt[:120] or scenario_focus}.",
                    "dependencies": ["portfolio_diagnosis"],
                    "can_run_in_parallel": False,
                    "expected_output": "Ao menos três cenários comparáveis.",
                    "priority": 2,
                },
                {
                    "step_id": "risk_suitability",
                    "agent_name": "Risk / Suitability Agent",
                    "objective": "Revisar cenários com foco em suitability, compliance e alertas.",
                    "dependencies": ["portfolio_diagnosis", "scenario_builder"],
                    "can_run_in_parallel": False,
                    "expected_output": "Alertas classificados por severidade.",
                    "priority": 3,
                },
                {
                    "step_id": "narrative_proposal",
                    "agent_name": "Narrative / Proposal Agent",
                    "objective": "Transformar diagnóstico e cenários em proposta consultiva pronta para reunião.",
                    "dependencies": ["portfolio_diagnosis", "scenario_builder", "risk_suitability"],
                    "can_run_in_parallel": False,
                    "expected_output": "Resumo executivo, proposta central e próximos passos.",
                    "priority": 4,
                },
                {
                    "step_id": "visualization",
                    "agent_name": "Visualization Agent",
                    "objective": "Gerar gráficos úteis e seus metadados a partir do estado compartilhado.",
                    "dependencies": ["portfolio_diagnosis", "scenario_builder"],
                    "can_run_in_parallel": True,
                    "expected_output": "Especificações estruturadas de gráficos e renderização plotly.",
                    "priority": 5,
                },
                {
                    "step_id": "pdf_builder",
                    "agent_name": "PDF Builder Agent",
                    "objective": "Montar o PDF final a partir do estado consolidado, sem novo raciocínio solto.",
                    "dependencies": ["narrative_proposal", "risk_suitability", "visualization"],
                    "can_run_in_parallel": False,
                    "expected_output": "Arquivo PDF local pronto para download.",
                    "priority": 6,
                },
            ],
            "execution_notes": [
                "A etapa de visualização deve consumir apenas métricas presentes no case_state.",
                "O PDF deve ser gerado somente após proposta, riscos e gráficos estarem consolidados.",
            ],
            "risks_to_execution": [
                "Lacunas de dados do cliente podem reduzir a precisão dos cenários.",
                "Conflitos entre prompt e cadastro exigem revisão humana antes da implementação final.",
            ],
        }

    def run(self, *, case_state: dict) -> AgentResult:
        fallback = self._build_fallback_plan(case_state=case_state)
        system_prompt, user_prompt = build_planner_prompts(case_state=case_state, heuristic_baseline=fallback)
        llm_payload = try_json_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=self.temperature,
        )
        payload = llm_payload if isinstance(llm_payload, dict) and llm_payload.get("steps") else fallback
        return AgentResult(payload=payload, summary=payload["summary"])
