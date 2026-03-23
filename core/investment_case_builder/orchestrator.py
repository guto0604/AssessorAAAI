from __future__ import annotations

from core.investment_case_builder.client_context import load_client_master_context, select_relevant_client_context
from core.investment_case_builder.data_relevance_agent import DataRelevanceAgent
from core.investment_case_builder.final_chat_agent import FinalConsultativeChatAgent
from core.investment_case_builder.narrative_proposal_agent import NarrativeProposalAgent
from core.investment_case_builder.pdf_builder_agent import PDFBuilderAgent
from core.investment_case_builder.planner_agent import PlannerAgent
from core.investment_case_builder.portfolio_diagnosis_agent import PortfolioDiagnosisAgent
from core.investment_case_builder.risk_suitability_agent import RiskSuitabilityAgent
from core.investment_case_builder.scenario_builder_agent import ScenarioBuilderAgent
from core.investment_case_builder.state import WORKFLOW_STEP_ORDER, append_audit_log, build_case_state, utc_now_iso
from core.investment_case_builder.visualization_agent import VisualizationAgent


class InvestmentCaseOrchestrator:
    def __init__(self):
        self.data_relevance_agent = DataRelevanceAgent()
        self.planner_agent = PlannerAgent()
        self.portfolio_diagnosis_agent = PortfolioDiagnosisAgent()
        self.scenario_builder_agent = ScenarioBuilderAgent()
        self.risk_suitability_agent = RiskSuitabilityAgent()
        self.narrative_proposal_agent = NarrativeProposalAgent()
        self.visualization_agent = VisualizationAgent()
        self.pdf_builder_agent = PDFBuilderAgent()
        self.final_chat_agent = FinalConsultativeChatAgent()

    def initialize_case(self, *, client_id: str, client_name: str, advisor_prompt: str, additional_notes: str = "", tone_focus: str = "") -> dict:
        case_state = build_case_state(
            client_id=client_id,
            client_name=client_name,
            advisor_prompt=advisor_prompt,
            additional_notes=additional_notes,
            tone_focus=tone_focus,
        )
        master_context = load_client_master_context(client_id)
        case_state["client_master_context"] = master_context
        append_audit_log(case_state, step="system", event="case_initialized", details={"client_id": client_id})
        return case_state

    def _mark_step(self, case_state: dict, step: str, status: str, details: str | None = None, error: str | None = None) -> None:
        step_state = case_state["workflow_status"][step]
        if status == "running":
            step_state["started_at"] = utc_now_iso()
        if status in {"completed", "error"}:
            step_state["finished_at"] = utc_now_iso()
        step_state["status"] = status
        step_state["details"] = details
        step_state["error"] = error

    def _run_data_relevance(self, case_state: dict) -> None:
        result = select_relevant_client_context(
            case_state["client_master_context"],
            case_state["advisor_prompt"],
            additional_notes=case_state.get("additional_notes", ""),
            tone_focus=case_state.get("tone_focus", ""),
        )
        case_state["selected_client_context"] = result["selected_context"]
        case_state["data_relevance_decisions"] = result

    def _run_planner(self, case_state: dict) -> None:
        result = self.planner_agent.run(case_state=case_state)
        case_state["workflow_plan"] = result.payload

    def _run_portfolio_diagnosis(self, case_state: dict) -> None:
        result = self.portfolio_diagnosis_agent.run(case_state=case_state)
        case_state["portfolio_diagnosis"] = result.payload

    def _run_scenarios(self, case_state: dict) -> None:
        result = self.scenario_builder_agent.run(case_state=case_state)
        case_state["scenarios"] = result.payload["scenarios"]

    def _run_risk_review(self, case_state: dict) -> None:
        result = self.risk_suitability_agent.run(case_state=case_state)
        case_state["risk_review"] = result.payload

    def _run_proposal(self, case_state: dict) -> None:
        result = self.narrative_proposal_agent.run(case_state=case_state)
        case_state["proposal"] = result.payload
        case_state["next_steps"] = result.payload.get("next_steps", [])
        case_state["chat_context"] = {
            "executive_summary": result.payload.get("executive_summary"),
            "scenario_names": [scenario.get("name") for scenario in case_state.get("scenarios", [])],
        }

    def _run_visualizations(self, case_state: dict) -> None:
        result = self.visualization_agent.run(case_state=case_state)
        case_state["visualizations"] = result.payload["charts"]

    def _run_pdf(self, case_state: dict) -> None:
        result = self.pdf_builder_agent.run(case_state=case_state)
        case_state["pdf_path"] = result.payload["pdf_path"]
        case_state["pdf_blueprint"] = result.payload.get("pdf_blueprint", {})

    def run_full_workflow(self, case_state: dict, *, start_from: str = "data_relevance") -> dict:
        step_handlers = {
            "data_relevance": self._run_data_relevance,
            "planner": self._run_planner,
            "portfolio_diagnosis": self._run_portfolio_diagnosis,
            "scenario_builder": self._run_scenarios,
            "risk_suitability": self._run_risk_review,
            "narrative_proposal": self._run_proposal,
            "visualization": self._run_visualizations,
            "pdf_builder": self._run_pdf,
        }
        start_index = WORKFLOW_STEP_ORDER.index(start_from)
        for step in WORKFLOW_STEP_ORDER[start_index:]:
            append_audit_log(case_state, step=step, event="step_started")
            self._mark_step(case_state, step, "running")
            try:
                step_handlers[step](case_state)
                self._mark_step(case_state, step, "completed", details="Etapa concluída com sucesso.")
                append_audit_log(case_state, step=step, event="step_completed")
            except Exception as exc:
                self._mark_step(case_state, step, "error", error=str(exc))
                case_state.setdefault("errors", []).append({"step": step, "error": str(exc)})
                append_audit_log(case_state, step=step, event="step_failed", details={"error": str(exc)})
                break
        case_state["updated_at"] = utc_now_iso()
        return case_state

    def answer_chat(self, *, case_state: dict, question: str) -> str:
        return self.final_chat_agent.answer(question=question, case_state=case_state)
