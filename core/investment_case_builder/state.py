from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

WORKFLOW_STEP_ORDER = [
    "data_relevance",
    "planner",
    "portfolio_diagnosis",
    "scenario_builder",
    "risk_suitability",
    "narrative_proposal",
    "visualization",
    "pdf_builder",
]

WORKFLOW_STEP_LABELS = {
    "data_relevance": "Data Relevance Agent",
    "planner": "Planner Agent",
    "portfolio_diagnosis": "Portfolio Diagnosis Agent",
    "scenario_builder": "Scenario Builder Agent",
    "risk_suitability": "Risk / Suitability Agent",
    "narrative_proposal": "Narrative / Proposal Agent",
    "visualization": "Visualization Agent",
    "pdf_builder": "PDF Builder Agent",
    "final_consultative_chat": "Final Consultative Chat Agent",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_empty_workflow_status() -> dict:
    return {
        step: {
            "label": WORKFLOW_STEP_LABELS[step],
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "error": None,
            "details": None,
        }
        for step in WORKFLOW_STEP_ORDER
    }


def build_case_state(
    *,
    client_id: str,
    advisor_prompt: str,
    additional_notes: str = "",
    tone_focus: str = "",
    client_name: str = "",
) -> dict:
    return {
        "case_id": f"case_{uuid4().hex[:12]}",
        "client_id": client_id,
        "client_name": client_name,
        "advisor_prompt": advisor_prompt,
        "additional_notes": additional_notes,
        "tone_focus": tone_focus,
        "selected_client_context": {},
        "client_master_context": {},
        "data_relevance_decisions": {},
        "workflow_plan": {},
        "workflow_status": build_empty_workflow_status(),
        "portfolio_diagnosis": {},
        "scenarios": [],
        "risk_review": {},
        "proposal": {},
        "visualizations": [],
        "pdf_path": None,
        "next_steps": [],
        "audit_log": [],
        "chat_context": {},
        "errors": [],
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def clone_case_state(case_state: dict) -> dict:
    return deepcopy(case_state)


def append_audit_log(case_state: dict, *, step: str, event: str, details: dict | None = None) -> None:
    case_state.setdefault("audit_log", []).append(
        {
            "timestamp": utc_now_iso(),
            "step": step,
            "event": event,
            "details": details or {},
        }
    )
    case_state["updated_at"] = utc_now_iso()
