from pathlib import Path

DEFAULT_AGENT_MODEL = "gpt-5-mini"
DEFAULT_AGENT_TEMPERATURE = 1.0
CASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "generated" / "investment_cases"

AGENT_CONFIG = {
    "data_relevance": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
    "planner": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
    "portfolio_diagnosis": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
    "scenario_builder": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
    "risk_suitability": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
    "narrative_proposal": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
    "visualization": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
    "pdf_builder": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
    "final_consultative_chat": {"model": DEFAULT_AGENT_MODEL, "temperature": DEFAULT_AGENT_TEMPERATURE},
}
