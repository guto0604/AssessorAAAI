from __future__ import annotations

from dataclasses import dataclass

from core.investment_case_builder.config import AGENT_CONFIG


@dataclass
class AgentResult:
    payload: dict
    summary: str


class BaseInvestmentCaseAgent:
    agent_name = "base"
    instruction = ""

    def __init__(self, *, model: str | None = None, temperature: float | None = None):
        defaults = AGENT_CONFIG.get(self.agent_name, {})
        self.model = model or defaults.get("model")
        self.temperature = temperature if temperature is not None else defaults.get("temperature")

    def describe(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "instruction": self.instruction,
            "model": self.model,
            "temperature": self.temperature,
        }
