from __future__ import annotations

import re
from typing import Any

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent
from core.investment_case_builder.llm_support import try_json_completion

PROFILE_LEVELS = {"conservador": 1, "moderado": 2, "arrojado": 3, "agressivo": 3}
PROMPT_PATTERN_MAP = {
    "liquidez": ["liquidez", "caixa", "resgate", "curto prazo"],
    "renda": ["renda", "cupom", "fluxo", "proventos"],
    "crescimento": ["crescimento", "valorização", "longo prazo", "apreciação"],
    "diversificacao": ["diversificação", "diversificar", "concentração", "rebalancear"],
    "proteção": ["proteção", "defensivo", "segurança", "preservação"],
}


class DataRelevanceAgent(BaseInvestmentCaseAgent):
    agent_name = "data_relevance"
    instruction = (
        "Selecionar o recorte mais útil do contexto do cliente, priorizando dados existentes do sistema, "
        "registrando conflitos e explicitando sobreposições quando o prompt do assessor introduzir novas premissas."
    )

    @staticmethod
    def _detect_prompt_signals(text: str) -> dict[str, Any]:
        lowered = (text or "").lower()
        detected = []
        for label, keywords in PROMPT_PATTERN_MAP.items():
            if any(keyword in lowered for keyword in keywords):
                detected.append(label)

        inferred_profile = None
        for profile in PROFILE_LEVELS:
            if profile in lowered:
                inferred_profile = profile
                break

        amounts = re.findall(r"(?:r\$\s*)?(\d+[\d\.,]*)", lowered)
        mentioned_amounts = []
        for raw in amounts:
            normalized = raw.replace(".", "").replace(",", ".")
            try:
                mentioned_amounts.append(float(normalized))
            except ValueError:
                continue

        return {
            "themes": detected,
            "inferred_profile": inferred_profile,
            "mentioned_amounts": mentioned_amounts,
        }

    def run(self, *, master_context: dict, advisor_prompt: str, additional_notes: str = "", tone_focus: str = "") -> AgentResult:
        prompt_signals = self._detect_prompt_signals(" ".join([advisor_prompt, additional_notes, tone_focus]))
        client_info = master_context.get("client_info", {})
        portfolio = master_context.get("portfolio", {})
        allocation = portfolio.get("allocation_by_category", [])
        holdings = portfolio.get("holdings", [])
        suitability = (client_info.get("Perfil_Suitability") or "Não informado").lower()

        selected_categories = allocation[:3]
        if "liquidez" in prompt_signals["themes"]:
            selected_categories = sorted(allocation, key=lambda item: item.get("liquidity_hint") != "alta")[:3] or allocation[:3]
        elif "diversificacao" in prompt_signals["themes"]:
            selected_categories = allocation[:5]

        selected_holdings = sorted(holdings, key=lambda item: item.get("invested_amount", 0), reverse=True)[:5]
        ignored_categories = allocation[3:] if len(allocation) > 3 else []
        gaps = []
        if not holdings:
            gaps.append("Carteira atual não possui posições disponíveis na base local.")
        if not client_info.get("Perfil_Suitability"):
            gaps.append("Perfil de suitability ausente.")
        if not client_info.get("Rentabilidade_12_meses"):
            gaps.append("Rentabilidade de 12 meses ausente ou não preenchida.")

        conflicts = []
        priority_decisions = []
        if prompt_signals.get("inferred_profile") and prompt_signals["inferred_profile"] != suitability:
            conflicts.append(
                {
                    "field": "Perfil_Suitability",
                    "client_value": client_info.get("Perfil_Suitability"),
                    "prompt_value": prompt_signals["inferred_profile"],
                    "reason": "O prompt sugere um apetite de risco diferente do cadastro atual.",
                }
            )
            priority_decisions.append(
                {
                    "field": "Perfil_Suitability",
                    "decision": "prompt_override_for_case_hypothesis",
                    "applied_value": prompt_signals["inferred_profile"],
                    "justification": "A diretriz do assessor prevalece para modelagem do case, mantendo rastreabilidade do conflito com o cadastro.",
                }
            )

        if prompt_signals.get("mentioned_amounts"):
            prompt_amount = max(prompt_signals["mentioned_amounts"])
            available = float(client_info.get("Dinheiro_Disponivel_Para_Investir") or 0.0)
            if prompt_amount > available and available > 0:
                conflicts.append(
                    {
                        "field": "Dinheiro_Disponivel_Para_Investir",
                        "client_value": available,
                        "prompt_value": prompt_amount,
                        "reason": "O prompt menciona um montante maior do que o caixa identificado na base local.",
                    }
                )
                priority_decisions.append(
                    {
                        "field": "Dinheiro_Disponivel_Para_Investir",
                        "decision": "prompt_override_for_case_hypothesis",
                        "applied_value": prompt_amount,
                        "justification": "O caso usará o valor citado pelo assessor como hipótese tática, sinalizando necessidade de validação humana.",
                    }
                )

        selected_context = {
            "client_summary": {
                "client_id": client_info.get("Cliente_ID"),
                "client_name": client_info.get("Nome"),
                "patrimonio_conosco": client_info.get("Patrimonio_Investido_Conosco"),
                "patrimonio_outros": client_info.get("Patrimonio_Investido_Outros"),
                "dinheiro_disponivel": client_info.get("Dinheiro_Disponivel_Para_Investir"),
                "perfil_suitability": client_info.get("Perfil_Suitability"),
                "rentabilidade_12_meses": client_info.get("Rentabilidade_12_meses"),
                "cdi_12_meses": client_info.get("CDI_12_Meses"),
            },
            "relevant_financial_data": selected_categories,
            "relevant_holdings": selected_holdings,
            "relevant_restrictions": [
                "Respeitar perfil de suitability cadastrado, salvo hipóteses explicitamente instruídas pelo assessor.",
                "Validar disponibilidade de caixa e liquidez antes da implementação final.",
            ],
            "relevant_behavioral_signals": prompt_signals,
        }

        llm_payload = try_json_completion(
            system_prompt=(
                "Você é o Data Relevance Agent de uma plataforma de assessoria. "
                "Revise o contexto estruturado e devolva JSON com chaves selected_context_summary, ignored_data_summary e rationale."
            ),
            user_prompt=str({
                "advisor_prompt": advisor_prompt,
                "additional_notes": additional_notes,
                "tone_focus": tone_focus,
                "selected_context": selected_context,
                "conflicts": conflicts,
                "priority_decisions": priority_decisions,
                "gaps": gaps,
            }),
            model=self.model,
            temperature=self.temperature,
        )

        payload = {
            "selected_context": selected_context,
            "selection_rationale": llm_payload or {
                "selected_context_summary": "Foram priorizados perfil, liquidez disponível, concentração atual e categorias mais relevantes ao objetivo do assessor.",
                "ignored_data_summary": "Categorias menos materiais para o objetivo do caso ficaram fora do recorte principal, mas permanecem disponíveis no estado mestre.",
                "rationale": "O agente iniciou pelo contexto cadastral e de carteira e só aplicou sobreposição quando o prompt trouxe hipótese explícita conflitante.",
            },
            "ignored_data": {
                "categories": ignored_categories,
                "holdings": sorted(holdings, key=lambda item: item.get("invested_amount", 0), reverse=True)[5:],
            },
            "conflicts_detected": conflicts,
            "priority_decisions": priority_decisions,
            "information_gaps": gaps,
        }
        summary = payload["selection_rationale"]["selected_context_summary"]
        return AgentResult(payload=payload, summary=summary)
