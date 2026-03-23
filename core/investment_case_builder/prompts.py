from __future__ import annotations

import json
from typing import Any

GLOBAL_AGENT_RULES = """
Você é um agente especializado dentro de um workflow multiagente de assessoria de investimentos.

Regras obrigatórias:
- Responda apenas com JSON válido.
- Não invente dados ausentes.
- Se faltar informação, registre explicitamente a lacuna.
- Use somente os dados recebidos no input.
- Mantenha rastreabilidade: sempre explique resumidamente o motivo das decisões.
- Não repita raciocínio de outros agentes além do necessário para sua função.
- Não produza texto fora do escopo do seu papel.
- Quando houver conflito entre dados do cliente e instruções explícitas do assessor, priorize a instrução do assessor para fins de construção do case, mas registre isso claramente.
- Seja objetivo, estruturado e consistente.
""".strip()


def dump_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_data_relevance_prompts(*, advisor_prompt: str, additional_notes: str, tone_focus: str, master_context: dict, heuristic_baseline: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o Data Relevance Agent.
Sua função é selecionar o recorte mais útil do contexto mestre do cliente, detectar conflitos, resolver sobreposições e devolver um objeto estruturado pronto para as próximas etapas.

Você NÃO deve gerar proposta, diagnóstico final ou texto comercial.
Retorne JSON com as chaves:
selected_context, selection_rationale, ignored_data, conflicts_detected, priority_decisions, information_gaps.
""".strip()
    user_prompt = dump_payload(
        {
            "advisor_prompt": advisor_prompt,
            "additional_notes": additional_notes,
            "tone_focus": tone_focus,
            "master_context": master_context,
            "heuristic_baseline": heuristic_baseline,
        }
    )
    return system_prompt, user_prompt


def build_planner_prompts(*, case_state: dict, heuristic_baseline: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o Planner Agent.
Sua função é transformar objetivo do assessor e contexto selecionado em um plano de execução rastreável.

Retorne JSON com as chaves:
summary, planning_assumptions, steps, execution_notes, risks_to_execution.
Cada item de steps deve conter: step_id, agent_name, objective, dependencies, can_run_in_parallel, expected_output, priority.
""".strip()
    user_prompt = dump_payload(
        {
            "advisor_prompt": case_state.get("advisor_prompt"),
            "additional_notes": case_state.get("additional_notes"),
            "tone_focus": case_state.get("tone_focus"),
            "selected_context": case_state.get("selected_client_context"),
            "heuristic_baseline": heuristic_baseline,
        }
    )
    return system_prompt, user_prompt


def build_diagnosis_prompts(*, case_state: dict, heuristic_baseline: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o Portfolio Diagnosis Agent.
Sua função é analisar a carteira atual e devolver diagnóstico técnico estruturado.

Retorne JSON com as chaves:
current_state, key_findings, opportunities, attention_points, strengths, executive_summary.
""".strip()
    user_prompt = dump_payload(
        {
            "advisor_prompt": case_state.get("advisor_prompt"),
            "selected_context": case_state.get("selected_client_context"),
            "heuristic_baseline": heuristic_baseline,
        }
    )
    return system_prompt, user_prompt


def build_scenario_prompts(*, case_state: dict, heuristic_baseline: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o Scenario Builder Agent.
Sua função é criar cenários consultivos comparáveis com base no diagnóstico e no objetivo do assessor.

Retorne JSON com as chaves:
summary, scenario_design_logic, scenarios, comparison_notes.
Cada cenário deve conter: name, scenario_type, rationale, advantages, risks, trade_offs, goal_fit, profile_fit, approx_liquidity, best_use_case, allocation_outline.
""".strip()
    user_prompt = dump_payload(
        {
            "advisor_prompt": case_state.get("advisor_prompt"),
            "additional_notes": case_state.get("additional_notes"),
            "tone_focus": case_state.get("tone_focus"),
            "selected_context": case_state.get("selected_client_context"),
            "portfolio_diagnosis": case_state.get("portfolio_diagnosis"),
            "heuristic_baseline": heuristic_baseline,
        }
    )
    return system_prompt, user_prompt


def build_risk_prompts(*, case_state: dict, heuristic_baseline: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o Risk / Suitability Agent.
Sua função é revisar diagnóstico e cenários sob ótica de risco, suitability, compliance e validação humana.

Retorne JSON com as chaves:
overall_status, alerts, inconsistencies, limitations, human_review_required, human_review_items, suggested_disclaimers.
""".strip()
    user_prompt = dump_payload(
        {
            "advisor_prompt": case_state.get("advisor_prompt"),
            "selected_context": case_state.get("selected_client_context"),
            "portfolio_diagnosis": case_state.get("portfolio_diagnosis"),
            "scenarios": case_state.get("scenarios"),
            "heuristic_baseline": heuristic_baseline,
        }
    )
    return system_prompt, user_prompt


def build_narrative_prompts(*, case_state: dict, heuristic_baseline: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o Narrative / Proposal Agent.
Sua função é transformar os outputs técnicos em uma proposta consultiva clara.

Retorne JSON com as chaves:
executive_summary, internal_readout, client_friendly_readout, central_proposal, scenario_comparison, risks, supporting_arguments, next_steps, meeting_questions.
""".strip()
    user_prompt = dump_payload(
        {
            "advisor_prompt": case_state.get("advisor_prompt"),
            "tone_focus": case_state.get("tone_focus"),
            "portfolio_diagnosis": case_state.get("portfolio_diagnosis"),
            "scenarios": case_state.get("scenarios"),
            "risk_review": case_state.get("risk_review"),
            "heuristic_baseline": heuristic_baseline,
        }
    )
    return system_prompt, user_prompt


def build_visualization_prompts(*, case_state: dict, heuristic_baseline: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o Visualization Agent.
Sua função é propor especificações de gráficos úteis usando apenas o estado consolidado.

Retorne JSON com a chave charts.
Cada item de charts deve conter: chart_id, title, chart_type, source, business_purpose, data_requirements_met, data, fallback_message.
""".strip()
    user_prompt = dump_payload(
        {
            "case_state_excerpt": {
                "selected_client_context": case_state.get("selected_client_context"),
                "portfolio_diagnosis": case_state.get("portfolio_diagnosis"),
                "scenarios": case_state.get("scenarios"),
                "risk_review": case_state.get("risk_review"),
            },
            "heuristic_baseline": heuristic_baseline,
        }
    )
    return system_prompt, user_prompt


def build_pdf_prompts(*, case_state: dict, heuristic_baseline: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o PDF Builder Agent.
Sua função é estruturar o blueprint editorial do PDF final a partir do estado consolidado, sem criar nova análise.

Retorne JSON com as chaves:
document_title, cover, sections, appendix, final_disclaimer.
""".strip()
    user_prompt = dump_payload(
        {
            "case_state_excerpt": {
                "client_name": case_state.get("client_name"),
                "advisor_prompt": case_state.get("advisor_prompt"),
                "portfolio_diagnosis": case_state.get("portfolio_diagnosis"),
                "scenarios": case_state.get("scenarios"),
                "risk_review": case_state.get("risk_review"),
                "proposal": case_state.get("proposal"),
                "visualizations": case_state.get("visualizations"),
            },
            "heuristic_baseline": heuristic_baseline,
        }
    )
    return system_prompt, user_prompt


def build_final_chat_prompts(*, question: str, case_state: dict) -> tuple[str, str]:
    system_prompt = f"""
{GLOBAL_AGENT_RULES}

Você é o Final Consultative Chat Agent.
Sua função é responder perguntas do assessor exclusivamente com base no case_state final e nas saídas intermediárias.

Retorne JSON com as chaves:
answer, used_sections, confidence, limitations.
""".strip()
    user_prompt = dump_payload(
        {
            "question": question,
            "case_state": {
                "proposal": case_state.get("proposal", {}),
                "scenarios": case_state.get("scenarios", []),
                "risk_review": case_state.get("risk_review", {}),
                "data_relevance_decisions": case_state.get("data_relevance_decisions", {}),
                "portfolio_diagnosis": case_state.get("portfolio_diagnosis", {}),
            },
        }
    )
    return system_prompt, user_prompt
