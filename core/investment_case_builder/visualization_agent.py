from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent
from core.investment_case_builder.llm_support import try_json_completion
from core.investment_case_builder.prompts import build_visualization_prompts


class VisualizationAgent(BaseInvestmentCaseAgent):
    agent_name = "visualization"
    instruction = (
        "Gerar especificações estruturadas e figuras plotly com base apenas no estado consolidado do case."
    )

    @staticmethod
    def _build_current_allocation_chart(categories: list[dict]) -> dict:
        if not categories:
            return {
                "chart_id": "current_allocation",
                "title": "Composição atual",
                "chart_type": "fallback",
                "type": "fallback",
                "source": "selected_client_context.relevant_financial_data",
                "business_purpose": "Explicar a composição atual da carteira.",
                "data_requirements_met": False,
                "data": {},
                "fallback_message": "Dados insuficientes para renderizar a composição atual.",
                "message": "Dados insuficientes para renderizar a composição atual.",
            }
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Pie(labels=[c["category"] for c in categories], values=[c["invested_amount"] for c in categories], hole=0.35)])
        fig.update_layout(title="Composição atual da carteira")
        return {
            "chart_id": "current_allocation",
            "title": "Composição atual da carteira",
            "chart_type": "pie",
            "type": "pie",
            "source": "selected_client_context.relevant_financial_data",
            "business_purpose": "Mostrar concentração por categoria na carteira atual.",
            "data_requirements_met": True,
            "data": categories,
            "figure_json": fig.to_json(),
            "fallback_message": "",
        }

    @staticmethod
    def _build_scenario_comparison_chart(scenarios: list[dict]) -> dict:
        if not scenarios:
            return {
                "chart_id": "scenario_comparison",
                "title": "Comparação entre cenários",
                "chart_type": "fallback",
                "type": "fallback",
                "source": "scenarios",
                "business_purpose": "Comparar alternativas do case.",
                "data_requirements_met": False,
                "data": {},
                "fallback_message": "Cenários ainda não disponíveis.",
                "message": "Cenários ainda não disponíveis.",
            }
        import plotly.graph_objects as go

        categories = []
        scenario_names = []
        for scenario in scenarios:
            scenario_names.append(scenario.get("name"))
            outline = {item.get("bucket"): item.get("pct") for item in scenario.get("allocation_outline", [])}
            categories.append(outline)
        all_buckets = sorted({bucket for outline in categories for bucket in outline.keys()})
        fig = go.Figure()
        for bucket in all_buckets:
            fig.add_bar(name=bucket, x=scenario_names, y=[outline.get(bucket, 0) for outline in categories])
        fig.update_layout(title="Comparação entre cenários", barmode="stack")
        return {
            "chart_id": "scenario_comparison",
            "title": "Comparação entre cenários",
            "chart_type": "stacked_bar",
            "type": "stacked_bar",
            "source": "scenarios[*].allocation_outline",
            "business_purpose": "Comparar a construção macro dos cenários.",
            "data_requirements_met": True,
            "data": {"scenario_names": scenario_names, "buckets": all_buckets, "series": categories},
            "figure_json": fig.to_json(),
            "fallback_message": "",
        }

    @staticmethod
    def _build_concentration_chart(holdings: list[dict]) -> dict:
        if not holdings:
            return {
                "chart_id": "concentration",
                "title": "Distribuição/concentração",
                "chart_type": "fallback",
                "type": "fallback",
                "source": "selected_client_context.relevant_holdings",
                "business_purpose": "Exibir concentração por posição.",
                "data_requirements_met": False,
                "data": {},
                "fallback_message": "Sem posições suficientes para analisar concentração.",
                "message": "Sem posições suficientes para analisar concentração.",
            }
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Bar(x=[item["product_name"] for item in holdings], y=[item["invested_amount"] for item in holdings])])
        fig.update_layout(title="Principais posições por valor investido")
        return {
            "chart_id": "concentration",
            "title": "Principais posições por valor investido",
            "chart_type": "bar",
            "type": "bar",
            "source": "selected_client_context.relevant_holdings",
            "business_purpose": "Evidenciar concentração nas maiores posições.",
            "data_requirements_met": True,
            "data": holdings,
            "figure_json": fig.to_json(),
            "fallback_message": "",
        }

    @staticmethod
    def _build_thesis_chart(opportunities: list[str], alerts: list[str]) -> dict:
        import plotly.graph_objects as go

        labels = ["Oportunidades", "Alertas"]
        values = [len(opportunities), len(alerts)]
        fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=["#2E86AB", "#D7263D"])])
        fig.update_layout(title="Balanço da tese consultiva")
        return {
            "chart_id": "thesis_balance",
            "title": "Balanço da tese consultiva",
            "chart_type": "summary_bar",
            "type": "summary_bar",
            "source": "portfolio_diagnosis.opportunities + risk_review.alerts",
            "business_purpose": "Comparar peso de oportunidades versus alertas na tese.",
            "data_requirements_met": True,
            "data": {"labels": labels, "values": values},
            "figure_json": fig.to_json(),
            "fallback_message": "",
        }

    def _build_fallback_charts(self, *, case_state: dict) -> dict:
        selected_context = case_state.get("selected_client_context", {})
        diagnosis = case_state.get("portfolio_diagnosis", {})
        scenarios = case_state.get("scenarios", [])
        return {
            "charts": [
                self._build_current_allocation_chart(selected_context.get("relevant_financial_data", [])),
                self._build_scenario_comparison_chart(scenarios),
                self._build_concentration_chart(selected_context.get("relevant_holdings", [])),
                self._build_thesis_chart(diagnosis.get("opportunities", []), diagnosis.get("attention_points", [])),
            ]
        }

    def run(self, *, case_state: dict) -> AgentResult:
        fallback = self._build_fallback_charts(case_state=case_state)
        system_prompt, user_prompt = build_visualization_prompts(case_state=case_state, heuristic_baseline=fallback)
        llm_payload = try_json_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=self.temperature,
        )

        payload = fallback
        if isinstance(llm_payload, dict) and llm_payload.get("charts"):
            chart_index = {chart["chart_id"]: chart for chart in fallback["charts"]}
            merged = []
            for chart in llm_payload["charts"]:
                chart_id = chart.get("chart_id")
                base = chart_index.get(chart_id)
                if not base:
                    continue
                merged.append({**base, **chart, "type": chart.get("chart_type", base.get("type"))})
            if merged:
                payload = {"charts": merged}
        return AgentResult(payload=payload, summary="Gráficos estruturados e prontos para a tela e para o PDF.")
