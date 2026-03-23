from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent


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
                "type": "fallback",
                "source": "selected_client_context.relevant_financial_data",
                "message": "Dados insuficientes para renderizar a composição atual.",
            }
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Pie(labels=[c["category"] for c in categories], values=[c["invested_amount"] for c in categories], hole=0.35)])
        fig.update_layout(title="Composição atual da carteira")
        return {
            "chart_id": "current_allocation",
            "title": "Composição atual da carteira",
            "type": "pie",
            "source": "selected_client_context.relevant_financial_data",
            "data": categories,
            "figure_json": fig.to_json(),
        }

    @staticmethod
    def _build_scenario_comparison_chart(scenarios: list[dict]) -> dict:
        if not scenarios:
            return {
                "chart_id": "scenario_comparison",
                "title": "Comparação entre cenários",
                "type": "fallback",
                "source": "scenarios",
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
            "type": "stacked_bar",
            "source": "scenarios[*].allocation_outline",
            "data": {"scenario_names": scenario_names, "buckets": all_buckets, "series": categories},
            "figure_json": fig.to_json(),
        }

    @staticmethod
    def _build_concentration_chart(holdings: list[dict]) -> dict:
        if not holdings:
            return {
                "chart_id": "concentration",
                "title": "Distribuição/concentração",
                "type": "fallback",
                "source": "selected_client_context.relevant_holdings",
                "message": "Sem posições suficientes para analisar concentração.",
            }
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Bar(x=[item["product_name"] for item in holdings], y=[item["invested_amount"] for item in holdings])])
        fig.update_layout(title="Principais posições por valor investido")
        return {
            "chart_id": "concentration",
            "title": "Principais posições por valor investido",
            "type": "bar",
            "source": "selected_client_context.relevant_holdings",
            "data": holdings,
            "figure_json": fig.to_json(),
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
            "type": "summary_bar",
            "source": "portfolio_diagnosis.opportunities + risk_review.alerts",
            "data": {"labels": labels, "values": values},
            "figure_json": fig.to_json(),
        }

    def run(self, *, case_state: dict) -> AgentResult:
        selected_context = case_state.get("selected_client_context", {})
        diagnosis = case_state.get("portfolio_diagnosis", {})
        scenarios = case_state.get("scenarios", [])
        charts = [
            self._build_current_allocation_chart(selected_context.get("relevant_financial_data", [])),
            self._build_scenario_comparison_chart(scenarios),
            self._build_concentration_chart(selected_context.get("relevant_holdings", [])),
            self._build_thesis_chart(diagnosis.get("opportunities", []), diagnosis.get("attention_points", [])),
        ]
        return AgentResult(payload={"charts": charts}, summary="Gráficos estruturados e prontos para a tela e para o PDF.")
