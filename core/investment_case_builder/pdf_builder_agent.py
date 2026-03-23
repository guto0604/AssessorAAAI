from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent
from core.investment_case_builder.config import CASE_OUTPUT_DIR


class PDFBuilderAgent(BaseInvestmentCaseAgent):
    agent_name = "pdf_builder"
    instruction = (
        "Montar um PDF executivo a partir do estado consolidado, com texto, riscos, cenários e gráficos derivados do próprio case_state."
    )

    def _paragraph(self, text: str, style_name: str, styles):
        from reportlab.platypus import Paragraph

        return Paragraph(text.replace("\n", "<br/>"), styles[style_name])

    def _render_chart(self, chart: dict):
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.graphics.shapes import Drawing, String
        from reportlab.lib import colors

        drawing = Drawing(460, 220)
        drawing.add(String(10, 200, chart.get("title", "Gráfico"), fontSize=12, fillColor=colors.HexColor("#0B1F33")))

        if chart.get("type") == "pie":
            pie = Pie()
            pie.x = 120
            pie.y = 20
            pie.width = 180
            pie.height = 160
            pie.data = [max(0, item.get("invested_amount", 0)) for item in chart.get("data", [])]
            pie.labels = [item.get("category", "-") for item in chart.get("data", [])]
            pie.slices.strokeWidth = 0.5
            drawing.add(pie)
            return drawing

        if chart.get("type") == "summary_bar":
            data = chart.get("data", {})
            bar = VerticalBarChart()
            bar.x = 60
            bar.y = 30
            bar.height = 130
            bar.width = 220
            bar.data = [tuple(data.get("values", [0, 0]))]
            bar.categoryAxis.categoryNames = data.get("labels", ["A", "B"])
            bar.bars[0].fillColor = colors.HexColor("#2E86AB")
            drawing.add(bar)
            return drawing

        if chart.get("type") in {"bar", "stacked_bar"}:
            data = chart.get("data", {})
            bar = VerticalBarChart()
            bar.x = 50
            bar.y = 30
            bar.height = 130
            bar.width = 320
            bar.strokeColor = colors.black
            if chart.get("type") == "stacked_bar":
                series = data.get("series", [])
                buckets = data.get("buckets", [])
                bar.data = tuple(tuple(outline.get(bucket, 0) for outline in series) for bucket in buckets) or ((0,),)
                bar.categoryAxis.categoryNames = data.get("scenario_names", [])
                for idx in range(len(bar.data)):
                    bar.bars[idx].fillColor = [colors.HexColor("#2E86AB"), colors.HexColor("#9C27B0"), colors.HexColor("#F18F01"), colors.HexColor("#28A745")][idx % 4]
            else:
                rows = chart.get("data", [])
                bar.data = [tuple(item.get("invested_amount", 0) for item in rows)] or [(0,)]
                bar.categoryAxis.categoryNames = [item.get("product_name", "-")[:12] for item in rows] or ["-"]
                bar.bars[0].fillColor = colors.HexColor("#2E86AB")
            drawing.add(bar)
            return drawing

        drawing.add(String(10, 100, chart.get("message", "Sem dados suficientes para este gráfico."), fontSize=10))
        return drawing

    def run(self, *, case_state: dict) -> AgentResult:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Spacer

        CASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CASE_OUTPUT_DIR / f"{case_state['case_id']}.pdf"
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], textColor=colors.HexColor("#0B1F33"), spaceAfter=8))
        styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=10, leading=14))

        story = []
        story.append(self._paragraph("Investment Case Builder", "Title", styles))
        story.append(self._paragraph(f"Cliente: {case_state.get('client_name', case_state.get('client_id'))}", "Heading3", styles))
        story.append(self._paragraph(f"Objetivo do assessor: {case_state.get('advisor_prompt', '')}", "BodySmall", styles))
        story.append(Spacer(1, 0.4 * cm))

        proposal = case_state.get("proposal", {})
        diagnosis = case_state.get("portfolio_diagnosis", {})
        risk_review = case_state.get("risk_review", {})

        story.append(self._paragraph("Resumo executivo", "SectionTitle", styles))
        story.append(self._paragraph(proposal.get("executive_summary", "Resumo não disponível."), "BodySmall", styles))
        story.append(Spacer(1, 0.3 * cm))

        story.append(self._paragraph("Contexto do caso", "SectionTitle", styles))
        story.append(self._paragraph(proposal.get("client_friendly_readout", ""), "BodySmall", styles))
        story.append(self._paragraph(proposal.get("internal_readout", ""), "BodySmall", styles))
        story.append(Spacer(1, 0.3 * cm))

        story.append(self._paragraph("Diagnóstico", "SectionTitle", styles))
        for item in diagnosis.get("key_findings", [])[:5]:
            story.append(self._paragraph(f"• {item}", "BodySmall", styles))
        story.append(Spacer(1, 0.3 * cm))

        story.append(self._paragraph("Cenários", "SectionTitle", styles))
        for scenario in case_state.get("scenarios", []):
            story.append(self._paragraph(f"<b>{scenario.get('name')}</b>", "BodySmall", styles))
            story.append(self._paragraph(scenario.get("rationale", ""), "BodySmall", styles))
            story.append(self._paragraph(f"Liquidez aproximada: {scenario.get('approx_liquidity', '-')}", "BodySmall", styles))
            story.append(Spacer(1, 0.2 * cm))

        story.append(self._paragraph("Riscos e alertas", "SectionTitle", styles))
        for alert in risk_review.get("alerts", [])[:6]:
            story.append(self._paragraph(f"• [{alert.get('severity', 'info').upper()}] {alert.get('scenario', '')}: {alert.get('message', '')}", "BodySmall", styles))
        story.append(Spacer(1, 0.3 * cm))

        story.append(self._paragraph("Próximos passos", "SectionTitle", styles))
        for item in proposal.get("next_steps", []):
            story.append(self._paragraph(f"• {item}", "BodySmall", styles))
        story.append(Spacer(1, 0.3 * cm))

        story.append(self._paragraph("Gráficos", "SectionTitle", styles))
        for chart in case_state.get("visualizations", []):
            story.append(self._render_chart(chart))
            story.append(Spacer(1, 0.2 * cm))

        story.append(self._paragraph("Disclaimer", "SectionTitle", styles))
        for disclaimer in risk_review.get("suggested_disclaimers", []):
            story.append(self._paragraph(f"• {disclaimer}", "BodySmall", styles))

        doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
        doc.build(story)

        return AgentResult(
            payload={"pdf_path": str(output_path)},
            summary=f"PDF executivo gerado em {output_path.name}.",
        )
