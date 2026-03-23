from __future__ import annotations

from core.investment_case_builder.agents_base import AgentResult, BaseInvestmentCaseAgent
from core.investment_case_builder.config import CASE_OUTPUT_DIR
from core.investment_case_builder.llm_support import try_json_completion
from core.investment_case_builder.prompts import build_pdf_prompts


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

        drawing.add(String(10, 100, chart.get("message", chart.get("fallback_message", "Sem dados suficientes para este gráfico.")), fontSize=10))
        return drawing

    def _build_fallback_blueprint(self, *, case_state: dict) -> dict:
        risk_review = case_state.get("risk_review", {})
        return {
            "document_title": "Investment Case Builder",
            "cover": {
                "subtitle": "Proposta consultiva estruturada a partir do workflow multiagente",
                "client_name": case_state.get("client_name", case_state.get("client_id")),
                "case_id": case_state.get("case_id"),
            },
            "sections": [
                {"section_id": "executive_summary", "title": "Resumo executivo", "content_blocks": [{"block_type": "paragraph", "content": {"source": "proposal.executive_summary"}}]},
                {"section_id": "case_context", "title": "Contexto do caso", "content_blocks": [{"block_type": "paragraph", "content": {"source": "proposal.client_friendly_readout"}}, {"block_type": "paragraph", "content": {"source": "proposal.internal_readout"}}]},
                {"section_id": "diagnosis", "title": "Diagnóstico", "content_blocks": [{"block_type": "bullets", "content": {"source": "portfolio_diagnosis.key_findings"}}]},
                {"section_id": "scenarios", "title": "Cenários", "content_blocks": [{"block_type": "table", "content": {"source": "scenarios"}}]},
                {"section_id": "risks", "title": "Riscos e alertas", "content_blocks": [{"block_type": "bullets", "content": {"source": "risk_review.alerts"}}]},
                {"section_id": "next_steps", "title": "Próximos passos", "content_blocks": [{"block_type": "bullets", "content": {"source": "proposal.next_steps"}}]},
                {"section_id": "charts", "title": "Gráficos", "content_blocks": [{"block_type": "chart_reference", "content": {"source": "visualizations"}}]},
            ],
            "appendix": [],
            "final_disclaimer": risk_review.get("suggested_disclaimers", []),
        }

    def _resolve_block_content(self, *, block: dict, case_state: dict):
        source = (block.get("content") or {}).get("source")
        if source == "proposal.executive_summary":
            return [case_state.get("proposal", {}).get("executive_summary", "Resumo não disponível.")]
        if source == "proposal.client_friendly_readout":
            return [case_state.get("proposal", {}).get("client_friendly_readout", "")]
        if source == "proposal.internal_readout":
            return [case_state.get("proposal", {}).get("internal_readout", "")]
        if source == "portfolio_diagnosis.key_findings":
            return case_state.get("portfolio_diagnosis", {}).get("key_findings", [])
        if source == "risk_review.alerts":
            return [f"[{item.get('severity', 'info').upper()}] {item.get('scenario', '')}: {item.get('message', '')}" for item in case_state.get("risk_review", {}).get("alerts", [])]
        if source == "proposal.next_steps":
            return case_state.get("proposal", {}).get("next_steps", [])
        if source == "scenarios":
            return case_state.get("scenarios", [])
        if source == "visualizations":
            return case_state.get("visualizations", [])
        return []

    def run(self, *, case_state: dict) -> AgentResult:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Spacer

        fallback_blueprint = self._build_fallback_blueprint(case_state=case_state)
        system_prompt, user_prompt = build_pdf_prompts(case_state=case_state, heuristic_baseline=fallback_blueprint)
        llm_payload = try_json_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=self.temperature,
        )
        blueprint = llm_payload if isinstance(llm_payload, dict) and llm_payload.get("sections") else fallback_blueprint

        CASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CASE_OUTPUT_DIR / f"{case_state['case_id']}.pdf"
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], textColor=colors.HexColor("#0B1F33"), spaceAfter=8))
        styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=10, leading=14))

        story = []
        story.append(self._paragraph(blueprint.get("document_title", "Investment Case Builder"), "Title", styles))
        cover = blueprint.get("cover", {})
        story.append(self._paragraph(f"Cliente: {cover.get('client_name', case_state.get('client_name', case_state.get('client_id')))}", "Heading3", styles))
        story.append(self._paragraph(f"Objetivo do assessor: {case_state.get('advisor_prompt', '')}", "BodySmall", styles))
        if cover.get("subtitle"):
            story.append(self._paragraph(cover["subtitle"], "BodySmall", styles))
        story.append(Spacer(1, 0.4 * cm))

        for section in blueprint.get("sections", []):
            story.append(self._paragraph(section.get("title", "Seção"), "SectionTitle", styles))
            for block in section.get("content_blocks", []):
                block_type = block.get("block_type")
                resolved = self._resolve_block_content(block=block, case_state=case_state)
                if block_type == "paragraph":
                    for item in resolved[:2]:
                        if item:
                            story.append(self._paragraph(str(item), "BodySmall", styles))
                elif block_type == "bullets":
                    for item in resolved[:6]:
                        story.append(self._paragraph(f"• {item}", "BodySmall", styles))
                elif block_type == "table":
                    for scenario in resolved[:3]:
                        story.append(self._paragraph(f"<b>{scenario.get('name')}</b>", "BodySmall", styles))
                        story.append(self._paragraph(scenario.get("rationale", ""), "BodySmall", styles))
                        story.append(self._paragraph(f"Liquidez aproximada: {scenario.get('approx_liquidity', '-')}", "BodySmall", styles))
                elif block_type == "chart_reference":
                    for chart in resolved:
                        story.append(self._render_chart(chart))
                elif block_type == "disclaimer":
                    for item in resolved:
                        story.append(self._paragraph(f"• {item}", "BodySmall", styles))
                story.append(Spacer(1, 0.2 * cm))

        story.append(self._paragraph("Disclaimer", "SectionTitle", styles))
        for disclaimer in blueprint.get("final_disclaimer", []):
            story.append(self._paragraph(f"• {disclaimer}", "BodySmall", styles))

        doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
        doc.build(story)

        return AgentResult(
            payload={"pdf_path": str(output_path), "pdf_blueprint": blueprint},
            summary=f"PDF executivo gerado em {output_path.name}.",
        )
