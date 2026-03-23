import unittest
from unittest.mock import patch

from core.investment_case_builder.data_relevance_agent import DataRelevanceAgent
from core.investment_case_builder.orchestrator import InvestmentCaseOrchestrator


MASTER_CONTEXT = {
    "client_info": {
        "Cliente_ID": "C1",
        "Nome": "Cliente Teste",
        "Patrimonio_Investido_Conosco": 1000000,
        "Patrimonio_Investido_Outros": 250000,
        "Dinheiro_Disponivel_Para_Investir": 50000,
        "Perfil_Suitability": "Conservador",
        "Rentabilidade_12_meses": 0.09,
        "CDI_12_Meses": 0.11,
    },
    "portfolio": {
        "total_invested": 800000,
        "holdings": [
            {"product_name": "CDB Liquidez", "category": "Renda Fixa", "invested_amount": 300000, "liquidity_hint": "alta"},
            {"product_name": "Fundo Crédito", "category": "Fundos", "invested_amount": 250000, "liquidity_hint": "média"},
            {"product_name": "Ações Brasil", "category": "Renda Variável", "invested_amount": 250000, "liquidity_hint": "média"},
        ],
        "allocation_by_category": [
            {"category": "Renda Fixa", "invested_amount": 300000, "allocation_pct": 37.5, "liquidity_hint": "alta"},
            {"category": "Fundos", "invested_amount": 250000, "allocation_pct": 31.25, "liquidity_hint": "média"},
            {"category": "Renda Variável", "invested_amount": 250000, "allocation_pct": 31.25, "liquidity_hint": "média"},
        ],
        "holdings_count": 3,
    },
    "product_reference": [],
    "sources": {},
}


class DataRelevanceAgentTests(unittest.TestCase):
    def test_prompt_override_is_registered_when_profile_conflicts(self):
        agent = DataRelevanceAgent(model="gpt-5-mini", temperature=1)
        result = agent.run(
            master_context=MASTER_CONTEXT,
            advisor_prompt="Monte um caso moderado com liquidez para alocar R$ 120000.",
            additional_notes="",
            tone_focus="Consultivo",
        )

        self.assertTrue(result.payload["conflicts_detected"])
        self.assertEqual(result.payload["priority_decisions"][0]["field"], "Perfil_Suitability")
        self.assertEqual(result.payload["priority_decisions"][0]["decision"], "prompt_override_for_case_hypothesis")


class OrchestratorTests(unittest.TestCase):
    @patch("core.investment_case_builder.orchestrator.load_client_master_context", return_value=MASTER_CONTEXT)
    @patch("core.investment_case_builder.visualization_agent.VisualizationAgent.run", return_value=type("R", (), {"payload": {"charts": [{"chart_id": "c1", "type": "fallback", "title": "t", "message": "m"}]}, "summary": "ok"})())
    @patch("core.investment_case_builder.pdf_builder_agent.PDFBuilderAgent.run", return_value=type("R", (), {"payload": {"pdf_path": "/tmp/fake_case.pdf"}, "summary": "ok"})())
    def test_full_workflow_populates_case_state(self, _mock_pdf, _mock_visualization, _mock_loader):
        orchestrator = InvestmentCaseOrchestrator()
        case_state = orchestrator.initialize_case(
            client_id="C1",
            client_name="Cliente Teste",
            advisor_prompt="Criar uma tese de diversificação com foco em liquidez.",
            additional_notes="Considerar reunião na próxima semana.",
            tone_focus="Executivo",
        )

        case_state = orchestrator.run_full_workflow(case_state)

        self.assertEqual(case_state["workflow_status"]["data_relevance"]["status"], "completed")
        self.assertEqual(case_state["workflow_status"]["pdf_builder"]["status"], "completed")
        self.assertTrue(case_state["selected_client_context"])
        self.assertTrue(case_state["workflow_plan"])
        self.assertTrue(case_state["portfolio_diagnosis"])
        self.assertEqual(len(case_state["scenarios"]), 3)
        self.assertTrue(case_state["risk_review"])
        self.assertTrue(case_state["proposal"])
        self.assertTrue(case_state["visualizations"])
        self.assertEqual(case_state["pdf_path"], "/tmp/fake_case.pdf")


if __name__ == "__main__":
    unittest.main()
