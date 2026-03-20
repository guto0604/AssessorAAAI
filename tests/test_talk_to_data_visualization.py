import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st

from ui.pages.talk_to_data import _build_patrimonio_rls_condition_sql, build_llm_prompt, render_visual


class TalkToDataVisualizationTests(unittest.TestCase):
    def tearDown(self):
        st.session_state.pop("rls_allowed_segments", None)

    def test_prompt_includes_color_field_instruction(self):
        """Valida o comportamento esperado deste fluxo por meio de um teste automatizado.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        prompt = build_llm_prompt("teste", "schema")
        self.assertIn("visualization.color", prompt)
        self.assertIn('"color": "campo ou vazio"', prompt)

    def test_prompt_includes_rls_instruction(self):
        prompt = build_llm_prompt("teste", "schema")
        self.assertIn("RLS por patrimônio já é aplicado de forma oculta", prompt)
        self.assertIn("prefira usar Cliente_ID como eixo/categoria identificadora", prompt)
        self.assertIn("inclua Nome apenas para tooltip/hover", prompt)
        self.assertNotIn("Cliente_ID na lista permitida", prompt)

    def test_build_patrimonio_rls_condition_sql_by_selected_segments(self):
        st.session_state["rls_allowed_segments"] = ["Até 300k", "2M+"]

        rls_sql = _build_patrimonio_rls_condition_sql()

        self.assertIn("Patrimonio_Investido_Conosco <= 300000", rls_sql)
        self.assertIn("Patrimonio_Investido_Conosco > 2000000", rls_sql)
        self.assertNotIn("Patrimonio_Investido_Conosco > 300000 AND Patrimonio_Investido_Conosco <= 2000000", rls_sql)

    def test_build_patrimonio_rls_condition_sql_without_segments(self):
        st.session_state["rls_allowed_segments"] = []

        rls_sql = _build_patrimonio_rls_condition_sql()

        self.assertEqual(rls_sql, "FALSE")

    @patch("ui.pages.talk_to_data.st.plotly_chart")
    @patch("ui.pages.talk_to_data.st.subheader")
    @patch("ui.pages.talk_to_data.px.scatter")
    def test_scatter_uses_color_column_when_provided(self, mock_scatter, _mock_subheader, mock_plotly_chart):
        """Valida o comportamento esperado deste fluxo por meio de um teste automatizado.

        Args:
            mock_scatter: Valor de entrada necessário para processar 'mock_scatter'.
            _mock_subheader: Valor de entrada necessário para processar '_mock_subheader'.
            mock_plotly_chart: Valor de entrada necessário para processar 'mock_plotly_chart'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        mock_scatter.return_value = object()
        df = pd.DataFrame(
            {
                "salario": [1000, 2000],
                "investido": [5000, 7000],
                "perfil": ["Conservador", "Arrojado"],
            }
        )
        spec = {
            "needed": True,
            "type": "scatter",
            "x": "salario",
            "y": "investido",
            "color": "perfil",
            "title": "Teste",
        }

        render_visual(df, spec)

        mock_scatter.assert_called_once_with(
            df,
            x="salario",
            y="investido",
            color="perfil",
            title="Teste",
            hover_data=[],
        )
        mock_plotly_chart.assert_called_once()

    @patch("ui.pages.talk_to_data.st.info")
    @patch("ui.pages.talk_to_data.px.bar")
    def test_shows_info_when_color_column_missing(self, mock_bar, mock_info):
        """Valida o comportamento esperado deste fluxo por meio de um teste automatizado.

        Args:
            mock_bar: Valor de entrada necessário para processar 'mock_bar'.
            mock_info: Valor de entrada necessário para processar 'mock_info'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        df = pd.DataFrame({"categoria": ["A"], "total": [1]})
        spec = {
            "needed": True,
            "type": "bar",
            "x": "categoria",
            "y": "total",
            "color": "segmento",
            "title": "Teste",
        }

        render_visual(df, spec)

        mock_info.assert_called_once()
        mock_bar.assert_called_once()

    @patch("ui.pages.talk_to_data.st.plotly_chart")
    @patch("ui.pages.talk_to_data.st.subheader")
    @patch("ui.pages.talk_to_data.px.line")
    def test_render_without_color_when_llm_returns_empty_color(self, mock_line, _mock_subheader, mock_plotly_chart):
        mock_line.return_value = object()
        df = pd.DataFrame({"mes": ["jan"], "valor": [10]})
        spec = {
            "needed": True,
            "type": "line",
            "x": "mes",
            "y": "valor",
            "color": "",
            "title": "Sem cor",
        }

        render_visual(df, spec)

        mock_line.assert_called_once_with(df, x="mes", y="valor", color=None, title="Sem cor", hover_data=[])
        mock_plotly_chart.assert_called_once()

    @patch("ui.pages.talk_to_data.st.plotly_chart")
    @patch("ui.pages.talk_to_data.st.subheader")
    @patch("ui.pages.talk_to_data.px.bar")
    def test_bar_uses_cliente_id_on_axis_and_nome_only_in_tooltip(self, mock_bar, _mock_subheader, mock_plotly_chart):
        mock_bar.return_value = object()
        df = pd.DataFrame(
            {
                "Cliente_ID": ["A_001", "A_002"],
                "Nome": ["Alex Silva", "Alex Silva"],
                "total": [10, 20],
            }
        )
        spec = {
            "needed": True,
            "type": "bar",
            "x": "Nome",
            "y": "total",
            "color": None,
            "title": "Clientes",
        }

        render_visual(df, spec)

        mock_bar.assert_called_once_with(
            df,
            x="Cliente_ID",
            y="total",
            color=None,
            title="Clientes",
            hover_data=["Nome"],
        )
        mock_plotly_chart.assert_called_once()



if __name__ == "__main__":
    unittest.main()
