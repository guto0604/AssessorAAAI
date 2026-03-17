import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st

from ui.pages.talk_to_data import _build_rls_filter_sql, build_llm_prompt, render_visual


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
        prompt = build_llm_prompt("teste", "schema", allowed_cliente_ids=["A_001", "A_002"])
        self.assertIn("Aplique RLS SEMPRE", prompt)
        self.assertIn("A_001, A_002", prompt)

    @patch("ui.pages.talk_to_data.pd.read_parquet")
    def test_build_rls_filter_sql_escapes_ids(self, mock_read_parquet):
        mock_read_parquet.return_value = pd.DataFrame(
            {
                "Cliente_ID": ["A_001", "X_'42"],
                "Patrimonio_Investido_Conosco": [100_000, 3_000_000],
            }
        )
        st.session_state["rls_allowed_segments"] = ["Até 300k", "2M+"]

        rls_sql = _build_rls_filter_sql()

        self.assertEqual(rls_sql, "'A_001', 'X_''42'")

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
        mock_bar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
