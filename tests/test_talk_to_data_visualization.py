import unittest
from unittest.mock import patch

import pandas as pd

from ui.pages.talk_to_data import build_llm_prompt, render_visual


class TalkToDataVisualizationTests(unittest.TestCase):
    def test_prompt_includes_color_field_instruction(self):
        prompt = build_llm_prompt("teste", "schema")
        self.assertIn("visualization.color", prompt)
        self.assertIn('"color": "campo ou vazio"', prompt)

    @patch("ui.pages.talk_to_data.st.plotly_chart")
    @patch("ui.pages.talk_to_data.st.subheader")
    @patch("ui.pages.talk_to_data.px.scatter")
    def test_scatter_uses_color_column_when_provided(self, mock_scatter, _mock_subheader, mock_plotly_chart):
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
