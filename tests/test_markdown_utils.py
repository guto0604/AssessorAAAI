import unittest

from ui.markdown_utils import escape_streamlit_markdown


class EscapeStreamlitMarkdownTests(unittest.TestCase):
    def test_escapes_unescaped_dollar_signs(self):
        text = 'Valores próximos: $10 e $20 na mesma frase.'

        self.assertEqual(
            escape_streamlit_markdown(text),
            'Valores próximos: \\$10 e \\$20 na mesma frase.',
        )

    def test_preserves_previously_escaped_dollar_signs(self):
        text = 'Valor já escapado: \\$30 e outro $40.'

        self.assertEqual(
            escape_streamlit_markdown(text),
            'Valor já escapado: \\$30 e outro \\$40.',
        )

    def test_handles_none(self):
        self.assertEqual(escape_streamlit_markdown(None), '')
