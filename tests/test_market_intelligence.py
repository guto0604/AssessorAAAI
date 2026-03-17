import unittest
from unittest.mock import patch

from core.market_intelligence import _dedupe, _normalize_result, fetch_market_intelligence


class _Msg:
    def __init__(self, content):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            content: Valor de entrada necessário para processar 'content'.
        """
        self.content = content


class _Choice:
    def __init__(self, content):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            content: Valor de entrada necessário para processar 'content'.
        """
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            content: Valor de entrada necessário para processar 'content'.
        """
        self.choices = [_Choice(content)]


class _Completions:
    def create(self, **kwargs):
        """Responsável por executar uma etapa do fluxo da aplicação de assessoria.

        Args:
            kwargs: Parâmetros adicionais repassados para a chamada interna.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        system = kwargs["messages"][0]["content"]
        if "Classifique cada notícia" in system:
            return _Resp('{"items":[{"idx":0,"resumo_pt":"Resumo 1","tipo_evento":"macro","tema_relacionado":"Brasil","score_relevancia":85}]}')
        return _Resp("Resumo consolidado")


class _Chat:
    completions = _Completions()


class _FakeClient:
    chat = _Chat()


class MarketIntelligenceTests(unittest.TestCase):
    def test_dedupe_by_url_and_title(self):
        """Valida o comportamento esperado deste fluxo por meio de um teste automatizado.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        items = [
            {"url": "https://a", "title": "A"},
            {"url": "https://a", "title": "A"},
            {"url": "https://b", "title": "A"},
        ]
        deduped = _dedupe(items)
        self.assertEqual(len(deduped), 2)

    def test_normalize_result(self):
        """Valida o comportamento esperado deste fluxo por meio de um teste automatizado.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        source = {"title": "Notícia", "url": "https://x", "highlights": ["h1"], "publishedDate": "2026-01-01T00:00:00Z"}
        normalized = _normalize_result(source, source_type="radar", sector="Bancos", company="Itaú")
        self.assertEqual(normalized["title"], "Notícia")
        self.assertEqual(normalized["sector"], "Bancos")

    @patch("core.market_intelligence.get_openai_client", return_value=_FakeClient())
    @patch("core.market_intelligence.search_tavily", return_value=[{"title": "T", "url": "https://u", "published_date": "2026-01-01T00:00:00Z", "content": "x"}])
    def test_fetch_market_intelligence_shape(self, _search_mock, _openai_mock):
        """Processa dados de mercado para gerar contexto acionável ao assessor.

        Args:
            _search_mock: Valor de entrada necessário para processar '_search_mock'.
            _openai_mock: Valor de entrada necessário para processar '_openai_mock'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        out = fetch_market_intelligence(days=1)
        self.assertIn("ranked_news", out)
        self.assertIn("sectors", out)
        self.assertTrue(len(out["sectors"]) > 0)


    @patch("core.market_intelligence.get_openai_client", return_value=_FakeClient())
    @patch("core.market_intelligence.search_tavily", return_value=[{"title": "T", "url": "https://u", "published_date": "2026-01-01T00:00:00Z", "content": "x"}])
    def test_fetch_market_intelligence_sector_filter(self, _search_mock, _openai_mock):
        """Processa dados de mercado para gerar contexto acionável ao assessor.

        Args:
            _search_mock: Valor de entrada necessário para processar '_search_mock'.
            _openai_mock: Valor de entrada necessário para processar '_openai_mock'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        out = fetch_market_intelligence(days=1, sector="Bancos")
        self.assertEqual(len(out["sectors"]), 1)
        self.assertEqual(out["sectors"][0]["sector"], "Bancos")


if __name__ == "__main__":
    unittest.main()
