import unittest
from unittest.mock import patch

from core.market_intelligence import _dedupe, _normalize_result, fetch_market_intelligence


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def create(self, **kwargs):
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
        items = [
            {"url": "https://a", "title": "A"},
            {"url": "https://a", "title": "A"},
            {"url": "https://b", "title": "A"},
        ]
        deduped = _dedupe(items)
        self.assertEqual(len(deduped), 2)

    def test_normalize_result(self):
        source = {"title": "Notícia", "url": "https://x", "highlights": ["h1"], "publishedDate": "2026-01-01T00:00:00Z"}
        normalized = _normalize_result(source, source_type="radar", sector="Bancos", company="Itaú")
        self.assertEqual(normalized["title"], "Notícia")
        self.assertEqual(normalized["sector"], "Bancos")

    @patch("core.market_intelligence.get_openai_client", return_value=_FakeClient())
    @patch("core.market_intelligence.search_exa", return_value=[{"title": "T", "url": "https://u", "publishedDate": "2026-01-01T00:00:00Z", "highlights": ["x"]}])
    def test_fetch_market_intelligence_shape(self, _search_mock, _openai_mock):
        out = fetch_market_intelligence(days=1)
        self.assertIn("radar_events", out)
        self.assertIn("sectors", out)
        self.assertTrue(len(out["sectors"]) > 0)


if __name__ == "__main__":
    unittest.main()
