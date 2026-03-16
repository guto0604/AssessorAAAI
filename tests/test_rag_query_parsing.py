import sys
import types
import unittest
from unittest.mock import patch

if "numpy" not in sys.modules:
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.array = lambda *args, **kwargs: args[0]
    sys.modules["numpy"] = fake_numpy

from rag.pipeline import RagService


class _EmbeddingRow:
    def __init__(self, embedding):
        self.embedding = embedding


class _Response:
    def __init__(self, content, model):
        self.choices = [type("_Choice", (), {"message": type("_Message", (), {"content": content})()})()]
        self.model = model
        self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class _EmbeddingsResponse:
    def __init__(self):
        self.data = [_EmbeddingRow([0.1, 0.2, 0.3])]
        self.model = "text-embedding-3-small"
        self.usage = {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3}


class _Completions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system_prompt = kwargs["messages"][0]["content"]
        if "parser de consultas" in system_prompt:
            return _Response("consulta expandida suitability renda fixa", kwargs["model"])
        return _Response("resposta final", kwargs["model"])


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _Embeddings:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _EmbeddingsResponse()


class _FakeClient:
    def __init__(self):
        self.chat = _Chat()
        self.embeddings = _Embeddings()




class _FakeCrossEncoder:
    def predict(self, pairs):
        # força rerank invertendo prioridade para o segundo trecho
        return [0.2, 0.9]

class RagQueryParsingTests(unittest.TestCase):
    @patch("rag.pipeline.LocalFaissStore.load", return_value=None)
    @patch("rag.pipeline.get_openai_client")
    def test_answer_question_uses_parsed_query_for_retrieval(self, mock_client_factory, _mock_store_load):
        fake_client = _FakeClient()
        mock_client_factory.return_value = fake_client

        service = RagService()
        service.ensure_index_exists = lambda: None
        service.store.search = lambda embedding, k: [
            (
                type(
                    "Meta",
                    (),
                    {
                        "source_path": "knowledge_base/operacional/doc.txt",
                        "file_name": "doc.txt",
                        "chunk_id": 1,
                        "text": "texto de contexto",
                    },
                )(),
                0.9,
            )
        ]

        result = service.answer_question("qual liquidez para cliente conservador?", include_api_metrics=True)

        self.assertEqual(fake_client.embeddings.calls[0]["input"], ["consulta expandida suitability renda fixa"])
        self.assertEqual(result["api_calls"][0]["step"], "query_parser")
        self.assertEqual(result["api_calls"][1]["step"], "embedding")
        self.assertEqual(result["api_calls"][2]["step"], "chat_completion")

        parser_call = fake_client.chat.completions.calls[0]
        self.assertEqual(parser_call["temperature"], 1)

        parser_metrics = result["api_calls"][0]
        self.assertEqual(parser_metrics["model"], "gpt-5-mini")
        self.assertEqual(parser_metrics["input_tokens"], 10)
        self.assertEqual(parser_metrics["output_tokens"], 5)
        self.assertEqual(parser_metrics["total_tokens"], 15)
        self.assertTrue(parser_metrics["latency_ms"] >= 0)

    @patch("rag.pipeline.LocalFaissStore.load", return_value=None)
    @patch("rag.pipeline.get_openai_client")
    def test_answer_question_applies_cross_encoder_rerank(self, mock_client_factory, _mock_store_load):
        fake_client = _FakeClient()
        mock_client_factory.return_value = fake_client

        service = RagService()
        service.ensure_index_exists = lambda: None
        service._get_cross_encoder = lambda: _FakeCrossEncoder()

        meta_a = type(
            "Meta",
            (),
            {
                "source_path": "knowledge_base/operacional/a.txt",
                "file_name": "a.txt",
                "chunk_id": 1,
                "text": "primeiro trecho",
            },
        )()
        meta_b = type(
            "Meta",
            (),
            {
                "source_path": "knowledge_base/operacional/b.txt",
                "file_name": "b.txt",
                "chunk_id": 2,
                "text": "segundo trecho",
            },
        )()

        service.store.search = lambda embedding, k: [(meta_a, 0.95), (meta_b, 0.90)]

        result = service.answer_question(
            "pergunta",
            top_k=2,
            top_n=1,
            cross_encoder_enabled=True,
            include_api_metrics=True,
        )

        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["source_path"], "knowledge_base/operacional/b.txt")
        self.assertEqual(result["api_calls"][2]["step"], "cross_encoder_rerank")



if __name__ == "__main__":
    unittest.main()
