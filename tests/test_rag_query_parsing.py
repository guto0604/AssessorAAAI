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
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            embedding: Valor de entrada necessário para processar 'embedding'.
        """
        self.embedding = embedding


class _Response:
    def __init__(self, content, model):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            content: Valor de entrada necessário para processar 'content'.
            model: Modelo utilizado para executar a etapa 'model'.
        """
        self.choices = [type("_Choice", (), {"message": type("_Message", (), {"content": content})()})()]
        self.model = model
        self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class _EmbeddingsResponse:
    def __init__(self):
        """Inicializa a classe com dependências e estado necessários para o fluxo.
        """
        self.data = [_EmbeddingRow([0.1, 0.2, 0.3])]
        self.model = "text-embedding-3-small"
        self.usage = {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3}


class _Completions:
    def __init__(self):
        """Inicializa a classe com dependências e estado necessários para o fluxo.
        """
        self.calls = []

    def create(self, **kwargs):
        """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

        Args:
            kwargs: Parâmetros adicionais repassados para a chamada interna.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        self.calls.append(kwargs)
        system_prompt = kwargs["messages"][0]["content"]
        if "parser de consultas" in system_prompt:
            return _Response("consulta expandida suitability renda fixa", kwargs["model"])
        return _Response("resposta final", kwargs["model"])


class _Chat:
    def __init__(self):
        """Inicializa a classe com dependências e estado necessários para o fluxo.
        """
        self.completions = _Completions()


class _Embeddings:
    def __init__(self):
        """Inicializa a classe com dependências e estado necessários para o fluxo.
        """
        self.calls = []

    def create(self, **kwargs):
        """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

        Args:
            kwargs: Parâmetros adicionais repassados para a chamada interna.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        self.calls.append(kwargs)
        return _EmbeddingsResponse()


class _FakeClient:
    def __init__(self):
        """Inicializa a classe com dependências e estado necessários para o fluxo.
        """
        self.chat = _Chat()
        self.embeddings = _Embeddings()


class RagQueryParsingTests(unittest.TestCase):
    @patch("rag.pipeline.LocalFaissStore.load", return_value=None)
    @patch("rag.pipeline.get_openai_client")
    def test_answer_question_uses_parsed_query_for_retrieval(self, mock_client_factory, _mock_store_load):
        """Trata consultas e resultados de dados para o fluxo Talk to Data com segurança.

        Args:
            mock_client_factory: Valor de entrada necessário para processar 'mock_client_factory'.
            _mock_store_load: Valor de entrada necessário para processar '_mock_store_load'.

        Returns:
            Consulta validada ou resultado tabular da execução, conforme a etapa.
        
        """
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


if __name__ == "__main__":
    unittest.main()
