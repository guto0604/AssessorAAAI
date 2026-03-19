import sys
import types
import unittest
from unittest.mock import patch

if "numpy" not in sys.modules:
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.array = lambda *args, **kwargs: args[0]
    sys.modules["numpy"] = fake_numpy

from rag.pipeline import RagService
from rag.vector_store import ChunkMetadata


class _FakeClient:
    def __init__(self):
        self.chat = type("_Chat", (), {"completions": object()})()
        self.embeddings = object()


class RagMetadataBackfillTests(unittest.TestCase):
    @patch("rag.pipeline.LocalFaissStore.load", return_value=None)
    @patch("rag.pipeline.get_openai_client")
    def test_apply_default_metadata_fills_missing_segments_and_dates(
        self, mock_client_factory, _mock_store_load
    ):
        mock_client_factory.return_value = _FakeClient()

        service = RagService()
        service.ensure_index_exists = lambda: None
        saved = {"called": False}
        service.store.save = lambda: saved.__setitem__("called", True)
        service.store.metadata = [
            ChunkMetadata(
                source_path="knowledge_base/operacional/doc.txt",
                source_hash="hash1",
                kb_folder="operacional",
                file_name="doc.txt",
                chunk_id=0,
                text="conteudo",
                allowed_segments=[],
                document_date=None,
                indexed_at=None,
            )
        ]
        service.store.document_registry = {}

        result = service.apply_default_metadata_to_all_missing(
            default_segments=["Até 300k", "300k-2M", "2M+"],
            default_date="2026-03-19",
        )

        updated_chunk = service.store.metadata[0]
        document = service.store.document_registry["knowledge_base/operacional/doc.txt"]

        self.assertEqual(updated_chunk.allowed_segments, ["Até 300k", "300k-2M", "2M+"])
        self.assertEqual(updated_chunk.document_date, "2026-03-19")
        self.assertIsNotNone(updated_chunk.indexed_at)
        self.assertEqual(document.allowed_segments, ["Até 300k", "300k-2M", "2M+"])
        self.assertEqual(document.document_date, "2026-03-19")
        self.assertEqual(result.updated_documents, 1)
        self.assertGreaterEqual(result.updated_chunks, 3)
        self.assertTrue(saved["called"])


if __name__ == "__main__":
    unittest.main()
