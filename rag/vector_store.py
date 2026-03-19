import json
from dataclasses import asdict, dataclass
from datetime import date

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

import numpy as np

from rag.config import (
    FAISS_HNSW_EF_CONSTRUCTION,
    FAISS_HNSW_EF_SEARCH,
    FAISS_HNSW_M,
    VECTORSTORE_DIR,
    VECTORSTORE_DOCUMENTS_PATH,
    VECTORSTORE_INDEX_PATH,
    VECTORSTORE_MANIFEST_PATH,
    VECTORSTORE_METADATA_PATH,
)


def _normalize_source_path(value: str) -> str:
    return (value or "").replace("\\", "/")


@dataclass
class ChunkMetadata:
    source_path: str
    source_hash: str
    kb_folder: str
    file_name: str
    chunk_id: int
    text: str
    allowed_segments: list[str] | None = None
    document_date: str | None = None
    indexed_at: str | None = None


@dataclass
class DocumentMetadata:
    source_path: str
    source_hash: str
    kb_folder: str
    file_name: str
    allowed_segments: list[str]
    document_date: str
    indexed_at: str


class LocalFaissStore:
    def __init__(self):
        """Inicializa a classe com dependências e estado necessários para o fluxo.
        """
        self.index = None
        self.metadata: list[ChunkMetadata] = []
        self.manifest: dict[str, str] = {}
        self.document_registry: dict[str, DocumentMetadata] = {}

    @staticmethod
    def _ensure_faiss_available():
        """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        if faiss is None:
            raise RuntimeError("Dependência 'faiss-cpu' não instalada. Instale para habilitar a base vetorial.")

    def _ensure_dir(self):
        """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    def _build_hnsw_index(self, dimension: int):
        """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

        Args:
            dimension: Valor de entrada necessário para processar 'dimension'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        self._ensure_faiss_available()
        index = faiss.IndexHNSWFlat(dimension, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = FAISS_HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = FAISS_HNSW_EF_SEARCH
        return index

    def exists(self) -> bool:
        """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        return VECTORSTORE_INDEX_PATH.exists() and VECTORSTORE_METADATA_PATH.exists()

    def load(self):
        """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

        Returns:
            Dados carregados e prontos para consumo no fluxo da aplicação.
        """
        if not self.exists():
            self.index = None
            self.metadata = []
            self.manifest = {}
            self.document_registry = {}
            return

        self._ensure_faiss_available()
        self.index = faiss.read_index(str(VECTORSTORE_INDEX_PATH))
        raw_metadata = json.loads(VECTORSTORE_METADATA_PATH.read_text(encoding="utf-8"))
        self.metadata = [
            ChunkMetadata(
                **{
                    **item,
                    "source_path": _normalize_source_path(item.get("source_path", "")),
                    "allowed_segments": list(item.get("allowed_segments") or []),
                }
            )
            for item in raw_metadata
        ]
        if VECTORSTORE_MANIFEST_PATH.exists():
            loaded_manifest = json.loads(VECTORSTORE_MANIFEST_PATH.read_text(encoding="utf-8"))
            self.manifest = {
                _normalize_source_path(source_path): source_hash
                for source_path, source_hash in loaded_manifest.items()
            }
        else:
            self.manifest = {}

        if VECTORSTORE_DOCUMENTS_PATH.exists():
            loaded_documents = json.loads(VECTORSTORE_DOCUMENTS_PATH.read_text(encoding="utf-8"))
            self.document_registry = {
                _normalize_source_path(source_path): DocumentMetadata(
                    **{
                        **item,
                        "source_path": _normalize_source_path(item.get("source_path", source_path)),
                        "allowed_segments": list(item.get("allowed_segments") or []),
                    }
                )
                for source_path, item in loaded_documents.items()
            }
        else:
            self.document_registry = {}
            for item in self.metadata:
                source_path = _normalize_source_path(item.source_path)
                if source_path in self.document_registry:
                    continue
                if item.allowed_segments and item.document_date and item.indexed_at:
                    self.document_registry[source_path] = DocumentMetadata(
                        source_path=source_path,
                        source_hash=item.source_hash,
                        kb_folder=item.kb_folder,
                        file_name=item.file_name,
                        allowed_segments=list(item.allowed_segments),
                        document_date=item.document_date,
                        indexed_at=item.indexed_at,
                    )

    def save(self):
        """Salva o resultado processado para persistir histórico e permitir consulta posterior.

        Returns:
            Referência do recurso salvo para uso posterior.
        """
        self._ensure_dir()
        if self.index is None:
            return
        self._ensure_faiss_available()
        faiss.write_index(self.index, str(VECTORSTORE_INDEX_PATH))
        VECTORSTORE_METADATA_PATH.write_text(
            json.dumps([asdict(m) for m in self.metadata], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        VECTORSTORE_MANIFEST_PATH.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        VECTORSTORE_DOCUMENTS_PATH.write_text(
            json.dumps(
                {source_path: asdict(metadata) for source_path, metadata in self.document_registry.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def clear(self):
        """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        self.index = None
        self.metadata = []
        self.manifest = {}
        self.document_registry = {}
        for path in [
            VECTORSTORE_INDEX_PATH,
            VECTORSTORE_METADATA_PATH,
            VECTORSTORE_MANIFEST_PATH,
            VECTORSTORE_DOCUMENTS_PATH,
        ]:
            if path.exists():
                path.unlink()

    def add_embeddings(self, embeddings: list[list[float]], metadata: list[ChunkMetadata]):
        """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

        Args:
            embeddings: Valor de entrada necessário para processar 'embeddings'.
            metadata: Valor de entrada necessário para processar 'metadata'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        if not embeddings:
            return

        self._ensure_faiss_available()
        matrix = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(matrix)

        if self.index is None:
            self.index = self._build_hnsw_index(matrix.shape[1])
        elif hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = FAISS_HNSW_EF_SEARCH

        self.index.add(matrix)
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: list[float],
        k: int = 4,
        filter_fn=None,
        search_k: int | None = None,
    ) -> list[tuple[ChunkMetadata, float]]:
        """Realiza a busca de informações com os filtros definidos para o contexto atual.

        Args:
            query_embedding: Valor de entrada necessário para processar 'query_embedding'.
            k: Valor de entrada necessário para processar 'k'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        """
        if self.index is None or not self.metadata:
            return []

        self._ensure_faiss_available()
        vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(vector)
        requested_k = search_k if search_k is not None else k
        scores, indices = self.index.search(vector, min(requested_k, len(self.metadata)))

        results: list[tuple[ChunkMetadata, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            metadata = self.metadata[int(idx)]
            if filter_fn and not filter_fn(metadata):
                continue
            results.append((metadata, float(score)))
            if len(results) >= k:
                break
        return results

    @staticmethod
    def today_iso() -> str:
        return date.today().isoformat()
