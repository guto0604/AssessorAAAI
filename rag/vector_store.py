import json
from dataclasses import asdict, dataclass

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
    VECTORSTORE_INDEX_PATH,
    VECTORSTORE_MANIFEST_PATH,
    VECTORSTORE_METADATA_PATH,
)


@dataclass
class ChunkMetadata:
    source_path: str
    source_hash: str
    kb_folder: str
    file_name: str
    chunk_id: int
    text: str


class LocalFaissStore:
    def __init__(self):
        """  init  .
        """
        self.index = None
        self.metadata: list[ChunkMetadata] = []
        self.manifest: dict[str, str] = {}

    @staticmethod
    def _ensure_faiss_available():
        """ ensure faiss available.

        Returns:
            Valor de retorno da função.
        """
        if faiss is None:
            raise RuntimeError("Dependência 'faiss-cpu' não instalada. Instale para habilitar a base vetorial.")

    def _ensure_dir(self):
        """ ensure dir.

        Returns:
            Valor de retorno da função.
        """
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    def _build_hnsw_index(self, dimension: int):
        """ build hnsw index.

        Args:
            dimension: Descrição do parâmetro `dimension`.

        Returns:
            Valor de retorno da função.
        """
        self._ensure_faiss_available()
        index = faiss.IndexHNSWFlat(dimension, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = FAISS_HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = FAISS_HNSW_EF_SEARCH
        return index

    def exists(self) -> bool:
        """Exists.

        Returns:
            Valor de retorno da função.
        """
        return VECTORSTORE_INDEX_PATH.exists() and VECTORSTORE_METADATA_PATH.exists()

    def load(self):
        """Load.

        Returns:
            Valor de retorno da função.
        """
        if not self.exists():
            self.index = None
            self.metadata = []
            self.manifest = {}
            return

        self._ensure_faiss_available()
        self.index = faiss.read_index(str(VECTORSTORE_INDEX_PATH))
        raw_metadata = json.loads(VECTORSTORE_METADATA_PATH.read_text(encoding="utf-8"))
        self.metadata = [ChunkMetadata(**item) for item in raw_metadata]
        if VECTORSTORE_MANIFEST_PATH.exists():
            self.manifest = json.loads(VECTORSTORE_MANIFEST_PATH.read_text(encoding="utf-8"))
        else:
            self.manifest = {}

    def save(self):
        """Save.

        Returns:
            Valor de retorno da função.
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

    def clear(self):
        """Clear.

        Returns:
            Valor de retorno da função.
        """
        self.index = None
        self.metadata = []
        self.manifest = {}
        for path in [VECTORSTORE_INDEX_PATH, VECTORSTORE_METADATA_PATH, VECTORSTORE_MANIFEST_PATH]:
            if path.exists():
                path.unlink()

    def add_embeddings(self, embeddings: list[list[float]], metadata: list[ChunkMetadata]):
        """Add embeddings.

        Args:
            embeddings: Descrição do parâmetro `embeddings`.
            metadata: Descrição do parâmetro `metadata`.

        Returns:
            Valor de retorno da função.
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

    def search(self, query_embedding: list[float], k: int = 4) -> list[tuple[ChunkMetadata, float]]:
        """Search.

        Args:
            query_embedding: Descrição do parâmetro `query_embedding`.
            k: Descrição do parâmetro `k`.

        Returns:
            Valor de retorno da função.
        """
        if self.index is None or not self.metadata:
            return []

        self._ensure_faiss_available()
        vector = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(k, len(self.metadata)))

        results: list[tuple[ChunkMetadata, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.metadata[int(idx)], float(score)))
        return results
