from pathlib import Path

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
VECTORSTORE_DIR = Path("data/vectorstore")
VECTORSTORE_INDEX_PATH = VECTORSTORE_DIR / "kb.faiss"
VECTORSTORE_METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
VECTORSTORE_MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"
VECTORSTORE_DOCUMENTS_PATH = VECTORSTORE_DIR / "documents.json"

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
QUERY_PARSER_MODEL = "gpt-5-mini"

RAG_SEGMENT_OPTIONS = [
    "Até 300k",
    "300k-2M",
    "2M+",
]

# HNSW parameters for FAISS ANN retrieval
FAISS_HNSW_M = 32
FAISS_HNSW_EF_CONSTRUCTION = 200
FAISS_HNSW_EF_SEARCH = 64
