from pathlib import Path

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
VECTORSTORE_DIR = Path("data/vectorstore")
VECTORSTORE_INDEX_PATH = VECTORSTORE_DIR / "kb.faiss"
VECTORSTORE_METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
VECTORSTORE_MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
