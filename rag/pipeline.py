from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from core.openai_client import get_openai_client
from rag.chunking import chunk_text
from rag.config import CHAT_MODEL, EMBEDDING_MODEL, KNOWLEDGE_BASE_DIR
from rag.document_loader import (
    InvalidDocumentError,
    extract_text_from_bytes,
    is_supported_file,
    load_text_from_file,
    sha256_bytes,
)
from rag.vector_store import ChunkMetadata, LocalFaissStore


@dataclass
class IngestResult:
    added_files: int
    added_chunks: int
    skipped_files: list[str]


class RagService:
    def __init__(self):
        self.store = LocalFaissStore()
        self.store.load()
        self.client = get_openai_client()

    def _extract_usage(self, usage: Any) -> tuple[int | None, int | None, int | None]:
        if usage is None:
            return None, None, None

        if hasattr(usage, "prompt_tokens"):
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            return input_tokens, output_tokens, total_tokens

        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")
            return input_tokens, output_tokens, total_tokens

        return None, None, None

    def _embed_texts(self, texts: list[str], include_api_metrics: bool = False):
        started_at = perf_counter()
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        latency_ms = int((perf_counter() - started_at) * 1000)
        embeddings = [row.embedding for row in response.data]

        if not include_api_metrics:
            return embeddings

        input_tokens, output_tokens, total_tokens = self._extract_usage(getattr(response, "usage", None))
        return {
            "embeddings": embeddings,
            "api_metrics": {
                "provider": "openai",
                "step": "embedding",
                "model": getattr(response, "model", EMBEDDING_MODEL),
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }

    def ingest_uploaded_file(self, folder: str, file_name: str, content: bytes) -> IngestResult:
        text = extract_text_from_bytes(file_name=file_name, content=content)
        file_hash = sha256_bytes(content)

        destination_dir = KNOWLEDGE_BASE_DIR / folder
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / file_name
        source_key = str(destination_path)

        if self.store.manifest.get(source_key) == file_hash:
            raise InvalidDocumentError("Reupload detectado: este arquivo já foi indexado sem alterações.")

        destination_path.write_bytes(content)
        chunks = chunk_text(text)
        if not chunks:
            raise InvalidDocumentError("Arquivo sem conteúdo útil após chunking.")

        embeddings = self._embed_texts([chunk.text for chunk in chunks])
        metadatas = [
            ChunkMetadata(
                source_path=source_key,
                source_hash=file_hash,
                kb_folder=folder,
                file_name=file_name,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
            )
            for chunk in chunks
        ]

        self.store.add_embeddings(embeddings, metadatas)
        self.store.manifest[source_key] = file_hash
        self.store.save()

        return IngestResult(added_files=1, added_chunks=len(chunks), skipped_files=[])

    def reindex_all_documents(self) -> IngestResult:
        files = [
            path
            for path in KNOWLEDGE_BASE_DIR.rglob("*")
            if path.is_file() and is_supported_file(path)
        ]
        self.store.clear()

        all_embeddings: list[list[float]] = []
        all_metadata: list[ChunkMetadata] = []
        skipped: list[str] = []

        for path in sorted(files):
            try:
                text = load_text_from_file(path)
                chunks = chunk_text(text)
                if not chunks:
                    skipped.append(str(path))
                    continue
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = self._embed_texts(chunk_texts)
                file_hash = sha256_bytes(path.read_bytes())
                folder = path.parent.relative_to(KNOWLEDGE_BASE_DIR).as_posix()

                all_embeddings.extend(embeddings)
                all_metadata.extend(
                    [
                        ChunkMetadata(
                            source_path=str(path),
                            source_hash=file_hash,
                            kb_folder=folder,
                            file_name=path.name,
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                        )
                        for chunk in chunks
                    ]
                )
                self.store.manifest[str(path)] = file_hash
            except Exception:
                skipped.append(str(path))

        self.store.add_embeddings(all_embeddings, all_metadata)
        self.store.save()

        return IngestResult(
            added_files=len(files) - len(skipped),
            added_chunks=len(all_metadata),
            skipped_files=skipped,
        )

    def ensure_index_exists(self):
        if not self.store.exists() or not self.store.metadata:
            self.reindex_all_documents()

    def answer_question(self, question: str, top_k: int = 4, include_api_metrics: bool = False):
        self.ensure_index_exists()

        query_embedding_result = self._embed_texts([question], include_api_metrics=include_api_metrics)
        if include_api_metrics:
            query_embedding = query_embedding_result["embeddings"][0]
            query_embedding_metrics = query_embedding_result["api_metrics"]
        else:
            query_embedding = query_embedding_result[0]
            query_embedding_metrics = None

        retrieved = self.store.search(query_embedding, k=top_k)

        if not retrieved:
            fallback_answer = "Não encontrei trechos relevantes na base para responder com segurança."
            if include_api_metrics:
                return {
                    "answer": fallback_answer,
                    "sources": [],
                    "api_calls": [query_embedding_metrics] if query_embedding_metrics else [],
                }
            return fallback_answer, []

        context_parts = []
        sources = []
        for i, (meta, score) in enumerate(retrieved, start=1):
            context_parts.append(
                f"[Trecho {i}] Fonte: {meta.source_path} | Similaridade: {score:.3f}\n{meta.text}"
            )
            sources.append(
                {
                    "source_path": meta.source_path,
                    "file_name": meta.file_name,
                    "chunk_id": meta.chunk_id,
                    "score": score,
                }
            )

        prompt = (
            "Você é um assistente de suporte operacional. "
            "Responda SOMENTE com base nos trechos fornecidos. "
            "Se não houver informação suficiente, diga explicitamente que não há base."
        )

        user_message = (
            f"Pergunta: {question}\n\n"
            "Trechos recuperados:\n"
            + "\n\n".join(context_parts)
            + "\n\nRegras: não invente informações, cite limitações quando necessário."
        )

        llm_started_at = perf_counter()
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            #temperature=0.1,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
        )
        llm_latency_ms = int((perf_counter() - llm_started_at) * 1000)
        answer = response.choices[0].message.content or "Sem resposta."
        if not include_api_metrics:
            return answer, sources

        input_tokens, output_tokens, total_tokens = self._extract_usage(getattr(response, "usage", None))
        completion_metrics = {
            "provider": "openai",
            "step": "chat_completion",
            "model": getattr(response, "model", CHAT_MODEL),
            "latency_ms": llm_latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        api_calls = [query_embedding_metrics, completion_metrics]
        return {
            "answer": answer,
            "sources": sources,
            "api_calls": [call for call in api_calls if call],
        }
