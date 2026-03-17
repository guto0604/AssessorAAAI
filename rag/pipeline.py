from dataclasses import dataclass
from pathlib import Path
import math
import re
from time import perf_counter
from typing import Any

from core.openai_client import get_openai_client
from rag.chunking import chunk_text
from rag.config import CHAT_MODEL, EMBEDDING_MODEL, KNOWLEDGE_BASE_DIR, QUERY_PARSER_MODEL
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


    def _parse_query_for_retrieval(self, question: str, include_api_metrics: bool = False):
        parser_prompt = (
            "Você é um parser de consultas para recuperação semântica em assessoria de investimentos. "
            "Reescreva a pergunta do assessor para busca vetorial em português do Brasil, "
            "mantendo intenção original, incluindo termos financeiros e operacionais relevantes, "
            "sinônimos úteis, e contexto de suitability/compliance quando aplicável. "
            "Não responda à pergunta; apenas devolva a consulta expandida em uma única linha."
        )

        llm_started_at = perf_counter()
        response = self.client.chat.completions.create(
            model=QUERY_PARSER_MODEL,
            temperature=1,
            messages=[
                {"role": "system", "content": parser_prompt},
                {"role": "user", "content": question},
            ],
        )
        llm_latency_ms = int((perf_counter() - llm_started_at) * 1000)
        parsed_query = (response.choices[0].message.content or "").strip()
        if not parsed_query:
            parsed_query = question

        if not include_api_metrics:
            return parsed_query

        input_tokens, output_tokens, total_tokens = self._extract_usage(getattr(response, "usage", None))
        parser_metrics = {
            "provider": "openai",
            "step": "query_parser",
            "model": getattr(response, "model", QUERY_PARSER_MODEL),
            "latency_ms": llm_latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "prompt": {"system": parser_prompt, "user": question},
            "output": parsed_query,
        }
        return {"parsed_query": parsed_query, "api_metrics": parser_metrics}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)

    def _bm25_search(self, query: str, k: int = 5) -> list[tuple[ChunkMetadata, float]]:
        if not self.store.metadata:
            return []

        tokenized_docs = [self._tokenize(meta.text) for meta in self.store.metadata]
        doc_count = len(tokenized_docs)
        avg_doc_len = sum(len(doc) for doc in tokenized_docs) / max(doc_count, 1)
        if avg_doc_len == 0:
            return []

        doc_freqs: dict[str, int] = {}
        term_freq_per_doc: list[dict[str, int]] = []
        for doc_tokens in tokenized_docs:
            term_freq: dict[str, int] = {}
            seen = set()
            for token in doc_tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
                if token not in seen:
                    doc_freqs[token] = doc_freqs.get(token, 0) + 1
                    seen.add(token)
            term_freq_per_doc.append(term_freq)

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        k1 = 1.5
        b = 0.75
        scores = [0.0] * doc_count
        for term in query_terms:
            df = doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
            for idx, term_freq in enumerate(term_freq_per_doc):
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue
                doc_len = len(tokenized_docs[idx])
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                scores[idx] += idf * (numerator / denominator)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[tuple[ChunkMetadata, float]] = []
        for idx, score in ranked[: min(k, len(ranked))]:
            if score <= 0:
                continue
            results.append((self.store.metadata[idx], float(score)))
        return results

    def _hybrid_search(
        self,
        parsed_query: str,
        query_embedding: list[float],
        top_k: int = 5,
        semantic_weight: float = 0.8,
        bm25_weight: float = 0.2,
    ) -> list[tuple[ChunkMetadata, float]]:
        retrieval_depth = max(top_k * 4, top_k)
        semantic_results = self.store.search(query_embedding, k=retrieval_depth)
        bm25_results = self._bm25_search(parsed_query, k=retrieval_depth)

        # Weighted Reciprocal Rank Fusion (RRF)
        rrf_constant = 60
        fused_scores: dict[tuple[str, int], dict[str, Any]] = {}

        for rank, (meta, _score) in enumerate(semantic_results, start=1):
            key = (meta.source_path, meta.chunk_id)
            if key not in fused_scores:
                fused_scores[key] = {"meta": meta, "score": 0.0}
            fused_scores[key]["score"] += semantic_weight * (1.0 / (rrf_constant + rank))

        for rank, (meta, _score) in enumerate(bm25_results, start=1):
            key = (meta.source_path, meta.chunk_id)
            if key not in fused_scores:
                fused_scores[key] = {"meta": meta, "score": 0.0}
            fused_scores[key]["score"] += bm25_weight * (1.0 / (rrf_constant + rank))

        ranked = sorted(
            ((item["meta"], float(item["score"])) for item in fused_scores.values()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

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

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        semantic_weight: float = 0.8,
        bm25_weight: float = 0.2,
        include_api_metrics: bool = False,
    ):
        self.ensure_index_exists()

        top_k = max(1, int(top_k))
        semantic_weight = max(0.0, min(1.0, float(semantic_weight)))
        bm25_weight = max(0.0, min(1.0, float(bm25_weight)))
        weights_sum = semantic_weight + bm25_weight
        if weights_sum <= 0:
            semantic_weight, bm25_weight = 0.8, 0.2
        else:
            semantic_weight /= weights_sum
            bm25_weight /= weights_sum

        parsed_query_result = self._parse_query_for_retrieval(question, include_api_metrics=include_api_metrics)
        if include_api_metrics:
            parsed_query = parsed_query_result["parsed_query"]
            query_parser_metrics = parsed_query_result["api_metrics"]
        else:
            parsed_query = parsed_query_result
            query_parser_metrics = None

        query_embedding_result = self._embed_texts([parsed_query], include_api_metrics=include_api_metrics)
        if include_api_metrics:
            query_embedding = query_embedding_result["embeddings"][0]
            query_embedding_metrics = query_embedding_result["api_metrics"]
        else:
            query_embedding = query_embedding_result[0]
            query_embedding_metrics = None

        retrieved = self._hybrid_search(
            parsed_query=parsed_query,
            query_embedding=query_embedding,
            top_k=top_k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

        if not retrieved:
            fallback_answer = "Não encontrei trechos relevantes na base para responder com segurança."
            if include_api_metrics:
                return {
                    "answer": fallback_answer,
                    "sources": [],
                    "api_calls": [call for call in [query_parser_metrics, query_embedding_metrics] if call],
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
            "Você é um assistente de suporte para responder dúvidas de assessores com base em documentos. "
            "Responda SOMENTE com base nos trechos fornecidos. "
            "Se não houver informação suficiente, diga explicitamente que não há base."
        )

        user_message = (
            f"Pergunta original do usuário: {question}\n"
            f"Configuração de recuperação híbrida (RRF): top_k={top_k}, peso_semantico={semantic_weight:.2f}, peso_bm25={bm25_weight:.2f}\n\n"
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
            "prompt": {"system": prompt, "user": user_message},
            "output": answer,
        }
        api_calls = [query_parser_metrics, query_embedding_metrics, completion_metrics]
        return {
            "answer": answer,
            "sources": sources,
            "api_calls": [call for call in api_calls if call],
        }
