from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from openai import OpenAI
from pypdf import PdfReader

from openai_client import get_effective_openai_api_key

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / ".vector_store"
COLLECTION_NAME = "assessor_kb"
ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict


class RagPipelineError(Exception):
    pass


class InvalidFileError(RagPipelineError):
    pass


class EmptyFileError(RagPipelineError):
    pass


class DuplicateFileError(RagPipelineError):
    pass


class VectorKnowledgeBase:
    def __init__(self, kb_dir: Path = KNOWLEDGE_BASE_DIR, vector_dir: Path = VECTOR_DB_DIR):
        self.kb_dir = kb_dir
        self.vector_dir = vector_dir
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.vector_dir))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def _get_openai_client(self) -> OpenAI:
        api_key = get_effective_openai_api_key()
        if not api_key:
            raise RagPipelineError(
                "Defina uma OPENAI_API_KEY no ambiente ou em Configurações para usar a aba Pergunte à IA."
            )
        return OpenAI(api_key=api_key)

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_openai_client()
        response = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in response.data]

    def _extract_text(self, file_name: str, file_bytes: bytes) -> str:
        ext = Path(file_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidFileError("Formato inválido. Apenas arquivos PDF e TXT são suportados.")

        if not file_bytes:
            raise EmptyFileError("Arquivo vazio.")

        if ext == ".txt":
            for encoding in ("utf-8", "latin-1"):
                try:
                    return file_bytes.decode(encoding).strip()
                except UnicodeDecodeError:
                    continue
            raise InvalidFileError("Não foi possível decodificar o TXT enviado.")

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        return text

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(start + chunk_size, len(cleaned))
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(cleaned):
                break
            start = max(end - overlap, 0)
        return chunks

    def _get_existing_file_hashes(self) -> set[str]:
        data = self.collection.get(include=["metadatas"])
        hashes = set()
        for metadata in data.get("metadatas", []):
            if metadata and metadata.get("file_hash"):
                hashes.add(str(metadata["file_hash"]))
        return hashes

    def _build_chunk_records(
        self,
        rel_path: str,
        folder: str,
        file_name: str,
        content_hash: str,
        chunks: Iterable[str],
    ) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        for idx, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    chunk_id=f"{content_hash}:{idx}",
                    text=chunk,
                    metadata={
                        "source": rel_path,
                        "source_path": rel_path,
                        "folder": folder,
                        "file_name": file_name,
                        "file_hash": content_hash,
                        "chunk_index": idx,
                    },
                )
            )
        return records

    def index_document(self, file_path: Path) -> int:
        rel_path = str(file_path.relative_to(self.kb_dir))
        folder = file_path.parent.relative_to(self.kb_dir).as_posix()
        file_bytes = file_path.read_bytes()
        text = self._extract_text(file_path.name, file_bytes)
        if not text:
            raise EmptyFileError("Arquivo sem conteúdo textual utilizável.")

        content_hash = self._sha256(file_bytes)
        if content_hash in self._get_existing_file_hashes():
            raise DuplicateFileError("Este arquivo já existe na base vetorial (mesmo conteúdo).")

        chunks = self._chunk_text(text)
        if not chunks:
            raise EmptyFileError("Não foi possível gerar chunks para indexação.")

        records = self._build_chunk_records(rel_path, folder, file_path.name, content_hash, chunks)

        self.collection.delete(where={"source_path": rel_path})
        embeddings = self._embed_texts([record.text for record in records])
        self.collection.add(
            ids=[record.chunk_id for record in records],
            documents=[record.text for record in records],
            metadatas=[record.metadata for record in records],
            embeddings=embeddings,
        )
        return len(records)

    def sync_existing_documents(self) -> tuple[int, list[str]]:
        indexed = 0
        errors: list[str] = []
        for file_path in sorted(self.kb_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            try:
                indexed += self.index_document(file_path)
            except DuplicateFileError:
                continue
            except RagPipelineError as exc:
                errors.append(f"{file_path.name}: {exc}")
        return indexed, errors

    def save_upload_and_index(self, file_name: str, file_bytes: bytes, target_folder: str) -> tuple[Path, int]:
        ext = Path(file_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidFileError("Formato inválido. Faça upload apenas de PDF ou TXT.")

        folder = target_folder.strip().strip("/") or "uploads"
        destination_dir = self.kb_dir / folder
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_path = destination_dir / file_name
        destination_path.write_bytes(file_bytes)

        try:
            chunk_count = self.index_document(destination_path)
        except Exception:
            destination_path.unlink(missing_ok=True)
            raise

        return destination_path, chunk_count

    def ask(self, question: str, top_k: int = 4) -> dict:
        if not question.strip():
            raise RagPipelineError("Digite uma pergunta antes de enviar.")

        query_embedding = self._embed_texts([question])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        if not documents:
            raise RagPipelineError("Nenhum trecho relevante foi encontrado na knowledge base.")

        context_blocks = []
        sources = []
        for idx, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            source_path = metadata.get("source_path", "desconhecido") if metadata else "desconhecido"
            context_blocks.append(f"[Fonte {idx}: {source_path}]\n{doc}")
            sources.append(
                {
                    "source": source_path,
                    "distance": float(distance) if distance is not None else None,
                    "snippet": doc[:220] + ("..." if len(doc) > 220 else ""),
                }
            )

        context_text = "\n\n".join(context_blocks)
        prompt = (
            "Você é um assistente que responde SOMENTE com base no contexto recuperado. "
            "Se a resposta não estiver no contexto, diga explicitamente que não há informação suficiente. "
            "Não invente fatos. Sempre cite as fontes usadas no formato [Fonte X].\n\n"
            f"Pergunta do usuário:\n{question}\n\n"
            f"Contexto recuperado:\n{context_text}"
        )

        client = self._get_openai_client()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "Responda em português de forma objetiva e fiel ao contexto."},
                {"role": "user", "content": prompt},
            ],
        )

        answer = completion.choices[0].message.content or ""
        return {"answer": answer.strip(), "sources": sources}
