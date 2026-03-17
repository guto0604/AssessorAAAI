from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_id: int


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Chunk]:
    """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

    Args:
        text: Texto de entrada a ser processado pela função.
        chunk_size: Valor de entrada necessário para processar 'chunk_size'.
        chunk_overlap: Valor de entrada necessário para processar 'chunk_overlap'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        piece = clean[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, chunk_id=chunk_id))
            chunk_id += 1
        if end == len(clean):
            break
        start = max(0, end - chunk_overlap)
    return chunks
