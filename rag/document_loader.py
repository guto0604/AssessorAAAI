from io import BytesIO
import hashlib
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

from rag.config import SUPPORTED_EXTENSIONS


class InvalidDocumentError(ValueError):
    pass


def is_supported_file(path: Path) -> bool:
    """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

    Args:
        path: Caminho ou arquivo de entrada relacionado a 'path'.

    Returns:
        Booleano indicando se a condição esperada foi atendida.
    
    """
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def sha256_bytes(content: bytes) -> str:
    """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

    Args:
        content: Valor de entrada necessário para processar 'content'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    return hashlib.sha256(content).hexdigest()


def extract_text_from_bytes(file_name: str, content: bytes) -> str:
    """Executa uma etapa do pipeline RAG para indexação, busca e resposta com contexto.

    Args:
        file_name: Caminho ou arquivo de entrada relacionado a 'file_name'.
        content: Valor de entrada necessário para processar 'content'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise InvalidDocumentError(f"Formato não suportado: {suffix}. Apenas PDF e TXT.")

    if not content:
        raise InvalidDocumentError("Arquivo vazio.")

    if suffix == ".txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="ignore")
    else:
        if PdfReader is None:
            raise InvalidDocumentError("Leitura de PDF indisponível: instale a dependência pypdf.")
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)

    text = (text or "").strip()
    if not text:
        raise InvalidDocumentError("Não foi possível extrair texto do arquivo.")
    return text


def load_text_from_file(path: Path) -> str:
    """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

    Args:
        path: Caminho ou arquivo de entrada relacionado a 'path'.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    if not path.exists():
        raise InvalidDocumentError(f"Arquivo inexistente: {path}")
    content = path.read_bytes()
    return extract_text_from_bytes(path.name, content)
