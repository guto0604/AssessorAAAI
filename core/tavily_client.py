import os
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        """Carrega dados da fonte esperada e devolve a estrutura pronta para uso no fluxo.

        Returns:
            Dados carregados e prontos para consumo no fluxo da aplicação.
        """
        return None

try:
    import streamlit as st
except Exception:
    class _DummyState(dict):
        pass

    class _DummyStreamlit:
        session_state = _DummyState()

    st = _DummyStreamlit()

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

load_dotenv()

SESSION_TAVILY_KEY = "user_tavily_api_key"


def get_effective_tavily_api_key() -> str | None:
    """Obtém a informação necessária para a etapa atual do processo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    env_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if env_key:
        return env_key
    session_key = (st.session_state.get(SESSION_TAVILY_KEY, "") or "").strip()
    if session_key:
        return session_key
    return None


def _extract_domain(url: str) -> str:
    """Responsável por extrair domain no contexto da aplicação de assessoria.

    Args:
        url: Valor de entrada necessário para processar 'url'.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    
    """
    if not url:
        return ""
    return url.split("//")[-1].split("/")[0].lower()


def search_tavily(
    query: str,
    *,
    days: int = 7,
    num_results: int = 12,
    include_domains: list[str] | None = None,
    lightweight: bool = False,
) -> list[dict[str, Any]]:
    """Realiza a busca de informações com os filtros definidos para o contexto atual.

    Args:
        query: Consulta usada para buscar dados ou informações externas.
        days: Valor de entrada necessário para processar 'days'.
        num_results: Quantidade máxima de resultados retornados pela busca.
        include_domains: Valor de entrada necessário para processar 'include_domains'.
        lightweight: Se verdadeiro, reduz payload para uma busca mais leve.

    Returns:
        Resultado da rotina, no tipo esperado pelo fluxo chamador.
    """
    api_key = get_effective_tavily_api_key()
    if not api_key:
        raise ValueError("TAVILY_API_KEY não configurada.")

    payload: dict[str, Any] = {
        "query": query,
        "topic": "news",
        "search_depth": "advanced",
        "max_results": num_results,
        "include_answer": False,
        "include_raw_content": not lightweight,
        "include_images": False,
        "days": days,
    }

    if TavilyClient is None:
        raise ImportError("Pacote 'tavily-python' não instalado. Adicione ao requirements para usar a busca Tavily.")

    client = TavilyClient(api_key=api_key)
    try:
        body = client.search(**payload) or {}
    except Exception as exc:
        # Fallback para reduzir payload e melhorar chance de resposta em cenários de timeout.
        fallback_payload = {
            **payload,
            "search_depth": "basic",
            "include_raw_content": False,
            "max_results": max(3, min(num_results, 6)),
        }
        error_msg = str(exc).lower()
        is_timeout_like = any(term in error_msg for term in ["timed out", "timeout", "read timed out"])
        if not is_timeout_like:
            raise
        body = client.search(**fallback_payload) or {}

    results = body.get("results", [])

    if include_domains:
        allowed = {domain.lower() for domain in include_domains}
        results = [item for item in results if _extract_domain(item.get("url") or "") in allowed]

    return results
