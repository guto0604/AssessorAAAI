import os
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        """Load dotenv.

        Returns:
            Valor de retorno da função.
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
    """Get effective tavily api key.

    Returns:
        Valor de retorno da função.
    """
    env_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if env_key:
        return env_key
    session_key = (st.session_state.get(SESSION_TAVILY_KEY, "") or "").strip()
    if session_key:
        return session_key
    return None


def _extract_domain(url: str) -> str:
    """ extract domain.

    Args:
        url: Descrição do parâmetro `url`.

    Returns:
        Valor de retorno da função.
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
    """Search tavily.

    Args:
        query: Descrição do parâmetro `query`.
        days: Descrição do parâmetro `days`.
        num_results: Descrição do parâmetro `num_results`.
        include_domains: Descrição do parâmetro `include_domains`.
        lightweight: Descrição do parâmetro `lightweight`.

    Returns:
        Valor de retorno da função.
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
