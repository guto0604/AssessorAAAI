import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None

try:
    import streamlit as st
except Exception:
    class _DummyState(dict):
        pass

    class _DummyStreamlit:
        session_state = _DummyState()

    st = _DummyStreamlit()

load_dotenv()

SESSION_TAVILY_KEY = "user_tavily_api_key"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def get_effective_tavily_api_key() -> str | None:
    env_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if env_key:
        return env_key
    session_key = (st.session_state.get(SESSION_TAVILY_KEY, "") or "").strip()
    return session_key or None


def _to_date(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    return url.split("//")[-1].split("/")[0].lower()


def search_tavily(
    query: str,
    *,
    days: int = 7,
    num_results: int = 12,
    include_domains: list[str] | None = None,
) -> list[dict[str, Any]]:
    api_key = get_effective_tavily_api_key()
    if not api_key:
        raise ValueError("TAVILY_API_KEY não configurada.")

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "topic": "news",
        "search_depth": "advanced",
        "max_results": num_results,
        "include_answer": False,
        "include_raw_content": True,
        "include_images": False,
        "days": days,
        "start_date": _to_date(days),
    }

    response = requests.post(
        TAVILY_SEARCH_URL,
        json=payload,
        timeout=40,
    )
    response.raise_for_status()
    body = response.json() or {}
    results = body.get("results", [])

    if include_domains:
        allowed = {domain.lower() for domain in include_domains}
        results = [item for item in results if _extract_domain(item.get("url") or "") in allowed]

    return results
