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

SESSION_EXA_KEY = "user_exa_api_key"
EXA_SEARCH_URL = "https://api.exa.ai/search"


def get_effective_exa_api_key() -> str | None:
    env_key = (os.getenv("EXA_API_KEY") or "").strip()
    if env_key:
        return env_key
    session_key = (st.session_state.get(SESSION_EXA_KEY, "") or "").strip()
    return session_key or None


def _to_iso(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def search_exa(
    query: str,
    *,
    days: int = 7,
    num_results: int = 12,
    include_domains: list[str] | None = None,
    category: str = "news",
) -> list[dict[str, Any]]:
    api_key = get_effective_exa_api_key()
    if not api_key:
        raise ValueError("EXA_API_KEY não configurada.")

    payload: dict[str, Any] = {
        "query": query,
        "type": "keyword",
        "category": category,
        "useAutoprompt": True,
        "numResults": num_results,
        "startPublishedDate": _to_iso(days),
        "text": True,
        "highlights": {"numSentences": 3, "query": query},
    }
    if include_domains:
        payload["includeDomains"] = include_domains

    response = requests.post(
        EXA_SEARCH_URL,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=40,
    )
    response.raise_for_status()
    body = response.json() or {}
    return body.get("results", [])
