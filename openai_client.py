import os

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()


SESSION_OPENAI_KEY = "user_openai_api_key"


def has_env_openai_api_key() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def get_effective_openai_api_key() -> str | None:
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key

    session_key = (st.session_state.get(SESSION_OPENAI_KEY, "") or "").strip()
    return session_key or None


def get_openai_client() -> OpenAI:
    api_key = get_effective_openai_api_key()
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()
