import os

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
    from openai import OpenAI
except Exception:
    class OpenAI:
        def __init__(self, *args, **kwargs):
            """Inicializa a classe com dependências e estado necessários para o fluxo.

            Args:
                args: Argumentos posicionais repassados para a chamada interna.
                kwargs: Parâmetros adicionais repassados para a chamada interna.
            """
            raise RuntimeError("openai package is required to execute real LLM calls")
try:
    import streamlit as st
except Exception:
    class _DummyState(dict):
        pass
    class _DummyStreamlit:
        session_state = _DummyState()
    st = _DummyStreamlit()

load_dotenv()


SESSION_OPENAI_KEY = "user_openai_api_key"


def has_env_openai_api_key() -> bool:
    """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

    Returns:
        Booleano indicando se a condição esperada foi atendida.
    
    """
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def get_effective_openai_api_key() -> str | None:
    """Obtém a informação necessária para a etapa atual do processo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key

    session_key = (st.session_state.get(SESSION_OPENAI_KEY, "") or "").strip()
    return session_key or None


def get_openai_client() -> OpenAI:
    """Obtém a informação necessária para a etapa atual do processo.

    Returns:
        Dados carregados e prontos para consumo no fluxo da aplicação.
    """
    api_key = get_effective_openai_api_key()
    if api_key:
        return OpenAI(api_key=api_key)
    return OpenAI()
