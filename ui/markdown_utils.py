import re


_UNESCAPED_DOLLAR_PATTERN = re.compile(r"(?<!\\)\$")


def escape_streamlit_markdown(text: str | None) -> str:
    """Escapa caracteres que o Streamlit pode interpretar como LaTeX/Markdown indevidamente."""
    if text is None:
        return ""
    return _UNESCAPED_DOLLAR_PATTERN.sub(r"\\$", str(text))
