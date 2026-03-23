from __future__ import annotations

import json

from core.openai_client import get_effective_openai_api_key, get_openai_client


def has_llm_support() -> bool:
    return bool(get_effective_openai_api_key())


def try_json_completion(*, system_prompt: str, user_prompt: str, model: str, temperature: float) -> dict | None:
    if not has_llm_support():
        return None

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception:
        return None
