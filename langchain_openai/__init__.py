from time import perf_counter

from core.openai_client import get_openai_client


class _Resp:
    def __init__(self, content, *, model=None, usage=None, elapsed_ms=None, response_id=None):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.elapsed_ms = elapsed_ms
        self.response_id = response_id


class ChatOpenAI:
    def __init__(self, model: str, temperature: float = 1, model_kwargs=None, api_key=None):
        self.model = model
        self.temperature = temperature
        self.model_kwargs = model_kwargs or {}
        self.api_key = api_key

    def invoke(self, input_value, config=None):
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": input_value,
        }
        if self.model_kwargs.get("response_format"):
            params["response_format"] = self.model_kwargs["response_format"]
        started = perf_counter()
        resp = get_openai_client().chat.completions.create(**params)
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        usage = getattr(resp, "usage", None)
        usage_payload = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        return _Resp(
            resp.choices[0].message.content,
            model=getattr(resp, "model", self.model),
            usage=usage_payload,
            elapsed_ms=elapsed_ms,
            response_id=getattr(resp, "id", None),
        )

    def __or__(self, other):
        from langchain_core.runnables import RunnableSequence

        return RunnableSequence([self, other])
