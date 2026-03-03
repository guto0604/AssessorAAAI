from openai_client import get_openai_client


class _Resp:
    def __init__(self, content):
        self.content = content


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
        resp = get_openai_client().chat.completions.create(**params)
        return _Resp(resp.choices[0].message.content)

    def __or__(self, other):
        from langchain_core.runnables import RunnableSequence

        return RunnableSequence([self, other])
