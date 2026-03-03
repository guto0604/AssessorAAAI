from .runnables import Runnable


class ChatPromptTemplate(Runnable):
    def __init__(self, messages):
        self.messages = messages

    @classmethod
    def from_messages(cls, messages):
        return cls(messages)

    def invoke(self, input_value, config=None):
        rendered = []
        for role, template in self.messages:
            try:
                content = template.format(**input_value)
            except Exception:
                content = template
            rendered.append({"role": role, "content": content})
        return rendered
