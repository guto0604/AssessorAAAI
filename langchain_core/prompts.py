from .runnables import Runnable


class ChatPromptTemplate(Runnable):
    def __init__(self, messages):
        """  init  .

        Args:
            messages: Descrição do parâmetro `messages`.
        """
        self.messages = messages

    @classmethod
    def from_messages(cls, messages):
        """From messages.

        Args:
            messages: Descrição do parâmetro `messages`.

        Returns:
            Valor de retorno da função.
        """
        return cls(messages)

    def invoke(self, input_value, config=None):
        """Invoke.

        Args:
            input_value: Descrição do parâmetro `input_value`.
            config: Descrição do parâmetro `config`.

        Returns:
            Valor de retorno da função.
        """
        rendered = []
        for role, template in self.messages:
            try:
                content = template.format(**input_value)
            except Exception:
                content = template
            rendered.append({"role": role, "content": content})
        return rendered
