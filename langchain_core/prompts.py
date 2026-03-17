from .runnables import Runnable


class ChatPromptTemplate(Runnable):
    def __init__(self, messages):
        """Inicializa a classe com dependências e estado necessários para o fluxo.

        Args:
            messages: Valor de entrada necessário para processar 'messages'.
        """
        self.messages = messages

    @classmethod
    def from_messages(cls, messages):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            messages: Valor de entrada necessário para processar 'messages'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        return cls(messages)

    def invoke(self, input_value, config=None):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            input_value: Valor de entrada necessário para processar 'input_value'.
            config: Valor de entrada necessário para processar 'config'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        rendered = []
        for role, template in self.messages:
            try:
                content = template.format(**input_value)
            except Exception:
                content = template
            rendered.append({"role": role, "content": content})
        return rendered
