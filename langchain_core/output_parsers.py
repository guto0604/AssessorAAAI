from .runnables import Runnable


class StrOutputParser(Runnable):
    def invoke(self, input_value, config=None):
        """Configura integração com provedores externos usados nos fluxos de IA da aplicação.

        Args:
            input_value: Valor de entrada necessário para processar 'input_value'.
            config: Valor de entrada necessário para processar 'config'.

        Returns:
            Resultado da rotina, no tipo esperado pelo fluxo chamador.
        
        """
        if isinstance(input_value, str):
            return input_value
        return getattr(input_value, "content", str(input_value))
